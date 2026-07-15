#!/usr/bin/env python3
"""fs_seed.launch.py — minimal FoundationStereo graph for the seed path.

Composes ONLY the stages our stream needs, taken verbatim from the packaged
isaac_ros_foundationstereo.launch.py but with the geometric no-op stages
removed: our rectifier already delivers rectified rgb8 at exactly the model's
576x960, so rectify/resize/pad add nothing except three extra stamp-sync
surfaces (which is precisely where the packaged graph was failing on our
stream). Inputs are remapped first-class onto the pipeline's real topics —
no relay, no SetRemap.

A NitrosCameraDropNode (stereo mode) decimates the whole input quad —
both images AND both camera_infos, stamp-coherently — before the graph.
Without it the decoder never publishes: its tensor<->camera_info join is a
hard ExactTime sync, and at the rectifier's full 15 Hz the camera_info for
stamp T is evicted from that sync's queue long before FoundationStereo's
(hundreds of ms) disparity tensor for T emerges — every pair drops with
"message is missing in unsynchronized pair". Decimated to ~2 Hz the match
survives for seconds, and TensorRT stops queueing behind input it can't
keep up with. The seed path doesn't care about depth frame rate.

    inputs:  <input_left>/<input_right>            (rgb8, 960x576, rectified)
             <input_right_camera_info>             (rect camera info, P with Tx)
    output:  /fs/disparity                         (stereo_msgs/DisparityImage)

Args: engine_file_path, input_left, input_right, input_right_camera_info,
      drop_x/drop_y (keep (Y-X) of every Y input frames; default 7/8 ≈ 2 Hz
      from the rectifier's 15 Hz).
"""
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

W, H, C = 960, 576, 3


def generate_launch_description():
    args = [
        DeclareLaunchArgument('engine_file_path'),
        DeclareLaunchArgument('input_left', default_value='/left/image_rect'),
        DeclareLaunchArgument('input_right', default_value='/right/image_rect'),
        DeclareLaunchArgument('input_left_camera_info',
                              default_value='/left/camera_info_rect'),
        DeclareLaunchArgument('input_right_camera_info',
                              default_value='/right/camera_info_rect'),
        DeclareLaunchArgument('drop_x', default_value='7'),
        DeclareLaunchArgument('drop_y', default_value='8'),
    ]

    # Input decimator: drops X of every Y frames, all four topics in
    # lockstep (its internal quad sync is ExactTime, which our stream
    # satisfies — the producer stamps both eyes identically and the
    # rectifier copies the image stamp onto camera_info).
    drop = ComposableNode(
        name='input_drop_node',
        namespace='fs',
        package='isaac_ros_nitros_topic_tools',
        plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
        parameters=[{'mode': 'stereo',
                     'X': LaunchConfiguration('drop_x'),
                     'Y': LaunchConfiguration('drop_y')}],
        remappings=[
            ('image_1',            LaunchConfiguration('input_left')),
            ('camera_info_1',
             LaunchConfiguration('input_left_camera_info')),
            ('image_2',            LaunchConfiguration('input_right')),
            ('camera_info_2',
             LaunchConfiguration('input_right_camera_info')),
            ('image_1_drop',       'left/image_drop'),
            ('camera_info_1_drop', 'left/camera_info_drop'),
            ('image_2_drop',       'right/image_drop'),
            ('camera_info_2_drop', 'right/camera_info_drop'),
        ])

    def eye(side, src):
        return [
            ComposableNode(
                name=f'{side}_format_node',
                namespace='fs',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                parameters=[{'image_width': W, 'image_height': H,
                             'encoding_desired': 'rgb8'}],
                remappings=[('image_raw', src),
                            ('image', f'{side}/image_rgb')]),
            ComposableNode(
                name=f'{side}_normalize_node',
                namespace='fs',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
                parameters=[{'mean': [123.675, 116.28, 103.53],
                             'stddev': [58.395, 57.12, 57.375]}],
                remappings=[('image', f'{side}/image_rgb'),
                            ('normalized_image', f'{side}/image_normalize')]),
            ComposableNode(
                name=f'{side}_tensor_node',
                namespace='fs',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
                parameters=[{'scale': False, 'tensor_name': f'{side}_image'}],
                remappings=[('image', f'{side}/image_normalize'),
                            ('tensor', f'{side}/tensor')]),
            ComposableNode(
                name=f'{side}_planar_node',
                namespace='fs',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
                parameters=[{'input_tensor_shape': [H, W, C],
                             'output_tensor_name': f'{side}_image'}],
                remappings=[('interleaved_tensor', f'{side}/tensor'),
                            ('planar_tensor', f'{side}/tensor_planar')]),
            ComposableNode(
                name=f'{side}_reshape_node',
                namespace='fs',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
                parameters=[{'output_tensor_name': f'{side}_image',
                             'input_tensor_shape': [C, H, W],
                             'output_tensor_shape': [1, C, H, W]}],
                remappings=[('tensor', f'{side}/tensor_planar'),
                            ('reshaped_tensor', f'{side}/tensor_reshape')]),
        ]

    sync = ComposableNode(
        name='tensor_pair_sync_node',
        namespace='fs',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        parameters=[{'input_tensor1_name': 'left_image',
                     'input_tensor2_name': 'right_image',
                     'output_tensor1_name': 'left_image',
                     'output_tensor2_name': 'right_image'}],
        remappings=[('tensor1', 'left/tensor_reshape'),
                    ('tensor2', 'right/tensor_reshape')])

    trt = ComposableNode(
        name='tensor_rt',
        namespace='fs',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{'engine_file_path':
                         LaunchConfiguration('engine_file_path'),
                     'model_file_path': '',
                     'input_tensor_names': ['left_image', 'right_image'],
                     'input_binding_names': ['left_image', 'right_image'],
                     'output_tensor_names': ['disparity'],
                     'output_binding_names': ['disparity'],
                     'verbose': False,
                     'force_engine_update': False}])

    decoder = ComposableNode(
        name='foundationstereo_decoder',
        namespace='fs',
        package='isaac_ros_foundationstereo',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::FoundationStereoDecoderNode',
        parameters=[{'disparity_tensor_name': 'disparity'}],
        remappings=[
            # BOTH camera_info inputs from the DECIMATED stream (relative
            # names resolve under /fs) — same ~2 Hz cadence as the tensors,
            # so the decoder's ExactTime tensor<->ci sync can actually pair
            # them; fed from the raw 15 Hz topics the ci is evicted before
            # the slow disparity tensor arrives and nothing ever publishes.
            ('left/camera_info',  'left/camera_info_drop'),
            ('right/camera_info', 'right/camera_info_drop')])

    container = ComposableNodeContainer(
        name='fs_seed_container',
        namespace='fs',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=(
            [drop]
            + eye('left', 'left/image_drop')
            + eye('right', 'right/image_drop')
            + [sync, trt, decoder]),
        output='screen')

    return launch.LaunchDescription(args + [container])