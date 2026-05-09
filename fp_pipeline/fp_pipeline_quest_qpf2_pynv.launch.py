#!/usr/bin/env python3
"""
fp_pipeline_quest_qpf2_pynv.launch.py

PyNITROS-accelerated QPF2 pipeline.

    bag_to_qpf2.py / Quest device
        ↓ TCP/QPF2 (port 7040)
    qpf2_receiver_pynv.py
        ├─ NVDEC decode (PyNvVideoCodec, GPU-resident)
        ├─ DLPack → torch.cuda → resize 1920×1080 → 960×576
        └─ PyNITROS publish on /pynitros_left, /pynitros_right
                              ↓ CUDA IPC (no D→H)
    [ESS container, component_container_mt]:
        ├─ ImageConverterNode (left)  → /left/image_rect (NitrosImage)
        ├─ ImageConverterNode (right) → /right/image_rect (NitrosImage)
        └─ ESSDisparityNode            → /disparity
                                       ↓
    stereo_depth_saver.py → /left/depth_image
                                       ↓
    FoundationPose → /tracking/pose_matrix_output

CameraInfo and segmentation flow over standard DDS (small payloads).

Pre-requisites
--------------
1. CUDA IPC permissions: `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`
2. PyNITROS installed: ros-jazzy-isaac-ros-pynitros
3. Bridge installed: ros-jazzy-isaac-ros-nitros-bridge-ros2

Args (same as the standard QPF2 launch)
    compressed_bag:=/path/to/h264_bag        (required)
    mask_path:=/path/to/mask.png             (required)
    play_rate:=1.0
    play_loop:=false
    enable_sim_source:=true
    enable_fp:=true
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


HOME = Path(os.path.expanduser("~"))
FP_PIPELINE_DIR = HOME / "fp_pipeline"
ISAAC_ROS_WS = Path(os.environ.get(
    "ISAAC_ROS_WS",
    str(HOME / "workspaces" / "isaac_ros-dev")))

GATE_PATH = "/tmp/qpf2_start"


def generate_launch_description():
    # ── Args ────────────────────────────────────────────────────────
    compressed_bag_arg = DeclareLaunchArgument("compressed_bag")
    mask_path_arg      = DeclareLaunchArgument("mask_path")
    play_rate_arg      = DeclareLaunchArgument("play_rate", default_value="1.0")
    play_loop_arg      = DeclareLaunchArgument(
        "play_loop", default_value="false", choices=["true", "false"])
    qpf2_port_arg      = DeclareLaunchArgument(
        "qpf2_port", default_value="7040")
    enable_sim_arg     = DeclareLaunchArgument(
        "enable_sim_source", default_value="true",
        choices=["true", "false"])
    enable_fp_arg      = DeclareLaunchArgument(
        "enable_fp", default_value="true",
        choices=["true", "false"])

    left_yaml_arg  = DeclareLaunchArgument(
        "left_camera_yaml",
        default_value=str(HOME / "experiments/25-10-09-experiment/pose1"
                          / "camera.left.image.camera_info_4.yaml"))
    right_yaml_arg = DeclareLaunchArgument(
        "right_camera_yaml",
        default_value=str(HOME / "experiments/25-10-09-experiment/pose1"
                          / "camera.right.image.camera_info_4.yaml"))

    ess_engine_arg = DeclareLaunchArgument(
        "ess_engine_file_path",
        default_value=str(
            ISAAC_ROS_WS / "isaac_ros_assets/models/dnn_stereo_disparity"
            / "dnn_stereo_disparity_v4.1.0_onnx_trt10.13/ess.engine"))
    ess_threshold_arg = DeclareLaunchArgument(
        "ess_threshold", default_value="0.0")

    # ── ESS + bridge nodes in the SAME container (zero-copy) ────────
    bridge_left = ComposableNode(
        name="pynitros_bridge_left",
        package="isaac_ros_nitros_bridge_ros2",
        plugin="nvidia::isaac_ros::nitros_bridge::ImageConverterNode",
        remappings=[
            ("ros2_input_bridge_image", "pynitros_left"),
            ("ros2_output_image",       "/left/image_rect"),
        ],
    )
    bridge_right = ComposableNode(
        name="pynitros_bridge_right",
        package="isaac_ros_nitros_bridge_ros2",
        plugin="nvidia::isaac_ros::nitros_bridge::ImageConverterNode",
        remappings=[
            ("ros2_input_bridge_image", "pynitros_right"),
            ("ros2_output_image",       "/right/image_rect"),
        ],
    )
    ess_node = ComposableNode(
        name="disparity",
        package="isaac_ros_ess",
        plugin="nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode",
        parameters=[{
            "engine_file_path":   LaunchConfiguration("ess_engine_file_path"),
            "threshold":          LaunchConfiguration("ess_threshold"),
            "input_layer_width":  960,
            "input_layer_height": 576,
        }],
    )
    ess_container = ComposableNodeContainer(
        name="pynv_ess_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[bridge_left, bridge_right, ess_node],
        output="screen",
    )

    # ── FoundationPose ──────────────────────────────────────────────
    fp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(FP_PIPELINE_DIR / "isaac_ros_foundationpose_med7.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("enable_fp")),
    )

    # ── Receiver ────────────────────────────────────────────────────
    receiver = ExecuteProcess(
        cmd=[
            "python", str(FP_PIPELINE_DIR / "qpf2_receiver_pynv.py"),
            "--port", LaunchConfiguration("qpf2_port"),
            "--left_yaml",  LaunchConfiguration("left_camera_yaml"),
            "--right_yaml", LaunchConfiguration("right_camera_yaml"),
            "--mask_path",  LaunchConfiguration("mask_path"),
            "--start_signal_file", GATE_PATH,
        ],
        output="screen",
    )

    # ── Disparity → depth ───────────────────────────────────────────
    depth_saver = ExecuteProcess(
        cmd=["python", str(FP_PIPELINE_DIR / "stereo_depth_saver.py"),
             "--backend", "ess"],
        output="screen",
    )

    # ── Sim source ──────────────────────────────────────────────────
    bag_sender = ExecuteProcess(
        cmd=[
            "python", str(FP_PIPELINE_DIR / "bag_to_qpf2.py"),
            "--bag",  LaunchConfiguration("compressed_bag"),
            "--host", "127.0.0.1",
            "--port", LaunchConfiguration("qpf2_port"),
            "--rate", LaunchConfiguration("play_rate"),
            "--loop", LaunchConfiguration("play_loop"),
            "--start_signal_file", GATE_PATH,
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_sim_source")),
    )

    # ── Staging ─────────────────────────────────────────────────────
    # Stage 1 (t=0):  ESS+bridge container, FP launch
    # Stage 2 (+4s):  receiver + depth_saver
    # Stage 3 (+9s):  bag sender (waits for gate)
    stage2 = TimerAction(period=4.0, actions=[
        LogInfo(msg="[stage2] starting receiver_pynv + depth_saver"),
        receiver,
        depth_saver,
    ])
    stage3 = TimerAction(period=9.0, actions=[
        LogInfo(msg=f"[stage3] sender ready. `touch {GATE_PATH}` to start."),
        bag_sender,
    ])

    return LaunchDescription([
        compressed_bag_arg, mask_path_arg, play_rate_arg, play_loop_arg,
        qpf2_port_arg, enable_sim_arg, enable_fp_arg,
        left_yaml_arg, right_yaml_arg,
        ess_engine_arg, ess_threshold_arg,
        LogInfo(msg=["PyNITROS-accelerated QPF2 pipeline. bag=",
                     LaunchConfiguration("compressed_bag")]),
        ess_container,
        fp_launch,
        stage2,
        stage3,
    ])
