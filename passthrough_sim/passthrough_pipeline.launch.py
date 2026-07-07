#!/usr/bin/env python3
"""
passthrough_pipeline.launch.py

Clean launch for the "passthrough on ROS topics" simulation. It assumes the
producer publishes H.264 CompressedImage straight to ROS topics
(passthrough_cloudxr_producer.py --image-codec h264 --ros-publish...), i.e. it
reproduces the real deployment boundary where the headset receiver has
already put the H.264 stream on ROS topics.

Pipeline (orchestrated here):

    bake_node                 (bake the URDF mesh once joints arrive)
      └─ on exit ─► ESS + FoundationPose come up
                    └─ +4s ─► passthrough_rectifier + ESS depth_saver
                    └─ +6s ─► fp_pose_recorder (init_only)

    passthrough_rectifier:  /xr/image_*/compressed (H.264)
                              → NVDEC decode → NV12-domain rectify 960x576
                              → /left|right/image_rect (+ camera_info_rect)

There is no legacy TCP server, no transcoder, and no mode switches beyond
RAW_IMAGES=1 for the raw test path. For the FoundationPose-vs-IPCAI comparison set
enable_fp:=true (default); to benchmark only the IPCAI consumer through the
H.264 path you can set enable_fp:=false enable_ess:=false to free the GPU.

All scripts this launch invokes (rectifier, bake_node + bake_lib,
stereo_depth_saver, fp_pose_recorder, and the FoundationPose graph wrapper)
live in THIS folder. The only external pieces are the NVIDIA
isaac_ros_foundationpose / isaac_ros_ess packages and their TensorRT engine
assets, referenced by path.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo,
                            RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

HOME = Path(os.path.expanduser("~"))

# Every script this launch invokes lives in THIS folder --- the pipeline is
# self-contained. (FoundationPose + ESS themselves remain external NVIDIA
# isaac_ros packages; only their thin launch wrapper + the .plan/.engine
# asset paths are referenced.)
PASSTHROUGH_DIR = Path(__file__).resolve().parent
ISAAC_ROS_WS = Path(os.environ.get(
    "ISAAC_ROS_WS", str(HOME / "workspaces" / "isaac_ros-dev")))

PID_DIR = "/tmp/fp_init_pids"
BAKE_DIR = "/tmp/fp_bake_runtime"


def _clean_runtime_dirs(_context):
    import shutil
    for d in (PID_DIR, BAKE_DIR):
        p = Path(d)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
    return []


def _start_recorder(context):
    """Build the `ros2 bag record` action at launch time (so the output dir
    gets a fresh timestamp). Returns [] when enable_bag != true."""
    import datetime
    if context.perform_substitution(
            LaunchConfiguration("enable_bag")) != "true":
        return []
    parent = context.perform_substitution(LaunchConfiguration("bag_dir"))
    storage = context.perform_substitution(LaunchConfiguration("bag_storage"))
    topics = [t for t in context.perform_substitution(
        LaunchConfiguration("bag_topics")).split() if t]
    if not topics:
        return [LogInfo(msg="[bag] enable_bag=true but bag_topics empty "
                            "— not recording")]
    os.makedirs(parent, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(parent, f"passthrough_{stamp}")
    recorder = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-s", storage, "-o", out, *topics],
        output="screen",
    )
    return [
        LogInfo(msg=f"[bag] recording {len(topics)} topic(s) → {out}"),
        recorder,
    ]


def generate_launch_description():
    # ── Args ─────────────────────────────────────────────────────────
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value=str(HOME / "roboreg/test/assets/lbr_med7_r800"
                          / "description/lbr_med7_r800.urdf"),
        description="URDF baked into the FoundationPose mesh.")
    joint_topic_arg = DeclareLaunchArgument(
        "joint_topic", default_value="/xr/joint_states",
        description="Joint states (from the producer bridge) used by bake.")
    joint_prefix_arg = DeclareLaunchArgument(
        "joint_name_prefix", default_value="lbr_A",
        description="Prefix filter on joint_state.name for the bake FK "
                    "(URDF actuated joints are lbr_A1..lbr_A7). Empty = all.")

    # Rectifier inputs (the receiver's H.264 output + source intrinsics).
    left_image_arg = DeclareLaunchArgument(
        "left_image_topic", default_value="/xr/image_left/compressed")
    right_image_arg = DeclareLaunchArgument(
        "right_image_topic", default_value="/xr/image_right/compressed")
    left_caminfo_arg = DeclareLaunchArgument(
        "left_camera_info_topic", default_value="/xr/image_left/camera_info")
    right_caminfo_arg = DeclareLaunchArgument(
        "right_camera_info_topic", default_value="/xr/image_right/camera_info")
    extrinsics_arg = DeclareLaunchArgument(
        "stereo_extrinsics_topic", default_value="/xr/baseline")

    # ESS / FP toggles + assets.
    enable_ess_arg = DeclareLaunchArgument(
        "enable_ess", default_value="true", choices=["true", "false"],
        description="Run ESS stereo depth (ess_launch + depth_saver).")
    enable_fp_arg = DeclareLaunchArgument(
        "enable_fp", default_value="true", choices=["true", "false"],
        description="Run FoundationPose (the IPCAI baseline / init seeder).")
    enable_recorder_arg = DeclareLaunchArgument(
        "enable_recorder", default_value="true", choices=["true", "false"],
        description="Run fp_pose_recorder (init_only) → latched /pose_init.")
    ess_engine_arg = DeclareLaunchArgument(
        "ess_engine_file_path",
        default_value=str(
            ISAAC_ROS_WS / "isaac_ros_assets/models/dnn_stereo_disparity"
            / "dnn_stereo_disparity_v4.1.0_onnx_trt10.13/ess.engine"))
    ess_threshold_arg = DeclareLaunchArgument(
        "ess_threshold", default_value="0.0")
    bake_timeout_arg = DeclareLaunchArgument(
        "bake_timeout", default_value="60.0",
        description="Seconds bake_node waits for the first joint_state "
                    "before giving up. Joints only flow once the producer "
                    "is publishing (press A), so allow margin.")

    # ── Bag recording (opt-in) ──────────────────────────────────────
    # Records the /xr topics so they can be replayed offline into the WebXR
    # viewer over rosbridge (the headset can't run CloudXR and the browser at
    # once, so we capture here and replay there). Off by default; enable with
    # enable_bag:=true. Recording starts with the launch and runs in parallel
    # with bake, so it captures joints from the moment the producer publishes.
    enable_bag_arg = DeclareLaunchArgument(
        "enable_bag", default_value="false", choices=["true", "false"],
        description="Record bag_topics to a rosbag for offline WebXR replay.")
    bag_dir_arg = DeclareLaunchArgument(
        "bag_dir", default_value=str(HOME / "roboreg_bags"),
        description="Parent dir for bags; a timestamped subdir "
                    "passthrough_YYYYmmdd_HHMMSS is created inside.")
    bag_topics_arg = DeclareLaunchArgument(
        "bag_topics",
        default_value=("/xr/extrinsic_left /xr/extrinsic_right "
                       "/xr/joint_states /xr/baseline "
                       "/xr/pose_left /xr/pose_right /xr/pose_base"),
        description="Space-separated topics to record. Default = the pose/"
                    "joint set the WebXR viewer needs (no images). Append "
                    "/xr/image_*/compressed + camera_info to also replay the "
                    "FP/ESS pipeline from the bag.")
    bag_storage_arg = DeclareLaunchArgument(
        "bag_storage", default_value="sqlite3", choices=["sqlite3", "mcap"],
        description="rosbag2 storage backend (mcap needs the mcap plugin).")

    # ── bake (runs first; waits for joints, then bakes the mesh) ─────
    bake_node = ExecuteProcess(
        cmd=[
            "python", str(PASSTHROUGH_DIR / "bake_node.py"),
            "--urdf_path",         LaunchConfiguration("urdf_path"),
            "--out_dir",           BAKE_DIR,
            "--joint_state_topic", LaunchConfiguration("joint_topic"),
            "--joint_name_prefix", LaunchConfiguration("joint_name_prefix"),
            "--timeout",           LaunchConfiguration("bake_timeout"),
        ],
        output="screen",
    )

    # ── ESS + FoundationPose (start when bake exits) ─────────────────
    ess_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("isaac_ros_ess"),
                "launch", "isaac_ros_ess.launch.py"])
        ]),
        launch_arguments={
            "engine_file_path": LaunchConfiguration("ess_engine_file_path"),
            "threshold":        LaunchConfiguration("ess_threshold"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_ess")),
    )
    fp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(PASSTHROUGH_DIR / "isaac_ros_foundationpose_med7.launch.py")),
        launch_arguments={
            "mesh_file_path": f"{BAKE_DIR}/lbr_med7_baked.obj",
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_fp")),
    )

    # ── The rectifier ────────────────────────────────────────────────
    # RAW_IMAGES=1 in the launch environment switches the rectifier to raw
    # rgb8 Image input (matches the producer's --raw-images). --raw-input
    # auto-strips the trailing '/compressed' from the topic args, so the
    # defaults below resolve to /xr/image_left and /xr/image_right.
    # Otherwise the compressed path is H.264 (NVDEC, NV12-domain rectify) —
    # the only codec now; the JPEG/nvJPEG path was removed, so the old
    # IMAGE_CODEC env switch is gone.
    _raw_input = os.environ.get("RAW_IMAGES", "").startswith("1")
    rectifier = ExecuteProcess(
        cmd=[
            "python", str(PASSTHROUGH_DIR / "passthrough_rectifier.py"),
            *(["--raw-input"] if _raw_input else []),
            "--left-image-topic",    LaunchConfiguration("left_image_topic"),
            "--right-image-topic",   LaunchConfiguration("right_image_topic"),
            "--left-caminfo-topic",  LaunchConfiguration("left_camera_info_topic"),
            "--right-caminfo-topic", LaunchConfiguration("right_camera_info_topic"),
            "--extrinsics-topic",    LaunchConfiguration("stereo_extrinsics_topic"),
        ],
        output="screen",
    )

    # ── ESS depth republisher + FP init recorder ────────────────────
    depth_saver = ExecuteProcess(
        cmd=["python", str(PASSTHROUGH_DIR / "stereo_depth_saver.py"),
             "--backend", "ess",
             "--pid_file", f"{PID_DIR}/depth_saver.pid"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_ess")),
    )
    pose_recorder = ExecuteProcess(
        cmd=["python", "-u", str(PASSTHROUGH_DIR / "fp_pose_recorder.py"),
             "--init_only",
             "--offset_npy", f"{BAKE_DIR}/lbr_med7_baked_offset.npy"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_recorder")),
    )

    # ── Staging: bake → (ONLY if bake produced its outputs) ESS+FP →
    #    +4s rectifier+depth → +6s recorder ───────────────────────────
    BAKED_OBJ    = f"{BAKE_DIR}/lbr_med7_baked.obj"
    BAKED_OFFSET = f"{BAKE_DIR}/lbr_med7_baked_offset.npy"

    def _on_bake_exit(event, context):
        # Fires when bake_node exits. Only proceed to ESS/FP/recorder if the
        # bake actually succeeded — exit 0 AND both artifacts on disk.
        # Otherwise everything downstream (FP mesh load, recorder offset load)
        # crashes with confusing GXF / FileNotFound errors instead of one
        # clear line. The usual cause of failure: no joint_states reached
        # bake within --bake_timeout, because the producer wasn't publishing
        # yet (press A) when the launch started.
        import os
        rc = getattr(event, "returncode", None)
        have_obj = os.path.exists(BAKED_OBJ)
        have_off = os.path.exists(BAKED_OFFSET)
        if rc == 0 and have_obj and have_off:
            return [
                LogInfo(msg="[stage1] bake OK — starting ESS + FP"),
                ess_launch,
                fp_launch,
                TimerAction(period=4.0, actions=[
                    LogInfo(msg="[stage2] starting rectifier + depth_saver"),
                    rectifier,
                    depth_saver,
                ]),
                TimerAction(period=6.0, actions=[
                    LogInfo(msg="[stage2b] starting pose recorder (init_only)"),
                    pose_recorder,
                ]),
            ]
        jt = context.perform_substitution(LaunchConfiguration("joint_topic"))
        return [LogInfo(msg=(
            f"[stage1] BAKE FAILED (exit={rc}, obj={have_obj}, "
            f"offset={have_off}) — NOT starting ESS/FP/recorder. "
            f"Most common cause: no joint_states on {jt} within --bake_timeout. "
            f"Start the producer, press A, confirm `ros2 topic hz {jt}` shows "
            f"~30 Hz, then relaunch."))]

    after_bake = RegisterEventHandler(
        OnProcessExit(
            target_action=bake_node,
            on_exit=_on_bake_exit,
        )
    )

    from launch.actions import OpaqueFunction
    return LaunchDescription([
        urdf_path_arg, joint_topic_arg, joint_prefix_arg,
        left_image_arg, right_image_arg,
        left_caminfo_arg, right_caminfo_arg, extrinsics_arg,
        enable_ess_arg, enable_fp_arg, enable_recorder_arg,
        ess_engine_arg, ess_threshold_arg, bake_timeout_arg,
        enable_bag_arg, bag_dir_arg, bag_topics_arg, bag_storage_arg,
        OpaqueFunction(function=_clean_runtime_dirs),
        OpaqueFunction(function=_start_recorder),
        LogInfo(msg="[stage0] baking mesh — press A once joints flow"),
        bake_node,
        after_bake,
    ])