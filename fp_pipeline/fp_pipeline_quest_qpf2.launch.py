#!/usr/bin/env python3
"""
fp_pipeline_quest_qpf2.launch.py

Quest pipeline with online URDF bake at startup.

Stages
------
Stage 0 (t=0):     bag_first_joint_publisher (sim) + bake_node
                   bake_node receives one /lbr/joint_states, runs FK,
                   writes /tmp/fp_bake_runtime/lbr_med7_baked.{obj,offset.npy},
                   exits cleanly.
Stage 1 (on bake exit, success): ESS + FP composable containers come up
                                 using the freshly-baked mesh+offset.
Stage 2 (+4s):     qpf2_receiver + stereo_depth_saver
Stage 3 (gated):   bag_to_qpf2 (video, gated) + ros2 bag play (caminfo + joints, ungated)
                   both block until /tmp/qpf2_start exists.

In live (Quest) mode set enable_sim_source:=false. The KUKA FRI reader
process must be running externally to produce /lbr/joint_states for
bake_node.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler,
                            TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


HOME = Path(os.path.expanduser("~"))
FP_PIPELINE_DIR = HOME / "fp_pipeline"
ISAAC_ROS_WS = Path(os.environ.get(
    "ISAAC_ROS_WS",
    str(HOME / "workspaces" / "isaac_ros-dev")))

GATE_PATH = "/tmp/qpf2_start"
PID_DIR = "/tmp/fp_init_pids"
BAKE_DIR = "/tmp/fp_bake_runtime"


def _clean_runtime_dirs(_):
    import shutil
    for d in (PID_DIR, BAKE_DIR):
        p = Path(d)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
    return []


def generate_launch_description():
    compressed_bag_arg = DeclareLaunchArgument("compressed_bag")
    mask_path_arg      = DeclareLaunchArgument(
        "mask_path", default_value="",
        description="Optional segmentation mask PNG. Empty in handoff "
                    "mode (IPCAI publishes /left/segmentation).")
    urdf_arg           = DeclareLaunchArgument(
        "urdf_path",
        default_value=str(HOME / "roboreg/test/assets/lbr_med7_r800"
                          / "description/lbr_med7_r800.urdf"))
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

    left_cinfo_topic_arg  = DeclareLaunchArgument(
        "left_camera_info_topic",
        default_value="/zed/zed_node/left/camera_info",
        description="ROS topic carrying CameraInfo for the left camera "
                    "(read from the bag in handoff mode).")
    right_cinfo_topic_arg = DeclareLaunchArgument(
        "right_camera_info_topic",
        default_value="/zed/zed_node/right/camera_info",
        description="ROS topic carrying CameraInfo for the right camera "
                    "(read from the bag in handoff mode).")

    ess_engine_arg = DeclareLaunchArgument(
        "ess_engine_file_path",
        default_value=str(
            ISAAC_ROS_WS / "isaac_ros_assets/models/dnn_stereo_disparity"
            / "dnn_stereo_disparity_v4.1.0_onnx_trt10.13/ess.engine"))
    ess_threshold_arg = DeclareLaunchArgument(
        "ess_threshold", default_value="0.0")
    joint_topic_arg = DeclareLaunchArgument(
        "joint_topic", default_value="/lbr/joint_states")

    # ── Stage 0: pre-bake ───────────────────────────────────────────
    bake_joint_publisher = ExecuteProcess(
        cmd=[
            "python", str(FP_PIPELINE_DIR / "bag_first_joint_publisher.py"),
            "--bag", LaunchConfiguration("compressed_bag"),
            "--joint_topic", LaunchConfiguration("joint_topic"),
            "--hold_seconds", "60.0",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_sim_source")),
    )
    bake_node = ExecuteProcess(
        cmd=[
            "python", str(FP_PIPELINE_DIR / "bake_node.py"),
            "--urdf_path",         LaunchConfiguration("urdf_path"),
            "--out_dir",           BAKE_DIR,
            "--joint_state_topic", LaunchConfiguration("joint_topic"),
            "--timeout",           "30.0",
        ],
        output="screen",
    )

    # ── Stage 1 (after bake): ESS + FP ──────────────────────────────
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
    )
    fp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(FP_PIPELINE_DIR / "isaac_ros_foundationpose_med7.launch.py")
        ),
        launch_arguments={
            "mesh_file_path": f"{BAKE_DIR}/lbr_med7_baked.obj",
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_fp")),
    )

    # ── Stage 2 ─────────────────────────────────────────────────────
    qpf2_receiver = ExecuteProcess(
        cmd=[
            "python", str(FP_PIPELINE_DIR / "qpf2_receiver.py"),
            "--port", LaunchConfiguration("qpf2_port"),
            "--left_camera_info_topic",  LaunchConfiguration("left_camera_info_topic"),
            "--right_camera_info_topic", LaunchConfiguration("right_camera_info_topic"),
            "--mask_path",  LaunchConfiguration("mask_path"),
            "--start_signal_file", GATE_PATH,
        ],
        output="screen",
    )
    depth_saver = ExecuteProcess(
        cmd=["python", str(FP_PIPELINE_DIR / "stereo_depth_saver.py"),
             "--backend", "ess",
             "--pid_file", f"{PID_DIR}/depth_saver.pid"],
        output="screen",
    )

    # ── Stage 3 (gated) ─────────────────────────────────────────────
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

    # Replay the bag's CameraInfo topics onto ROS so qpf2_receiver
    # (in topic mode) can subscribe. Filtered to just caminfo so we
    # don't double-up on the H.264 streams (those go through bag_sender).
    # Single ros2 bag play for caminfo + joint_states. Both topics are
    # rate-paced and stamped from the bag, which is what the downstream
    # consumers (qpf2_receiver, FP, IPCAI) expect.
    # Not gated on /tmp/qpf2_start: starts as soon as stage 3 fires.
    # Joints/caminfo stream during the warmup window before the user
    # touches the gate; qpf2_receiver caches the first caminfo it sees
    # and ignores the rest, so this is harmless.
    bag_play_aux = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            LaunchConfiguration("compressed_bag"),
            "--topics",
            LaunchConfiguration("left_camera_info_topic"),
            LaunchConfiguration("right_camera_info_topic"),
            LaunchConfiguration("joint_topic"),
            "--rate", LaunchConfiguration("play_rate"),
            "--loop",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_sim_source")),
    )

    # When bake_node exits, kick off everything downstream.
    after_bake = RegisterEventHandler(
        OnProcessExit(
            target_action=bake_node,
            on_exit=[
                LogInfo(msg="[stage1] bake done — starting ESS + FP"),
                ess_launch,
                fp_launch,
                TimerAction(period=4.0, actions=[
                    LogInfo(msg="[stage2] starting qpf2_receiver + "
                                "depth_saver"),
                    qpf2_receiver,
                    depth_saver,
                ]),
                TimerAction(period=9.0, actions=[
                    LogInfo(msg=f"[stage3] sender ready. "
                                f"`touch {GATE_PATH}` to start."),
                    bag_sender,
                    bag_play_aux,
                ]),
            ],
        )
    )

    return LaunchDescription([
        compressed_bag_arg, mask_path_arg, urdf_arg,
        play_rate_arg, play_loop_arg,
        qpf2_port_arg, enable_sim_arg, enable_fp_arg,
        left_cinfo_topic_arg, right_cinfo_topic_arg,
        ess_engine_arg, ess_threshold_arg,
        joint_topic_arg,
        OpaqueFunction(function=_clean_runtime_dirs),
        LogInfo(msg=["Quest-QPF2 pipeline (live bake). bag=",
                     LaunchConfiguration("compressed_bag")]),
        bake_joint_publisher,
        bake_node,
        after_bake,
    ])