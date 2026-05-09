#!/usr/bin/env python3
"""
fp_pipeline.launch.py

Offline FP pipeline. Runs against image/depth files via a separate
publisher process (fp_offline_publisher.py).

Three stages:
  1. Stereo backend (FoundationStereo or ESS) -> /disparity
  2. stereo_depth_saver (Python) -> /left/depth_image + optional .npy
  3. FoundationPose -> /tracking/pose_matrix_output

The recorder (fp_pose_recorder.py) and the publisher
(fp_offline_publisher.py) are run separately, not from this launch.

Usage:
    ros2 launch ~/fp_pipeline/fp_pipeline.launch.py stereo_backend:=ess
    ros2 launch ~/fp_pipeline/fp_pipeline.launch.py stereo_backend:=fs

Optional overrides:
    fs_engine_file_path:=...
    ess_engine_file_path:=...
    depth_out_dir:=...                  (.npy depth save dir)
    depth_viz_dir:=...                  (depth/disp viz PNGs)
    enable_depth_saver:=true|false      (false in live-NITROS mode)
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo)
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


HOME = Path(os.path.expanduser("~"))
FP_PIPELINE_DIR = HOME / "fp_pipeline"
ISAAC_ROS_WS = Path(os.environ.get(
    "ISAAC_ROS_WS",
    str(HOME / "workspaces" / "isaac_ros-dev")))


def generate_launch_description():
    backend_arg = DeclareLaunchArgument(
        "stereo_backend", default_value="fs",
        choices=["fs", "ess"])

    fs_engine_arg = DeclareLaunchArgument(
        "fs_engine_file_path",
        default_value=str(
            ISAAC_ROS_WS / "isaac_ros_assets/models/foundationstereo"
            / "deployable_v2.0/foundationstereo_576x960.engine"))
    ess_engine_arg = DeclareLaunchArgument(
        "ess_engine_file_path",
        default_value=str(
            ISAAC_ROS_WS / "isaac_ros_assets/models/dnn_stereo_disparity"
            / "dnn_stereo_disparity_v4.1.0_onnx_trt10.13/ess.engine"))

    depth_out_arg = DeclareLaunchArgument(
        "depth_out_dir", default_value="",
        description="If set, depth_saver writes .npy files here.")
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic", default_value="/left/depth_image")
    enable_depth_saver_arg = DeclareLaunchArgument(
        "enable_depth_saver", default_value="true",
        choices=["true", "false"],
        description="Set false in live mode where NITROS publishes "
                    "/left/depth_image directly.")
    viz_dir_arg = DeclareLaunchArgument(
        "depth_viz_dir", default_value="",
        description="If set, depth_saver writes coloured viz PNGs here.")

    # ── Stereo backends (mutually exclusive) ────────────────────────
    fs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("isaac_ros_examples"),
                "launch", "isaac_ros_examples.launch.py"])
        ]),
        launch_arguments={
            "launch_fragments": "foundationstereo",
            "engine_file_path": LaunchConfiguration("fs_engine_file_path"),
        }.items(),
        condition=LaunchConfigurationEquals("stereo_backend", "fs"),
    )
    ess_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("isaac_ros_ess"),
                "launch", "isaac_ros_ess.launch.py"])
        ]),
        launch_arguments={
            "engine_file_path": LaunchConfiguration("ess_engine_file_path"),
        }.items(),
        condition=LaunchConfigurationEquals("stereo_backend", "ess"),
    )

    # ── Depth saver ─────────────────────────────────────────────────
    depth_saver = ExecuteProcess(
        cmd=[
            "python", str(FP_PIPELINE_DIR / "stereo_depth_saver.py"),
            "--backend",     LaunchConfiguration("stereo_backend"),
            "--out_dir",     LaunchConfiguration("depth_out_dir"),
            "--depth_topic", LaunchConfiguration("depth_topic"),
            "--viz_dir",     LaunchConfiguration("depth_viz_dir"),
        ],
        output="screen",
        condition=LaunchConfigurationEquals("enable_depth_saver", "true"),
    )

    # ── FoundationPose ──────────────────────────────────────────────
    fp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(FP_PIPELINE_DIR / "isaac_ros_foundationpose_med7.launch.py")
        ),
    )

    return LaunchDescription([
        backend_arg, fs_engine_arg, ess_engine_arg,
        depth_out_arg, depth_topic_arg, enable_depth_saver_arg, viz_dir_arg,
        LogInfo(msg=["FP offline pipeline. backend=",
                     LaunchConfiguration("stereo_backend")]),
        fs_launch, ess_launch,
        depth_saver,
        fp_launch,
    ])