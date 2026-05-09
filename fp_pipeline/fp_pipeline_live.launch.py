#!/usr/bin/env python3
"""
fp_pipeline_live.launch.py

Live FP pipeline (sim or real source). Same downstream as the offline
launch — ESS, depth_saver, FoundationPose — but the source is a live
publisher process rather than fp_offline_publisher.

Sources
-------
    source:=sim   — runs fp_live_source_sim.py (read frames from disk,
                    publish at fixed rate, simulating live)
    source:=real  — does NOT start a sim source. Bring up your camera
                    (e.g. zed_wrapper) externally and ensure the topics
                    /left/image_rect, /right/image_rect, /left/camera_info,
                    /left/segmentation are published at 960×576.

Usage (sim)
-----------
    ros2 launch ~/fp_pipeline/fp_pipeline_live.launch.py \\
        sim_image_dir:=/.../images/left \\
        sim_right_image_dir:=/.../images/right \\
        sim_camera_yaml:=/.../camera.left.image.camera_info_4.yaml \\
        sim_right_camera_yaml:=/.../camera.right.image.camera_info_4.yaml \\
        sim_mask_path:=/.../mask_left_0.png \\
        sim_rate:=30.0 \\
        sim_preload_ram:=true
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
    source_arg = DeclareLaunchArgument(
        "source", default_value="sim", choices=["sim", "real"])

    # ── Sim source args ─────────────────────────────────────────────
    sim_image_dir_arg       = DeclareLaunchArgument("sim_image_dir",       default_value="")
    sim_right_image_dir_arg = DeclareLaunchArgument("sim_right_image_dir", default_value="")
    sim_camera_yaml_arg     = DeclareLaunchArgument("sim_camera_yaml",     default_value="")
    sim_right_camera_yaml_arg = DeclareLaunchArgument("sim_right_camera_yaml", default_value="")
    sim_mask_path_arg       = DeclareLaunchArgument("sim_mask_path",       default_value="")
    sim_rate_arg            = DeclareLaunchArgument("sim_rate",            default_value="30.0")
    sim_on_end_arg = DeclareLaunchArgument(
        "sim_on_end", default_value="stop",
        choices=["loop", "hold_last", "stop"])
    sim_preload_ram_arg = DeclareLaunchArgument(
        "sim_preload_ram", default_value="true",
        choices=["true", "false"],
        description="Preload all frames to RAM (required for 30 Hz).")
    sim_publish_full_arg = DeclareLaunchArgument(
        "sim_publish_full", default_value="1",
        description="Publish full-res topics for recorder overlay.")

    # ── ESS engine ──────────────────────────────────────────────────
    ess_engine_arg = DeclareLaunchArgument(
        "ess_engine_file_path",
        default_value=str(
            ISAAC_ROS_WS / "isaac_ros_assets/models/dnn_stereo_disparity"
            / "dnn_stereo_disparity_v4.1.0_onnx_trt10.13/ess.engine"))

    # ── Depth saver ─────────────────────────────────────────────────
    depth_save_dir_arg = DeclareLaunchArgument(
        "depth_save_dir", default_value="",
        description="If set, depth_saver writes .npy files. "
                    "Empty = republish-only (default).")
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic", default_value="/left/depth_image")

    # ── Sim source process ──────────────────────────────────────────
    sim_source = ExecuteProcess(
        cmd=[
            "python", str(FP_PIPELINE_DIR / "fp_live_source_sim.py"),
            "--image_dir",         LaunchConfiguration("sim_image_dir"),
            "--right_image_dir",   LaunchConfiguration("sim_right_image_dir"),
            "--camera_yaml",       LaunchConfiguration("sim_camera_yaml"),
            "--right_camera_yaml", LaunchConfiguration("sim_right_camera_yaml"),
            "--mask_path",         LaunchConfiguration("sim_mask_path"),
            "--rate",              LaunchConfiguration("sim_rate"),
            "--on_end",            LaunchConfiguration("sim_on_end"),
            "--preload_ram",       LaunchConfiguration("sim_preload_ram"),
            "--publish_full",      LaunchConfiguration("sim_publish_full"),
        ],
        output="screen",
        condition=LaunchConfigurationEquals("source", "sim"),
    )
    real_note = LogInfo(
        msg=["source:=real selected. Bring up your camera externally with "
             "/left/image_rect, /right/image_rect, /left/camera_info, "
             "/left/segmentation at 960×576."],
        condition=LaunchConfigurationEquals("source", "real"),
    )

    # ── ESS ─────────────────────────────────────────────────────────
    ess_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("isaac_ros_ess"),
                "launch", "isaac_ros_ess.launch.py"])
        ]),
        launch_arguments={
            "engine_file_path": LaunchConfiguration("ess_engine_file_path"),
        }.items(),
    )

    # ── Depth saver (republish-only by default) ─────────────────────
    depth_saver = ExecuteProcess(
        cmd=[
            "python", str(FP_PIPELINE_DIR / "stereo_depth_saver.py"),
            "--backend",     "ess",
            "--depth_topic", LaunchConfiguration("depth_topic"),
            "--out_dir",     LaunchConfiguration("depth_save_dir"),
        ],
        output="screen",
    )

    # ── FoundationPose ──────────────────────────────────────────────
    fp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(FP_PIPELINE_DIR / "isaac_ros_foundationpose_med7.launch.py")
        ),
    )

    return LaunchDescription([
        source_arg,
        sim_image_dir_arg, sim_right_image_dir_arg,
        sim_camera_yaml_arg, sim_right_camera_yaml_arg,
        sim_mask_path_arg, sim_rate_arg, sim_on_end_arg,
        sim_preload_ram_arg, sim_publish_full_arg,
        ess_engine_arg,
        depth_save_dir_arg, depth_topic_arg,
        LogInfo(msg=["FP live pipeline. source=",
                     LaunchConfiguration("source"),
                     "  rate=", LaunchConfiguration("sim_rate")]),
        sim_source,
        real_note,
        ess_launch,
        depth_saver,
        fp_launch,
    ])