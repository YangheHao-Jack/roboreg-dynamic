#!/usr/bin/env python3
"""
fp_bag_compress.launch.py

One-shot bag compression. Reads a raw bag of bgra8 ZED stereo images,
converts to rgb8, NVENC-encodes via Isaac ROS H.264, records to a new
bag.

    raw bag (bgra8 /image_rect_color)
      → ImageFormatConverterNode  → rgb8 (/image_rgb)
      → H264EncoderNode           → /image_compressed
      → ros2 bag record           → output bag

Output bag contains:
    /zed/zed_node/{l,r}/image_compressed   (CompressedImage, H.264)
    /zed/zed_node/{l,r}/camera_info        (passthrough)
    /lbr/joint_states, /lbr/robot_description, /tf, /tf_static

Usage:
    ros2 launch ~/fp_pipeline/fp_bag_compress.launch.py \\
        input_bag:=/path/to/raw_bag \\
        output_bag:=/path/to/h264_bag

Stop with Ctrl-C once the input bag finishes playing.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def _format_node(name: str, in_topic: str, out_topic: str, w, h):
    return ComposableNode(
        name=name,
        package="isaac_ros_image_proc",
        plugin="nvidia::isaac_ros::image_proc::ImageFormatConverterNode",
        parameters=[{
            "image_width":      w,
            "image_height":     h,
            "encoding_desired": "rgb8",
        }],
        remappings=[("image_raw", in_topic), ("image", out_topic)],
    )


def _encoder_node(name: str, in_topic: str, out_topic: str, w, h, config):
    return ComposableNode(
        name=name,
        package="isaac_ros_h264_encoder",
        plugin="nvidia::isaac_ros::h264_encoder::EncoderNode",
        parameters=[{
            "gpu_id":       0,
            "input_width":  w,
            "input_height": h,
            "config":       config,
        }],
        remappings=[("image_raw", in_topic), ("image_compressed", out_topic)],
    )


def generate_launch_description():
    input_bag_arg  = DeclareLaunchArgument("input_bag")
    output_bag_arg = DeclareLaunchArgument("output_bag")
    width_arg      = DeclareLaunchArgument("input_width",  default_value="1920")
    height_arg     = DeclareLaunchArgument("input_height", default_value="1080")
    config_arg     = DeclareLaunchArgument(
        "encoder_config", default_value="pframe_cqp",
        description="Encoder preset: pframe_cqp (default, fast), "
                    "iframe_cqp (all I-frames, larger but no temporal artifacts).")
    rate_arg       = DeclareLaunchArgument("play_rate", default_value="1.0")

    w = LaunchConfiguration("input_width")
    h = LaunchConfiguration("input_height")
    cfg = LaunchConfiguration("encoder_config")

    container = ComposableNodeContainer(
        name="encoder_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            _format_node("bag_left_format_node",
                         "/zed/zed_node/left/image_rect_color",
                         "/zed/zed_node/left/image_rgb",  w, h),
            _format_node("bag_right_format_node",
                         "/zed/zed_node/right/image_rect_color",
                         "/zed/zed_node/right/image_rgb", w, h),
            _encoder_node("left_encoder_node",
                          "/zed/zed_node/left/image_rgb",
                          "/zed/zed_node/left/image_compressed",  w, h, cfg),
            _encoder_node("right_encoder_node",
                          "/zed/zed_node/right/image_rgb",
                          "/zed/zed_node/right/image_compressed", w, h, cfg),
        ],
        output="screen",
    )

    rosbag_record = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "/zed/zed_node/left/image_compressed",
            "/zed/zed_node/right/image_compressed",
            "/zed/zed_node/left/camera_info",
            "/zed/zed_node/right/camera_info",
            "/lbr/joint_states", "/lbr/robot_description",
            "/tf", "/tf_static",
            "-o", LaunchConfiguration("output_bag"),
        ],
        output="screen",
    )
    rosbag_play = ExecuteProcess(
        cmd=["ros2", "bag", "play", LaunchConfiguration("input_bag"),
             "--rate", LaunchConfiguration("play_rate")],
        output="screen",
    )

    return LaunchDescription([
        input_bag_arg, output_bag_arg, width_arg, height_arg,
        config_arg, rate_arg,
        LogInfo(msg=["Bag compress: ",
                     LaunchConfiguration("input_bag"), " → ",
                     LaunchConfiguration("output_bag"),
                     "  (rate=", LaunchConfiguration("play_rate"),
                     ", config=", cfg, ")"]),
        container, rosbag_record, rosbag_play,
    ])