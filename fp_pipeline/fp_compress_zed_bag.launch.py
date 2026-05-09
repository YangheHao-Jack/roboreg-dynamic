#!/usr/bin/env python3
"""
fp_compress_zed_bag.launch.py

Compress a raw ZED bag's left + right image streams using
isaac_ros_h264_encoder (NVENC), and record the result + the
camera_info topics into a new bag.

This produces the input for fp_pipeline_quest_sim.launch.py — the
compressed bag simulates Quest 3's H.264 output (same h264_msgs/H264Packet
message type that isaac_ros_h264_decoder consumes on the receive side).

Pipeline:
   ros2 bag play <input_bag>
       publishes /zed/zed_node/{left,right}/image_rect_color (raw 1080p)
                 /zed/zed_node/{left,right}/camera_info
                  │
                  ▼
   ┌──────────────────────────────────────────────────┐
   │ encoder_container (composable, NITROS)           │
   │   EncoderNode (left)                             │
   │     in:  /zed/zed_node/left/image_rect_color     │
   │     out: /zed/zed_node/left/image_compressed     │
   │   EncoderNode (right)                            │
   │     in:  /zed/zed_node/right/image_rect_color    │
   │     out: /zed/zed_node/right/image_compressed    │
   └──────────────────────────────────────────────────┘
                  │
                  ▼
   ros2 bag record /zed/zed_node/{left,right}/image_compressed
                   /zed/zed_node/{left,right}/camera_info

Concurrency caveat: NVIDIA documents (nvbugs/5554121) that running
encoder and decoder concurrently can cause the decoder to intermittently
produce no output. So compression and playback are done in *separate*
launches: this one writes the compressed bag, then exits; playback runs
afterward against the saved bag.

Usage:
    ros2 launch ~/fp_pipeline/fp_compress_zed_bag.launch.py \\
        input_bag:="/path/to/raw_zed_bag" \\
        output_bag:="/path/to/output_h264_bag" \\
        play_rate:=1.0

Notes:
- input_bag must contain /zed/zed_node/{left,right}/image_rect_color
  as raw sensor_msgs/Image (1080p ZED native).
- play_rate=1.0 plays at recorded timestamps (real-time). Use a smaller
  value (0.5, 0.25) if NVENC under-throughputs at 1080p stereo 30 Hz on
  your hardware (it shouldn't on a 5090, but useful as a safety knob).
- The launch terminates when bag playback finishes. The 'record' process
  is sent SIGINT shortly after, which finalises the output bag cleanly.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            LogInfo, RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


HOME = Path(os.path.expanduser("~"))


def generate_launch_description():
    # ── Arguments ──────────────────────────────────────────────────────
    input_bag_arg = DeclareLaunchArgument(
        "input_bag", description="Path to the raw ZED bag to compress.")
    output_bag_arg = DeclareLaunchArgument(
        "output_bag",
        description="Path to write the compressed bag to (must not exist).")
    play_rate_arg = DeclareLaunchArgument(
        "play_rate", default_value="1.0",
        description="ros2 bag play --rate value. 1.0 = real-time. "
                    "Lower if NVENC under-throughputs.")
    input_width_arg = DeclareLaunchArgument(
        "input_width", default_value="1920",
        description="Source image width (ZED 2i native = 1920 at HD1080).")
    input_height_arg = DeclareLaunchArgument(
        "input_height", default_value="1080",
        description="Source image height.")

    # Topic names (left as args so this can be retargeted if your bag uses
    # different topic names; defaults match the ZED ROS2 wrapper).
    left_image_topic_arg = DeclareLaunchArgument(
        "left_image_topic",
        default_value="/zed/zed_node/left/image_rect_color")
    right_image_topic_arg = DeclareLaunchArgument(
        "right_image_topic",
        default_value="/zed/zed_node/right/image_rect_color")
    left_cinfo_topic_arg = DeclareLaunchArgument(
        "left_cinfo_topic", default_value="/zed/zed_node/left/camera_info")
    right_cinfo_topic_arg = DeclareLaunchArgument(
        "right_cinfo_topic",
        default_value="/zed/zed_node/right/camera_info")
    left_compressed_topic_arg = DeclareLaunchArgument(
        "left_compressed_topic",
        default_value="/zed/zed_node/left/image_compressed")
    right_compressed_topic_arg = DeclareLaunchArgument(
        "right_compressed_topic",
        default_value="/zed/zed_node/right/image_compressed")

    # ── Encoder nodes ──────────────────────────────────────────────────
    # Plugin/parameter form per NVIDIA Isaac ROS Compression docs:
    #   plugin: nvidia::isaac_ros::h264_encoder::EncoderNode
    #   params: input_width, input_height
    #   topics (default): image_raw -> image_compressed
    encoder_left = ComposableNode(
        package="isaac_ros_h264_encoder",
        plugin="nvidia::isaac_ros::h264_encoder::EncoderNode",
        name="encoder_left",
        parameters=[{
            "input_width":  LaunchConfiguration("input_width"),
            "input_height": LaunchConfiguration("input_height"),
        }],
        remappings=[
            ("image_raw",        LaunchConfiguration("left_image_topic")),
            ("image_compressed",
             LaunchConfiguration("left_compressed_topic")),
        ],
    )
    encoder_right = ComposableNode(
        package="isaac_ros_h264_encoder",
        plugin="nvidia::isaac_ros::h264_encoder::EncoderNode",
        name="encoder_right",
        parameters=[{
            "input_width":  LaunchConfiguration("input_width"),
            "input_height": LaunchConfiguration("input_height"),
        }],
        remappings=[
            ("image_raw",        LaunchConfiguration("right_image_topic")),
            ("image_compressed",
             LaunchConfiguration("right_compressed_topic")),
        ],
    )

    encoder_container = ComposableNodeContainer(
        name="encoder_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[encoder_left, encoder_right],
        output="screen",
    )

    # ── Bag record (compressed image topics + camera_info) ────────────
    # Started before the player so all messages are captured from t=0.
    bag_record = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "-o", LaunchConfiguration("output_bag"),
            LaunchConfiguration("left_compressed_topic"),
            LaunchConfiguration("right_compressed_topic"),
            LaunchConfiguration("left_cinfo_topic"),
            LaunchConfiguration("right_cinfo_topic"),
        ],
        output="screen",
    )

    # ── Bag play (delayed slightly so encoders + recorder are up) ─────
    bag_play = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            LaunchConfiguration("input_bag"),
            "--rate", LaunchConfiguration("play_rate"),
        ],
        output="screen",
    )
    bag_play_delayed = TimerAction(period=3.0, actions=[bag_play])

    # When playback finishes, give recorder another 2 s to flush, then
    # shut down. Without this the launch hangs on the recorder which
    # would otherwise wait for new messages forever.
    on_play_done = RegisterEventHandler(
        OnProcessExit(
            target_action=bag_play,
            on_exit=[
                LogInfo(msg="Bag playback finished. Stopping recorder in 2 s."),
                TimerAction(
                    period=2.0,
                    actions=[
                        # Send SIGINT to bag record so it finalises cleanly.
                        ExecuteProcess(
                            cmd=["pkill", "-INT", "-f",
                                 "ros2 bag record"],
                            output="screen",
                        ),
                    ],
                ),
            ],
        ),
    )

    return LaunchDescription([
        input_bag_arg, output_bag_arg, play_rate_arg,
        input_width_arg, input_height_arg,
        left_image_topic_arg, right_image_topic_arg,
        left_cinfo_topic_arg, right_cinfo_topic_arg,
        left_compressed_topic_arg, right_compressed_topic_arg,
        LogInfo(msg=["Compressing ", LaunchConfiguration("input_bag"),
                     " -> ", LaunchConfiguration("output_bag")]),
        encoder_container,
        bag_record,
        bag_play_delayed,
        on_play_done,
    ])
