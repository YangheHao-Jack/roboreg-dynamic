#!/usr/bin/env python3
"""
fp_bag_compress_custom.launch.py

Sibling to fp_bag_compress.launch.py. Same structure (bgra8 → rgb8 →
NVENC encode → bag record), but config defaults to 'custom' and
exposes the four parameters that NVIDIA's Isaac ROS H.264 encoder
honors only in custom mode: qp, iframe_interval, profile, hw_preset.

Use this when you want to override the encoder's preset behaviour —
for example, to mirror Quest 3's MediaCodec settings (1-second GOP,
Baseline profile, real-time HW preset, qp tuned to ≈16 Mbps target).

Per NVIDIA's docs, qp/iframe_interval/profile/hw_preset are ignored
when config:=iframe_cqp or pframe_cqp; the preset's baked-in defaults
apply instead. This launch defaults config to 'custom' so the
overrides actually take effect. If you want preset behaviour, just
use the original fp_bag_compress.launch.py.

Usage examples
--------------

# Quest-mirror settings (1-second GOP at 30 fps, Baseline profile,
# Ultrafast preset, qp tuned to ≈16 Mbps target — closest CQP analog
# to Quest's 16 Mbps VBR). All four overrides default to these values
# in this launch, so the simplest invocation is:
    ros2 launch ~/fp_pipeline/fp_bag_compress_custom.launch.py \\
        input_bag:=/media/jack/新加卷/26_04_13_displacement_0_1 \\
        output_bag:=/media/jack/新加卷/26_04_13_displacement_0_1_h264_quest \\
        play_rate:=1.0

# Override individual parameters (e.g. higher quality):
    ros2 launch ~/fp_pipeline/fp_bag_compress_custom.launch.py \\
        input_bag:=... output_bag:=... \\
        qp:=18 hw_preset:=2

# Near-lossless / quasi-lossless reference encoding (very large bag):
    ros2 launch ~/fp_pipeline/fp_bag_compress_custom.launch.py \\
        input_bag:=... output_bag:=... \\
        qp:=0 iframe_interval:=1 profile:=2 hw_preset:=3 play_rate:=0.5

Parameter reference (from official NVIDIA Isaac ROS docs):
    qp              uint32  default 20.   0–51, lower = higher quality.
                    qp=0 is the lowest quantizer (effectively lossless on
                    luma; chroma is 4:2:0-subsampled before encode so
                    chroma alone is not bit-exact).
    iframe_interval int32   default 5.    Frames between I-frames.
                    1 = all-I-frame, N = 1 I-frame per N frames.
    profile         uint32  default 0.    0=Main, 1=Baseline, 2=High.
    hw_preset       uint32  default 0.    0=Ultrafast, 1=Fast, 2=Medium,
                    3=Slow. Higher = better quality, slower encode.
    config          string  default pframe_cqp.  iframe_cqp / pframe_cqp /
                    custom. The custom-parameter overrides above apply
                    ONLY when config:=custom.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


HOME = Path(os.path.expanduser("~"))


def generate_launch_description():
    # ── Bag I/O arguments ──────────────────────────────────────────────
    input_bag_arg = DeclareLaunchArgument(
        "input_bag",
        description="Path of the input rosbag (uncompressed bgra8 ZED).",
    )
    output_bag_arg = DeclareLaunchArgument(
        "output_bag",
        description="Output path for the compressed bag (will be created).",
    )
    width_arg = DeclareLaunchArgument(
        "input_width", default_value="1920")
    height_arg = DeclareLaunchArgument(
        "input_height", default_value="1080")
    rate_arg = DeclareLaunchArgument(
        "play_rate", default_value="1.0",
        description="Bag playback rate. 1.0 = real-time. Drop to 0.5 if "
                    "NVENC under-throughputs at heavier settings (e.g. "
                    "qp=0 with hw_preset=3 may not keep up at 1080p "
                    "stereo 30 Hz). Default 1.0 should work for typical "
                    "Quest-mirror settings on RTX 5090.",
    )

    # ── Encoder preset / overrides ─────────────────────────────────────
    config_arg = DeclareLaunchArgument(
        "encoder_config", default_value="custom",
        description="Encoder preset. Defaults to 'custom' in this sibling "
                    "launch, so qp / iframe_interval / profile / hw_preset "
                    "are honored. Set 'iframe_cqp' or 'pframe_cqp' to fall "
                    "back to preset behaviour (and the four overrides "
                    "below will be ignored).",
    )
    qp_arg = DeclareLaunchArgument(
        "qp", default_value="22",
        description="Constant QP value, 0–51. Lower = higher quality, "
                    "larger files. qp=22 is a Quest-mirror default — "
                    "yields ≈16 Mbps for 1080p natural content, matching "
                    "Quest's 16 Mbps VBR target. qp=20 is the encoder's "
                    "own documented default (slightly higher quality). "
                    "qp=0 is the lowest quantizer (quasi-lossless luma).",
    )
    iframe_interval_arg = DeclareLaunchArgument(
        "iframe_interval", default_value="30",
        description="Frames between I-frames. Default 30 = 1-second GOP "
                    "at 30 fps, mirroring Quest's 1-second GOP at 60 fps. "
                    "1 = all-I-frame, 5 = encoder's own documented default.",
    )
    profile_arg = DeclareLaunchArgument(
        "profile", default_value="1",
        description="H.264 profile. 0=Main (encoder's own default), "
                    "1=Baseline (Quest-mirror — Snapdragon MediaCodec "
                    "defaults to Baseline when KEY_PROFILE is unset, "
                    "as per the Quest APK source), 2=High (better "
                    "compression, less compatible with old decoders).",
    )
    hw_preset_arg = DeclareLaunchArgument(
        "hw_preset", default_value="0",
        description="NVENC hardware preset. 0=Ultrafast (Quest-mirror — "
                    "real-time HW encoder behaviour, also the encoder's "
                    "own default), 1=Fast, 2=Medium, 3=Slow (best quality, "
                    "slowest encode).",
    )

    # ── Convenience handles ────────────────────────────────────────────
    width  = LaunchConfiguration("input_width")
    height = LaunchConfiguration("input_height")
    enc_config       = LaunchConfiguration("encoder_config")
    enc_qp           = LaunchConfiguration("qp")
    enc_iframe_intvl = LaunchConfiguration("iframe_interval")
    enc_profile      = LaunchConfiguration("profile")
    enc_hw_preset    = LaunchConfiguration("hw_preset")

    # ── bgra8 → rgb8 converters (one per side) ─────────────────────────
    left_format = ComposableNode(
        name="bag_left_format_node",
        package="isaac_ros_image_proc",
        plugin="nvidia::isaac_ros::image_proc::ImageFormatConverterNode",
        parameters=[{
            "image_width":      width,
            "image_height":     height,
            "encoding_desired": "rgb8",
        }],
        remappings=[
            ("image_raw", "/zed/zed_node/left/image_rect_color"),
            ("image",     "/zed/zed_node/left/image_rgb"),
        ],
    )
    right_format = ComposableNode(
        name="bag_right_format_node",
        package="isaac_ros_image_proc",
        plugin="nvidia::isaac_ros::image_proc::ImageFormatConverterNode",
        parameters=[{
            "image_width":      width,
            "image_height":     height,
            "encoding_desired": "rgb8",
        }],
        remappings=[
            ("image_raw", "/zed/zed_node/right/image_rect_color"),
            ("image",     "/zed/zed_node/right/image_rgb"),
        ],
    )

    # ── H.264 encoders ─────────────────────────────────────────────────
    # The four custom parameters (qp, iframe_interval, profile, hw_preset)
    # are passed unconditionally — they're ignored by the encoder when
    # config != 'custom', so this is safe regardless of which preset is
    # selected. (Per NVIDIA's docs.)
    left_encoder = ComposableNode(
        name="left_encoder_node",
        package="isaac_ros_h264_encoder",
        plugin="nvidia::isaac_ros::h264_encoder::EncoderNode",
        parameters=[{
            "gpu_id":          0,
            "input_width":     width,
            "input_height":    height,
            "config":          enc_config,
            "qp":              enc_qp,
            "iframe_interval": enc_iframe_intvl,
            "profile":         enc_profile,
            "hw_preset":       enc_hw_preset,
        }],
        remappings=[
            ("image_raw",        "/zed/zed_node/left/image_rgb"),
            ("image_compressed", "/zed/zed_node/left/image_compressed"),
        ],
    )
    right_encoder = ComposableNode(
        name="right_encoder_node",
        package="isaac_ros_h264_encoder",
        plugin="nvidia::isaac_ros::h264_encoder::EncoderNode",
        parameters=[{
            "gpu_id":          0,
            "input_width":     width,
            "input_height":    height,
            "config":          enc_config,
            "qp":              enc_qp,
            "iframe_interval": enc_iframe_intvl,
            "profile":         enc_profile,
            "hw_preset":       enc_hw_preset,
        }],
        remappings=[
            ("image_raw",        "/zed/zed_node/right/image_rgb"),
            ("image_compressed", "/zed/zed_node/right/image_compressed"),
        ],
    )

    container = ComposableNodeContainer(
        name="encoder_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            left_format, right_format,
            left_encoder, right_encoder,
        ],
        output="screen",
    )

    # ── Bag record (compressed topics + metadata, not raw images) ─────
    rosbag_record = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "/zed/zed_node/left/image_compressed",
            "/zed/zed_node/right/image_compressed",
            "/zed/zed_node/left/camera_info",
            "/zed/zed_node/right/camera_info",
            "/lbr/joint_states",
            "/lbr/robot_description",
            "/tf", "/tf_static",
            "-o", LaunchConfiguration("output_bag"),
        ],
        output="screen",
    )

    # ── Bag play ──────────────────────────────────────────────────────
    rosbag_play = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            LaunchConfiguration("input_bag"),
            "--rate", LaunchConfiguration("play_rate"),
        ],
        output="screen",
    )

    return LaunchDescription([
        # Bag I/O
        input_bag_arg, output_bag_arg, width_arg, height_arg, rate_arg,
        # Encoder preset / overrides
        config_arg, qp_arg, iframe_interval_arg, profile_arg, hw_preset_arg,
        # Banner
        LogInfo(msg=["Bag compression: ",
                     LaunchConfiguration("input_bag"),
                     "  →  ",
                     LaunchConfiguration("output_bag"),
                     "   (rate=", LaunchConfiguration("play_rate"),
                     ", config=", enc_config,
                     ", qp=", enc_qp,
                     ", iframe_interval=", enc_iframe_intvl,
                     ", profile=", enc_profile,
                     ", hw_preset=", enc_hw_preset,
                     ")  [custom params honored only when config=custom]"]),
        container,
        rosbag_record,
        rosbag_play,
    ])
