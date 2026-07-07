#!/usr/bin/env python3
"""
isaac_ros_foundationpose_med7.launch.py

FoundationPose **pose-estimation only** with a user-provided segmentation
mask (no RT-DETR detection branch, no Selector, no tracking node).

Why estimation-only:
    The Selector + tracking topology runs pose estimation on exactly ONE
    frame (reset_period = int32 max = init once) and routes everything else
    to the tracker. If that single frame's mask/depth isn't clean the init
    silently fails and never retries -> intermittent (~1/10) init success.
    We don't need tracking here: the recorder grabs the first init pose and
    exits. So we wire the four inputs straight into the FoundationPoseNode,
    which then runs pose estimation on *every* synced frame and publishes a
    pose each time. The first clean frame wins; no single-frame gamble.

Inputs (subscribed directly by the estimator, remapped from pose_estimation/*):
    /left/image_rect, /left/camera_info_rect,
    /left/depth_image, /left/segmentation

Outputs (published by FP):
    /pose_estimation/pose_matrix_output  (TensorList, 4x4 col-major fp32)
    /pose_estimation/output              (Detection3DArray, debug)

Run:
    ros2 launch ~/fp_pipeline/isaac_ros_foundationpose_med7.launch.py
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


HOME = Path(os.path.expanduser("~"))
FP_ASSETS = HOME / "FoundationPose_assets"
FP_MODELS = (HOME / "workspaces" / "isaac_ros-dev" / "isaac_ros_assets"
             / "models" / "foundationpose")


def generate_launch_description():
    mesh_arg = DeclareLaunchArgument(
        "mesh_file_path",
        default_value=str(FP_ASSETS / "lbr_med7_baked.obj"))
    refine_arg = DeclareLaunchArgument(
        "refine_engine_file_path",
        default_value=str(FP_MODELS / "refine_trt_engine.plan"))
    score_arg = DeclareLaunchArgument(
        "score_engine_file_path",
        default_value=str(FP_MODELS / "score_trt_engine.plan"))
    sync_arg = DeclareLaunchArgument(
        "sync_threshold", default_value="100000000",
        description="Sync threshold (ns) for FP input pairing. "
                    "100ms is robust; below this is the inter-frame interval.")

    # Estimation-only: the FoundationPoseNode subscribes to the four inputs
    # directly (its pose_estimation/* topics remapped to /left/*) and runs
    # pose estimation on every synced tuple. No Selector (which would gate to
    # a single init frame) and no tracker.
    estimator = ComposableNode(
        name="foundationpose_node",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseNode",
        parameters=[{
            "mesh_file_path":            LaunchConfiguration("mesh_file_path"),
            "refine_engine_file_path":   LaunchConfiguration("refine_engine_file_path"),
            "refine_input_tensor_names":  ["input_tensor1", "input_tensor2"],
            "refine_input_binding_names": ["input1", "input2"],
            "refine_output_tensor_names":  ["output_tensor1", "output_tensor2"],
            "refine_output_binding_names": ["output1", "output2"],
            "score_engine_file_path":    LaunchConfiguration("score_engine_file_path"),
            "score_input_tensor_names":  ["input_tensor1", "input_tensor2"],
            "score_input_binding_names": ["input1", "input2"],
            "score_output_tensor_names":  ["output_tensor"],
            "score_output_binding_names": ["output1"],
            "sync_threshold":            LaunchConfiguration("sync_threshold"),
        }],
        remappings=[
            ("pose_estimation/image",        "/left/image_rect"),
            ("pose_estimation/camera_info",  "/left/camera_info_rect"),
            ("pose_estimation/depth_image",  "/left/depth_image"),
            ("pose_estimation/segmentation", "/left/segmentation"),
        ],
    )

    container = ComposableNodeContainer(
        name="foundationpose_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[estimator],
        output="screen",
    )

    return LaunchDescription([
        mesh_arg, refine_arg, score_arg, sync_arg,
        container,
    ])