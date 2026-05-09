#!/usr/bin/env python3
"""
isaac_ros_foundationpose_med7.launch.py

FoundationPose tracking with a user-provided segmentation mask
(no RT-DETR detection branch). Used by all three pipelines (offline,
live, qpf2).

Inputs (subscribed):
    /left/image_rect, /left/camera_info_rect,
    /left/depth_image, /left/segmentation

Outputs (published by FP):
    /pose_estimation/pose_matrix_output  (TensorList, init pose)
    /tracking/pose_matrix_output         (TensorList, per-frame tracked)
    /pose_estimation/output, /tracking/output  (Detection3DArray)

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
    reset_arg = DeclareLaunchArgument(
        "reset_period", default_value="2147483647",
        description="Selector reset period in ms. int32 max = init once.")
    sync_arg = DeclareLaunchArgument(
        "sync_threshold", default_value="100000000",
        description="Sync threshold (ns) for FP input pairing. "
                    "100ms is robust; below this is the inter-frame interval.")

    selector = ComposableNode(
        name="selector_node",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::Selector",
        parameters=[{"reset_period": LaunchConfiguration("reset_period")}],
        remappings=[
            ("image",        "/left/image_rect"),
            ("camera_info",  "/left/camera_info_rect"),
            ("depth_image",  "/left/depth_image"),
            ("segmentation", "/left/segmentation"),
        ],
    )

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
    )

    tracker = ComposableNode(
        name="foundationpose_tracking_node",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode",
        parameters=[{
            "mesh_file_path":            LaunchConfiguration("mesh_file_path"),
            "refine_engine_file_path":   LaunchConfiguration("refine_engine_file_path"),
            "refine_input_tensor_names":  ["input_tensor1", "input_tensor2"],
            "refine_input_binding_names": ["input1", "input2"],
            "refine_output_tensor_names":  ["output_tensor1", "output_tensor2"],
            "refine_output_binding_names": ["output1", "output2"],
            "sync_threshold":            LaunchConfiguration("sync_threshold"),
        }],
    )

    container = ComposableNodeContainer(
        name="foundationpose_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[selector, estimator, tracker],
        output="screen",
    )

    return LaunchDescription([
        mesh_arg, refine_arg, score_arg, reset_arg, sync_arg,
        container,
    ])