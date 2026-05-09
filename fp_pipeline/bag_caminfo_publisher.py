#!/usr/bin/env python3
"""
bag_caminfo_publisher.py

Reads the FIRST CameraInfo message from each side in a rosbag and
republishes them on /left/camera_info_full and /right/camera_info_full
with TRANSIENT_LOCAL durability so late subscribers still receive
them. Holds for hold_seconds, then exits.

Used so IPCAI (which expects 1080p K) can subscribe to the bag's
original 1080p camera_info topics directly, bypassing the rescaled
camera_info from qpf2_receiver (which is at 960x576 to match its
downscaled images).

Usage
    python bag_caminfo_publisher.py \\
        --bag /path/to/bag \\
        --left_in_topic  /zed/zed_node/left/camera_info \\
        --right_in_topic /zed/zed_node/right/camera_info \\
        --left_out_topic  /left/camera_info_full \\
        --right_out_topic /right/camera_info_full \\
        --hold_seconds 60.0
"""

import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSDurabilityPolicy, QoSHistoryPolicy)
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo

try:
    from rosbag2_py import (SequentialReader, StorageOptions,
                            ConverterOptions, StorageFilter)
except ImportError:
    sys.exit("rosbag2_py not available; source ROS2 first")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--left_in_topic",  default="/zed/zed_node/left/camera_info")
    p.add_argument("--right_in_topic", default="/zed/zed_node/right/camera_info")
    p.add_argument("--left_out_topic",  default="/left/camera_info_full")
    p.add_argument("--right_out_topic", default="/right/camera_info_full")
    p.add_argument("--hold_seconds", type=float, default=60.0)
    return p.parse_args()


def detect_storage_id(bag_path: str) -> str:
    bag = Path(bag_path)
    if not bag.exists():
        sys.exit(f"--bag does not exist: {bag_path}")
    for child in bag.iterdir():
        if child.suffix == ".db3":
            return "sqlite3"
        if child.suffix == ".mcap":
            return "mcap"
    return "mcap"


def find_first_caminfos(bag_path, left_topic, right_topic):
    """Return (left_msg, right_msg) — first CameraInfo on each side."""
    storage_id = detect_storage_id(bag_path)
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"))
    reader.set_filter(StorageFilter(topics=[left_topic, right_topic]))
    left_msg, right_msg = None, None
    while reader.has_next():
        topic, raw, _t_ns = reader.read_next()
        if topic == left_topic and left_msg is None:
            left_msg = deserialize_message(raw, CameraInfo)
        elif topic == right_topic and right_msg is None:
            right_msg = deserialize_message(raw, CameraInfo)
        if left_msg is not None and right_msg is not None:
            break
    return left_msg, right_msg


def main():
    args = parse_args()
    left_in, right_in = find_first_caminfos(
        args.bag, args.left_in_topic, args.right_in_topic)
    if left_in is None:
        sys.exit(f"No messages on {args.left_in_topic} in {args.bag}")
    if right_in is None:
        sys.exit(f"No messages on {args.right_in_topic} in {args.bag}")

    rclpy.init()
    node = rclpy.create_node("bag_caminfo_publisher")
    log = node.get_logger()

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1)
    pub_l = node.create_publisher(CameraInfo, args.left_out_topic, qos)
    pub_r = node.create_publisher(CameraInfo, args.right_out_topic, qos)

    # Re-stamp to "now" so consumers don't see stale bag time
    left_in.header.stamp = node.get_clock().now().to_msg()
    right_in.header.stamp = node.get_clock().now().to_msg()
    pub_l.publish(left_in)
    pub_r.publish(right_in)

    log.info(
        f"Published {args.left_out_topic}: "
        f"{left_in.width}x{left_in.height}, "
        f"fx={left_in.k[0]:.1f} fy={left_in.k[4]:.1f} "
        f"cx={left_in.k[2]:.1f} cy={left_in.k[5]:.1f}")
    log.info(
        f"Published {args.right_out_topic}: "
        f"{right_in.width}x{right_in.height}, "
        f"fx={right_in.k[0]:.1f} fy={right_in.k[4]:.1f} "
        f"cx={right_in.k[2]:.1f} cy={right_in.k[5]:.1f}")
    log.info(f"Holding for {args.hold_seconds:.1f}s...")

    t0 = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - t0 < args.hold_seconds:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
