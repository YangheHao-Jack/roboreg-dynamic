#!/usr/bin/env python3
"""
bag_first_joint_publisher.py

Reads ONLY the first joint_state message from a rosbag and publishes
it once with TRANSIENT_LOCAL durability so late subscribers (bake_node)
can pick it up. Then exits.

Used as a sidecar for the pre-bake stage of fp_pipeline_quest_qpf2.

Usage
    python bag_first_joint_publisher.py \\
        --bag /path/to/bag \\
        --joint_topic /lbr/joint_states \\
        --hold_seconds 60.0
"""

import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSDurabilityPolicy, QoSHistoryPolicy)

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState

try:
    from rosbag2_py import (SequentialReader, StorageOptions,
                            ConverterOptions, StorageFilter)
except ImportError:
    sys.exit("rosbag2_py not available; source ROS2 first")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--joint_topic", default="/lbr/joint_states")
    p.add_argument("--hold_seconds", type=float, default=60.0,
                   help="Stay alive after publishing for this many seconds, "
                        "so transient_local subscribers connecting late can "
                        "still pick the message up. 0 = exit immediately.")
    p.add_argument("--stop_signal_file", default="",
                   help="Optional path. If set, exit when this file appears "
                        "(in addition to hold_seconds timeout). Used to "
                        "stop pre-publishing the moment the bag streamer "
                        "is gated to start.")
    return p.parse_args()


def detect_storage_id(bag_path: str) -> str:
    bag = Path(bag_path)
    if not bag.exists():
        sys.exit(f"--bag does not exist: {bag_path}")
    # mcap vs sqlite3
    for child in bag.iterdir():
        if child.suffix == ".db3":
            return "sqlite3"
        if child.suffix == ".mcap":
            return "mcap"
    return "mcap"  # default for newer rosbags


def find_first_joint_state(bag_path: str, joint_topic: str) -> JointState:
    storage_id = detect_storage_id(bag_path)
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"))
    reader.set_filter(StorageFilter(topics=[joint_topic]))

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic == joint_topic:
            return deserialize_message(raw, JointState)
    return None


def main():
    args = parse_args()
    msg = find_first_joint_state(args.bag, args.joint_topic)
    if msg is None:
        sys.exit(f"No messages found on {args.joint_topic} in {args.bag}")

    rclpy.init()
    node = rclpy.create_node("bag_first_joint_publisher")

    # VOLATILE so we're QoS-compatible with subscribers that follow the
    # live-system convention (e.g. bake_node listening on /lbr/joint_states
    # would normally talk to a KUKA FRI reader publishing VOLATILE).
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10)
    pub = node.create_publisher(JointState, args.joint_topic, qos)
    log = node.get_logger()
    log.info(
        f"Will publish JointState on {args.joint_topic} at 10 Hz "
        f"for {args.hold_seconds:.1f}s "
        f"(names={list(msg.name)}, first 3 positions={list(msg.position[:3])})")

    # Publish at 10 Hz for hold_seconds, OR until stop_signal_file appears.
    period = 0.1  # 10 Hz
    t0 = time.monotonic()
    n = 0
    stop_path = Path(args.stop_signal_file) if args.stop_signal_file else None
    try:
        while rclpy.ok() and time.monotonic() - t0 < max(period, args.hold_seconds):
            if stop_path is not None and stop_path.exists():
                log.info(f"Stop signal file appeared: {stop_path}; exiting")
                break
            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            n += 1
            rclpy.spin_once(node, timeout_sec=period)
    except KeyboardInterrupt:
        pass

    log.info(f"Published {n} JointState messages over "
             f"{time.monotonic() - t0:.1f}s; exiting")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()