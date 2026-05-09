#!/usr/bin/env python3
"""
bag_joint_replayer.py

Replays /lbr/joint_states (or any chosen topic) from a rosbag at the
bag's recorded rate. Gated by --start_signal_file, identical to
bag_to_qpf2.py — touch the gate file to start both at the same time.

Usage
    python bag_joint_replayer.py \\
        --bag /path/to/bag \\
        --joint_topic /lbr/joint_states \\
        --rate 1.0 \\
        --loop false \\
        --start_signal_file /tmp/qpf2_start
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
    p.add_argument("--rate", type=float, default=1.0,
                   help="Playback rate multiplier. 1.0 = original bag rate.")
    p.add_argument("--loop", default="false", choices=["true", "false"])
    p.add_argument("--start_signal_file", default="",
                   help="If set, wait until this file exists before publishing.")
    p.add_argument("--restamp", default="true", choices=["true", "false"],
                   help="If true, set msg.header.stamp = wall clock at "
                        "publish time. If false, keep bag's original stamp.")
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


def wait_for_gate(gate_path: str, logger):
    if not gate_path:
        return
    p = Path(gate_path)
    if p.exists():
        return
    logger.info(f"Waiting for gate file: {gate_path}")
    while not p.exists():
        time.sleep(0.1)
    logger.info("Gate file detected; starting playback")


def open_reader(bag_path: str, joint_topic: str):
    storage_id = detect_storage_id(bag_path)
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"))
    reader.set_filter(StorageFilter(topics=[joint_topic]))
    return reader


def main():
    args = parse_args()
    loop = args.loop.lower() == "true"
    restamp = args.restamp.lower() == "true"

    rclpy.init()
    node = rclpy.create_node("bag_joint_replayer")

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10)
    pub = node.create_publisher(JointState, args.joint_topic, qos)
    log = node.get_logger()
    log.info(f"Publisher ready on {args.joint_topic}")

    wait_for_gate(args.start_signal_file, log)

    while rclpy.ok():
        reader = open_reader(args.bag, args.joint_topic)

        first_bag_t_ns = None
        wall_t0 = None
        n = 0

        while reader.has_next() and rclpy.ok():
            topic, raw, t_ns = reader.read_next()
            if topic != args.joint_topic:
                continue

            if first_bag_t_ns is None:
                first_bag_t_ns = t_ns
                wall_t0 = time.monotonic()

            # Pace to bag time, scaled by rate
            elapsed_bag = (t_ns - first_bag_t_ns) / 1e9
            target_wall = elapsed_bag / max(1e-9, args.rate)
            sleep_for = target_wall - (time.monotonic() - wall_t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

            msg = deserialize_message(raw, JointState)
            if restamp:
                msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            n += 1

        log.info(f"Bag exhausted after {n} joint messages")
        if not loop:
            break
        log.info("Looping...")

    log.info(f"Done. Published {n} messages total.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
