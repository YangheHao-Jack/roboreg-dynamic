#!/usr/bin/env python3
"""
bag_first_caminfo_publisher.py

Reads the FIRST CameraInfo from a rosbag (left + right) and publishes
them at 10Hz on the same topics qpf2_receiver uses for streaming
(/left/camera_info_rect and /right/camera_info_rect by default).

Used during the IPCAI setup pre-phase: lets IPCAI fetch the camera
intrinsics BEFORE the bag's video stream starts, so the seg model and
scene can be built. When the gate file appears, this exits and
qpf2_receiver takes over per-frame caminfo publishing in lockstep
with the H.264 video frames.

Important: K is rescaled from native bag resolution (1920x1080) to
the qpf2_receiver output resolution (960x576) so the pre-phase
caminfo MATCHES what qpf2_receiver will publish during streaming.
That way IPCAI's scene built during pre-phase remains valid.

Usage
    python bag_first_caminfo_publisher.py \\
        --bag /path/to/bag \\
        --left_in_topic  /zed/zed_node/left/camera_info \\
        --right_in_topic /zed/zed_node/right/camera_info \\
        --left_out_topic  /left/camera_info_rect \\
        --right_out_topic /right/camera_info_rect \\
        --out_w 960 --out_h 576 \\
        --hold_seconds 60.0 \\
        --stop_signal_file /tmp/qpf2_start
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
    p.add_argument("--left_out_topic",  default="/left/camera_info_rect")
    p.add_argument("--right_out_topic", default="/right/camera_info_rect")
    p.add_argument("--out_w", type=int, default=960,
                   help="Target width to rescale K to (matches qpf2_receiver "
                        "output). Set 0 to skip rescale.")
    p.add_argument("--out_h", type=int, default=576,
                   help="Target height to rescale K to (matches qpf2_receiver "
                        "output). Set 0 to skip rescale.")
    p.add_argument("--hold_seconds", type=float, default=60.0)
    p.add_argument("--stop_signal_file", default="",
                   help="Exit when this file appears (in addition to "
                        "hold_seconds timeout).")
    p.add_argument("--rate_hz", type=float, default=10.0)
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


def rescale_caminfo(msg: CameraInfo, target_w: int, target_h: int) -> CameraInfo:
    """Pure-scaling rescale of K and P from (msg.width, msg.height) to
    (target_w, target_h). Returns a new CameraInfo. No-op if already at
    target."""
    if msg.width <= 0 or msg.height <= 0:
        return msg
    if msg.width == target_w and msg.height == target_h:
        return msg
    sx = float(target_w) / float(msg.width)
    sy = float(target_h) / float(msg.height)
    out = CameraInfo()
    out.header = msg.header
    out.width = target_w
    out.height = target_h
    out.distortion_model = msg.distortion_model
    out.d = list(msg.d)
    out.r = list(msg.r)
    out.binning_x = msg.binning_x
    out.binning_y = msg.binning_y
    K = list(msg.k)
    K[0] *= sx; K[1] *= sx; K[2] *= sx
    K[3] *= sy; K[4] *= sy; K[5] *= sy
    out.k = K
    if len(msg.p) == 12:
        P = list(msg.p)
        P[0] *= sx; P[1] *= sx; P[2] *= sx; P[3] *= sx
        P[4] *= sy; P[5] *= sy; P[6] *= sy; P[7] *= sy
        out.p = P
    else:
        out.p = list(msg.p)
    return out


def main():
    args = parse_args()
    left_in, right_in = find_first_caminfos(
        args.bag, args.left_in_topic, args.right_in_topic)
    if left_in is None:
        sys.exit(f"No messages on {args.left_in_topic} in {args.bag}")
    if right_in is None:
        sys.exit(f"No messages on {args.right_in_topic} in {args.bag}")

    if args.out_w > 0 and args.out_h > 0:
        left_msg = rescale_caminfo(left_in, args.out_w, args.out_h)
        right_msg = rescale_caminfo(right_in, args.out_w, args.out_h)
    else:
        left_msg = left_in
        right_msg = right_in

    rclpy.init()
    node = rclpy.create_node("bag_first_caminfo_publisher")
    log = node.get_logger()

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10)
    pub_l = node.create_publisher(CameraInfo, args.left_out_topic, qos)
    pub_r = node.create_publisher(CameraInfo, args.right_out_topic, qos)

    log.info(
        f"Will publish CameraInfo at {args.rate_hz:.1f} Hz on "
        f"'{args.left_out_topic}' / '{args.right_out_topic}' "
        f"({left_msg.width}x{left_msg.height}, "
        f"fx={left_msg.k[0]:.1f} cx={left_msg.k[2]:.1f}) "
        f"for {args.hold_seconds:.1f}s or until stop signal.")

    period = 1.0 / max(args.rate_hz, 1e-3)
    t0 = time.monotonic()
    n = 0
    stop_path = Path(args.stop_signal_file) if args.stop_signal_file else None
    try:
        while (rclpy.ok()
               and time.monotonic() - t0 < max(period, args.hold_seconds)):
            if stop_path is not None and stop_path.exists():
                log.info(f"Stop signal file appeared: {stop_path}; exiting")
                break
            now = node.get_clock().now().to_msg()
            left_msg.header.stamp = now
            right_msg.header.stamp = now
            pub_l.publish(left_msg)
            pub_r.publish(right_msg)
            n += 1
            rclpy.spin_once(node, timeout_sec=period)
    except KeyboardInterrupt:
        pass

    log.info(f"Published {n} CameraInfo pairs over "
             f"{time.monotonic() - t0:.1f}s; exiting")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
