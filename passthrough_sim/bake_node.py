#!/usr/bin/env python3
"""
bake_node.py

One-shot ROS2 bake. Subscribes to /lbr/joint_states, snapshots the FIRST
message, runs bake_from_joint_dict (URDF + FK + mesh concatenation +
bbox centering), writes the baked .obj + offset.npy + sentinel, exits.

Designed to run BEFORE FoundationPose in a launch chain. Use the
sentinel as the OnProcessExit trigger to start downstream stages.

Usage
    python bake_node.py \\
        --urdf_path /path/to/lbr_med7_r800.urdf \\
        --out_dir   /tmp/fp_bake_runtime \\
        --joint_state_topic /lbr/joint_states \\
        --joint_name_prefix lbr_A \\
        --timeout 30.0

Exit codes
    0 — bake successful, sentinel written
    1 — bake failed (URDF, mesh load, FK, etc.)
    2 — timed out waiting for joint_state message
"""

import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSDurabilityPolicy, QoSHistoryPolicy)

from sensor_msgs.msg import JointState

# Allow `from bake_lib import ...` when both files live in the same dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_lib import bake_from_joint_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf_path", required=True,
                   help="Path to the robot URDF.")
    p.add_argument("--out_dir", default="/tmp/fp_bake_runtime",
                   help="Where to write baked .obj and offset.npy.")
    p.add_argument("--obj_name", default="lbr_med7_baked")
    p.add_argument("--joint_state_topic", default="/lbr/joint_states")
    p.add_argument("--joint_name_prefix", default="lbr_A",
                   help="Filter joint_state.name entries with this prefix. "
                        "Use empty string to take all names.")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Seconds to wait for the first joint_state message.")
    p.add_argument("--sentinel_filename", default=".bake_done")
    return p.parse_args()


class BakeNode(Node):

    def __init__(self, args):
        super().__init__("bake_node")
        self.args = args
        self.done = False
        self.success = False

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)
        self.create_subscription(
            JointState, args.joint_state_topic, self._on_joint_state, qos)
        self.get_logger().info(
            f"bake_node listening on {args.joint_state_topic} "
            f"(timeout {args.timeout:.1f}s)")
        self.get_logger().info(
            f"URDF:    {args.urdf_path}")
        self.get_logger().info(
            f"Output:  {args.out_dir}/{args.obj_name}.{{obj,offset.npy}}")

    def _on_joint_state(self, msg: JointState):
        if self.done:
            return
        if not msg.name or len(msg.position) != len(msg.name):
            self.get_logger().warn(
                f"joint_state with name={len(msg.name)} pos={len(msg.position)}; "
                f"waiting for a complete one")
            return

        prefix = self.args.joint_name_prefix
        joint_dict = {n: float(p) for n, p in zip(msg.name, msg.position)
                      if (not prefix or n.startswith(prefix))}
        if not joint_dict:
            self.get_logger().warn(
                f"joint_state had no names matching prefix '{prefix}': "
                f"{msg.name}")
            return

        self.get_logger().info(
            f"Got joint_state with {len(joint_dict)} joints. Baking...")
        try:
            obj_path, offset_path, info = bake_from_joint_dict(
                self.args.urdf_path, joint_dict,
                self.args.out_dir, self.args.obj_name,
                log=self.get_logger().info)
        except Exception as e:
            self.get_logger().error(f"Bake failed: {e}")
            self.success = False
            self.done = True
            return

        # Sentinel last (after the .obj is on disk)
        sentinel = Path(self.args.out_dir) / self.args.sentinel_filename
        sentinel.write_text(
            f"obj={obj_path}\n"
            f"offset={offset_path}\n"
            f"n_verts={info['n_verts']}\n"
            f"n_faces={info['n_faces']}\n"
            f"bbox_center={info['bbox_center']}\n")
        self.get_logger().info(
            f"Bake complete. Sentinel: {sentinel}")
        self.success = True
        self.done = True


def main():
    args = parse_args()

    if not Path(args.urdf_path).exists():
        sys.exit(f"--urdf_path does not exist: {args.urdf_path}")

    # Wipe stale outputs (sentinel, mesh, offset) so OnProcessExit
    # downstream sees a fresh result
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in (args.sentinel_filename,
                  f"{args.obj_name}.obj",
                  f"{args.obj_name}_offset.npy"):
        p = out_dir / fname
        if p.exists():
            p.unlink()

    rclpy.init()
    node = BakeNode(args)

    t0 = time.monotonic()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - t0 > args.timeout:
                node.get_logger().error(
                    f"Timed out after {args.timeout:.1f}s waiting for "
                    f"joint_state on {args.joint_state_topic}")
                node.destroy_node()
                rclpy.shutdown()
                sys.exit(2)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if node.success else 1)


if __name__ == "__main__":
    main()