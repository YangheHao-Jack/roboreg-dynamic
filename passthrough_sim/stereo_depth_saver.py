#!/usr/bin/env python3
"""
stereo_depth_saver.py

Subscribes to /disparity (stereo_msgs/DisparityImage), converts to
metric depth (32FC1), and:

  - republishes /left/depth_image for FoundationPose (always)
  - optionally writes 000NNN.npy files (NVlabs FoundationStereo format)
  - optionally writes coloured depth + disparity visualisation PNGs

Usage
-----
    # republish only (default — what pipelines use)
    python stereo_depth_saver.py --backend ess

    # also save .npy files
    python stereo_depth_saver.py --backend ess \\
        --out_dir /path/to/depth_npy/

    # also save viz PNGs (offline-only, slow)
    python stereo_depth_saver.py --backend fs \\
        --out_dir /path/to/depth_npy/ \\
        --viz_dir /path/to/viz/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy,
)

from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["fs", "ess"], required=True,
                   help="Stereo backend (used only for log messages).")
    p.add_argument("--disparity_topic", default="/disparity")
    p.add_argument("--depth_topic", default="/left/depth_image",
                   help="Topic to republish depth on (32FC1). "
                        "Empty disables republish.")
    p.add_argument("--depth_frame_id", default="zed_left_camera_optical_frame",
                   help="frame_id to stamp on the republished depth_image. "
                        "ESS leaves DisparityImage.image.header.frame_id "
                        "empty, which breaks FP's selector that matches "
                        "(stamp, frame_id) across image/depth/seg/cinfo. "
                        "Set this to the same frame_id qpf2_receiver uses. "
                        "Empty string disables the override.")
    p.add_argument("--out_dir", default="",
                   help="If set, save 000NNN.npy float32 depth files here.")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--zero_pad", type=int, default=6)
    p.add_argument("--min_disparity", type=float, default=1e-3)
    p.add_argument("--max_depth_m", type=float, default=10.0)
    p.add_argument("--viz_dir", default="",
                   help="If set, save coloured depth+disparity PNGs here. "
                        "Creates depth_viz/ and disparity_viz/ subdirs.")
    p.add_argument("--viz_min_depth_m", type=float, default=0.3)
    p.add_argument("--viz_max_depth_m", type=float, default=3.0)
    p.add_argument("--viz_max_disparity", type=float, default=200.0)
    p.add_argument("--viz_colormap", default="JET",
                   help="OpenCV colormap (JET, TURBO, VIRIDIS, MAGMA, ...)")
    p.add_argument("--pid_file", default="",
                   help="If set, write own PID to this file at startup. "
                        "Used by --init_only fp_pose_recorder for clean shutdown.")
    return p.parse_args()


def disparity_to_depth(disp, f, T, min_disp, max_depth_m):
    """depth = f * T / disp, with masking. f in pixels, T in metres."""
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > min_disp
    depth[valid] = (f * T) / disp[valid]
    depth[depth > max_depth_m] = 0.0
    depth[depth < 0] = 0.0
    return depth


def numpy_to_image_msg(arr, encoding, header):
    msg = Image()
    msg.header = header
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * (4 if encoding == "32FC1" else 1)
    msg.data = arr.tobytes()
    return msg


def wipe_dir(path: Path, ext: str):
    """Delete all files matching *ext in path, return count wiped."""
    n = 0
    if path.exists():
        for f in path.iterdir():
            if f.is_file() and f.suffix == ext:
                f.unlink()
                n += 1
    return n


class StereoDepthSaver(Node):

    def __init__(self, args):
        super().__init__("stereo_depth_saver")
        self.args = args
        self.frame_idx = args.start_frame

        # ── Output dir for .npy ─────────────────────────────────────
        self.out_dir: Optional[Path] = None
        if args.out_dir and args.out_dir.lower() != "none":
            self.out_dir = Path(args.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            wiped = wipe_dir(self.out_dir, ".npy")
            self.get_logger().info(
                f"[{args.backend.upper()}] Saving depth .npy to {self.out_dir}"
                + (f" (wiped {wiped} stale files)" if wiped else ""))
        else:
            self.get_logger().info(
                f"[{args.backend.upper()}] Republish-only mode "
                f"(no .npy save)")

        # ── Visualisation ───────────────────────────────────────────
        self._cv2 = None
        self.viz_depth_dir: Optional[Path] = None
        self.viz_disp_dir: Optional[Path] = None
        self.viz_cmap = None
        if args.viz_dir:
            import cv2
            self._cv2 = cv2
            root = Path(args.viz_dir)
            self.viz_depth_dir = root / "depth_viz"
            self.viz_disp_dir = root / "disparity_viz"
            self.viz_depth_dir.mkdir(parents=True, exist_ok=True)
            self.viz_disp_dir.mkdir(parents=True, exist_ok=True)
            wipe_dir(self.viz_depth_dir, ".png")
            wipe_dir(self.viz_disp_dir, ".png")
            self.viz_cmap = getattr(
                cv2, f"COLORMAP_{args.viz_colormap.upper()}",
                cv2.COLORMAP_JET)
            self.get_logger().info(
                f"VIZ MODE: writing PNGs to {root} "
                f"(cmap={args.viz_colormap})")

        # ── QoS + topics ────────────────────────────────────────────
        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50)
        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)

        self.pub_depth = None
        if args.depth_topic:
            self.pub_depth = self.create_publisher(
                Image, args.depth_topic, pub_qos)
            self.get_logger().info(f"Republishing on {args.depth_topic}")

        self.create_subscription(
            DisparityImage, args.disparity_topic, self._on_disparity, sub_qos)
        self.get_logger().info(f"Subscribed to {args.disparity_topic}")

    def _on_disparity(self, msg: DisparityImage):
        img = msg.image
        if img.encoding != "32FC1":
            self.get_logger().warn(f"Unexpected encoding {img.encoding}")
            return

        h, w = img.height, img.width
        disp = np.frombuffer(img.data, dtype=np.float32).reshape(h, w).copy()
        f, T = float(msg.f), float(msg.t)
        if f <= 0 or T <= 0:
            self.get_logger().warn(f"Invalid f={f} T={T}")
            return

        depth = disparity_to_depth(
            disp, f, T, self.args.min_disparity, self.args.max_depth_m)

        # Save .npy (optional)
        out_path: Optional[Path] = None
        if self.out_dir is not None:
            stem = f"{self.frame_idx:0{self.args.zero_pad}d}"
            out_path = self.out_dir / f"{stem}.npy"
            np.save(str(out_path), depth)

        # Republish for FP
        if self.pub_depth is not None:
            depth_msg = numpy_to_image_msg(depth, "32FC1", img.header)
            # ESS leaves frame_id empty, which breaks FP's selector
            # (it matches (stamp, frame_id) across image/depth/seg/cinfo).
            # Override with the ZED convention so depth matches the others.
            if self.args.depth_frame_id:
                depth_msg.header.frame_id = self.args.depth_frame_id
            self.pub_depth.publish(depth_msg)

        # Visualisation PNGs (optional)
        if self._cv2 is not None:
            stem = f"{self.frame_idx:0{self.args.zero_pad}d}"
            self._save_viz(stem, depth, disp)

        # Periodic log
        if self.frame_idx == 0 or self.frame_idx % 25 == 0:
            valid = int((depth > 0).sum())
            d_min = depth[depth > 0].min() if valid else 0.0
            tail = f" → {out_path.name}" if out_path else " (republish)"
            self.get_logger().info(
                f"Frame {self.frame_idx}: {h}x{w}  f={f:.1f}  T={T*1000:.1f}mm  "
                f"valid={valid}/{h*w}  range=[{d_min:.2f},{depth.max():.2f}]m"
                f"{tail}")

        self.frame_idx += 1

    def _save_viz(self, stem: str, depth: np.ndarray, disp: np.ndarray):
        a = self.args
        cv2 = self._cv2

        depth_viz = self._colorise(
            depth, a.viz_min_depth_m, a.viz_max_depth_m,
            invalid=(depth <= 0))
        cv2.imwrite(str(self.viz_depth_dir / f"{stem}.png"), depth_viz)

        disp_viz = self._colorise(
            disp, 0.0, a.viz_max_disparity,
            invalid=(disp <= a.min_disparity))
        cv2.imwrite(str(self.viz_disp_dir / f"{stem}.png"), disp_viz)

    def _colorise(self, arr, vmin, vmax, invalid=None):
        cv2 = self._cv2
        a = np.clip(arr.astype(np.float32, copy=True), vmin, vmax)
        a = (a - vmin) / max(1e-9, vmax - vmin)
        a = (a * 255.0).astype(np.uint8)
        rgb = cv2.applyColorMap(a, self.viz_cmap)
        if invalid is not None:
            rgb[invalid] = 0
        return rgb


def main():
    args = parse_args()
    if args.pid_file:
        import os
        from pathlib import Path as _P
        _pf = _P(args.pid_file)
        _pf.parent.mkdir(parents=True, exist_ok=True)
        _pf.write_text(str(os.getpid()))
    rclpy.init()
    node = StereoDepthSaver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Done. Frames processed: {node.frame_idx - args.start_frame}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
