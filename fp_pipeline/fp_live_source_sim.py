#!/usr/bin/env python3
"""
fp_live_source_sim.py

Simulated live source for the FP pipeline. This is a drop-in for the
*offline publisher's roles* in the existing fp_pipeline.launch.py:
publishes RGB + camera_info + segmentation on the same topics the
offline publisher uses, but free-running at 30 Hz (no recorder
backpressure).

The offline pipeline graph is reused as-is:
    ESS via isaac_ros_ess.launch.py    (no remappings)
    stereo_depth_saver.py              (with --out_dir "" to disable .npy)
    isaac_ros_foundationpose_med7.launch.py

This sim publishes EXACTLY what the offline publisher does at processing
resolution (960×576), since the offline pipeline already operates at
that resolution and the FP graph is wired to those topic names.

Topics published (matches offline publisher defaults):
    /left/image_rect              (bgr8, 960×576, sensor QoS)
    /left/camera_info_rect        (CameraInfo at 960×576)
    /left/camera_info             (alias — ESS subscribes here)
    /right/image_rect             (bgr8, 960×576, sensor QoS)
    /right/camera_info_rect
    /right/camera_info            (alias)
    /left/segmentation            (mono8, 960×576, latched)

Optional full-res topics (for recorder overlay drawing) — controlled by
--publish_full:
    /left/image_full
    /left/camera_info_full
    /right/image_full
    /right/camera_info_full

QoS: RELIABLE+VOLATILE for sensor topics (matches offline publisher
choice). The ESS launch's NITROS subscribers accept this without issue
because the offline pipeline uses the same QoS.

Usage:
    python ~/fp_pipeline/fp_live_source_sim.py \\
        --image_dir       "/.../images/left" \\
        --right_image_dir "/.../images/right" \\
        --camera_yaml       /.../camera.left.image.camera_info_4.yaml \\
        --right_camera_yaml /.../camera.right.image.camera_info_4.yaml \\
        --mask_path  "/.../mask_left_0.png" \\
        --rate 30.0 --on_end loop --preload_ram
"""

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml as pyyaml

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header


# ──────────────────────────────────────────────────────────────────────────
# Helpers (consistent with fp_offline_publisher.py)
# ──────────────────────────────────────────────────────────────────────────


def list_frames(image_dir: Path, side: str = "left"):
    rx = re.compile(rf"camera_image_{side}_(\d+)\.png$")
    pairs = []
    for p in image_dir.iterdir():
        m = rx.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_camera_info_from_yaml(yaml_path: Path, frame_id: str) -> CameraInfo:
    with open(yaml_path) as f:
        d = pyyaml.safe_load(f)

    def _get(*keys):
        for k in keys:
            if k in d:
                return d[k]
        sys.exit(f"YAML missing key. Tried: {keys}. Have: {list(d.keys())}")

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.height = int(_get("height"))
    msg.width = int(_get("width"))
    K = _get("k", "K", "camera_matrix")
    if isinstance(K, dict) and "data" in K:
        K = K["data"]
    msg.k = [float(x) for x in K]
    P = _get("p", "P", "projection_matrix")
    if isinstance(P, dict) and "data" in P:
        P = P["data"]
    msg.p = [float(x) for x in P]
    D = _get("d", "D", "distortion_coefficients")
    if isinstance(D, dict) and "data" in D:
        D = D["data"]
    msg.d = [float(x) for x in D]
    R = _get("r", "R", "rectification_matrix")
    if isinstance(R, dict) and "data" in R:
        R = R["data"]
    msg.r = [float(x) for x in R]
    msg.distortion_model = str(_get("distortion_model"))
    return msg


def numpy_to_image_msg(arr: np.ndarray, encoding: str, header: Header) -> Image:
    msg = Image()
    msg.header = header
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding in ("bgr8", "rgb8"):
        msg.step = arr.shape[1] * 3
    elif encoding == "mono8":
        msg.step = arr.shape[1]
    else:
        sys.exit(f"Unsupported encoding {encoding}")
    msg.data = arr.tobytes()
    return msg


# ──────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────


class LiveSourceSim(Node):
    def __init__(self, args):
        super().__init__("fp_live_source_sim")
        self.args = args

        # ── Frame lists ──────────────────────────────────────────────────
        self.image_dir = Path(args.image_dir)
        if not self.image_dir.is_dir():
            sys.exit(f"image_dir not found: {self.image_dir}")
        self.frames = list_frames(self.image_dir, side="left")
        if not self.frames:
            sys.exit(f"No camera_image_left_N.png in {self.image_dir}")

        self.right_image_dir = Path(args.right_image_dir)
        if not self.right_image_dir.is_dir():
            sys.exit(f"right_image_dir not found: {self.right_image_dir}")
        self.right_frames = list_frames(self.right_image_dir, side="right")
        if not self.right_frames:
            sys.exit(f"No camera_image_right_N.png in "
                     f"{self.right_image_dir}")

        # ── Camera info (full-res from YAML) ─────────────────────────────
        self.left_cinfo = load_camera_info_from_yaml(
            Path(args.camera_yaml), args.frame_id
        )
        self.full_W = self.left_cinfo.width
        self.full_H = self.left_cinfo.height
        self.right_cinfo = load_camera_info_from_yaml(
            Path(args.right_camera_yaml), args.right_frame_id
        )

        # ── Processing-resolution scaling (matches offline publisher) ───
        self.proc_W = int(args.proc_width)
        self.proc_H = int(args.proc_height)
        self.scale_x = self.proc_W / self.full_W
        self.scale_y = self.proc_H / self.full_H
        self.get_logger().info(
            f"Source: {self.full_W}x{self.full_H} -> proc "
            f"{self.proc_W}x{self.proc_H} "
            f"(sx={self.scale_x:.4f}, sy={self.scale_y:.4f})"
        )

        # Pre-build proc-res K and P (same as offline publisher)
        K_full = np.array(self.left_cinfo.k, dtype=np.float64).reshape(3, 3)
        P_full = np.array(self.left_cinfo.p, dtype=np.float64).reshape(3, 4)
        K_proc = K_full.copy()
        K_proc[0, 0] *= self.scale_x; K_proc[0, 2] *= self.scale_x
        K_proc[1, 1] *= self.scale_y; K_proc[1, 2] *= self.scale_y
        P_proc = P_full.copy()
        P_proc[0, 0] *= self.scale_x; P_proc[0, 2] *= self.scale_x
        P_proc[0, 3] *= self.scale_x
        P_proc[1, 1] *= self.scale_y; P_proc[1, 2] *= self.scale_y
        self._left_k_proc = [float(x) for x in K_proc.flatten()]
        self._left_p_proc = [float(x) for x in P_proc.flatten()]

        R_K_full = np.array(self.right_cinfo.k, dtype=np.float64).reshape(3, 3)
        R_P_full = np.array(self.right_cinfo.p, dtype=np.float64).reshape(3, 4)
        R_K_proc = R_K_full.copy()
        R_K_proc[0, 0] *= self.scale_x; R_K_proc[0, 2] *= self.scale_x
        R_K_proc[1, 1] *= self.scale_y; R_K_proc[1, 2] *= self.scale_y
        R_P_proc = R_P_full.copy()
        R_P_proc[0, 0] *= self.scale_x; R_P_proc[0, 2] *= self.scale_x
        R_P_proc[0, 3] *= self.scale_x
        R_P_proc[1, 1] *= self.scale_y; R_P_proc[1, 2] *= self.scale_y
        self._right_k_proc = [float(x) for x in R_K_proc.flatten()]
        self._right_p_proc = [float(x) for x in R_P_proc.flatten()]

        # ── Mask (resized to proc res, NEAREST) ──────────────────────────
        mask = cv2.imread(str(args.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            sys.exit(f"Failed to read mask: {args.mask_path}")
        mask_bin = (mask > 127).astype(np.uint8) * 255
        if mask_bin.shape != (self.proc_H, self.proc_W):
            mask_bin = cv2.resize(mask_bin, (self.proc_W, self.proc_H),
                                  interpolation=cv2.INTER_NEAREST)
        self.mask_arr = mask_bin
        self.get_logger().info(
            f"Mask: {self.mask_arr.shape}, "
            f"foreground={(self.mask_arr > 0).sum()} px"
        )

        # ── QoS — match offline publisher exactly ───────────────────────
        # The offline publisher uses RELIABLE+VOLATILE+depth=5. ESS, FP,
        # and stereo_depth_saver all happily subscribe to that QoS in the
        # offline pipeline. Don't second-guess this.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Publishers ──────────────────────────────────────────────────
        # Left
        self.pub_left_rgb_rect   = self.create_publisher(
            Image,      args.left_rgb_topic_rect,   sensor_qos)
        self.pub_left_cinfo_rect = self.create_publisher(
            CameraInfo, args.left_cinfo_topic_rect, sensor_qos)
        self.pub_left_cinfo_alias = self.create_publisher(
            CameraInfo, args.left_cinfo_topic_alias, sensor_qos)
        # Right
        self.pub_right_rgb_rect   = self.create_publisher(
            Image,      args.right_rgb_topic_rect,   sensor_qos)
        self.pub_right_cinfo_rect = self.create_publisher(
            CameraInfo, args.right_cinfo_topic_rect, sensor_qos)
        self.pub_right_cinfo_alias = self.create_publisher(
            CameraInfo, args.right_cinfo_topic_alias, sensor_qos)
        # Mask — same QoS as the offline publisher (RELIABLE+VOLATILE),
        # which the FP selector is known to subscribe to successfully.
        # Published every tick (matches offline publisher behaviour). FP
        # itself runs at ~10 fps on RTX 5090 per NVIDIA's benchmarks,
        # so it's the bottleneck regardless of mask cost — and per-tick
        # republish avoids any startup race against FP's TRT engine load.
        self.pub_mask = self.create_publisher(
            Image, args.mask_topic, sensor_qos)
        # Optional full-res (for recorder overlay)
        self.publish_full = bool(args.publish_full)
        if self.publish_full:
            self.pub_left_rgb_full   = self.create_publisher(
                Image,      args.left_rgb_topic_full,   sensor_qos)
            self.pub_left_cinfo_full = self.create_publisher(
                CameraInfo, args.left_cinfo_topic_full, sensor_qos)
            self.pub_right_rgb_full   = self.create_publisher(
                Image,      args.right_rgb_topic_full,   sensor_qos)
            self.pub_right_cinfo_full = self.create_publisher(
                CameraInfo, args.right_cinfo_topic_full, sensor_qos)

        # ── Optional RAM preload ────────────────────────────────────────
        # PNG decode is the ~16 Hz bottleneck on first attempts; preload
        # is required to sustain 30 Hz with proc-res publish.
        self.preload = (args.preload_ram == "true")
        if self.preload:
            self.get_logger().info(
                f"Preloading {len(self.frames)} left + "
                f"{len(self.right_frames)} right frames..."
            )
            t0 = time.monotonic()
            # Decode at full-res, resize to proc-res once, cache.
            self._left_cache_proc = []
            self._left_cache_full = [] if self.publish_full else None
            for _, p in self.frames:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    im = np.zeros((self.full_H, self.full_W, 3), np.uint8)
                if self.publish_full:
                    self._left_cache_full.append(im)
                self._left_cache_proc.append(
                    cv2.resize(im, (self.proc_W, self.proc_H),
                               interpolation=cv2.INTER_AREA)
                )
            self._right_cache_proc = []
            self._right_cache_full = [] if self.publish_full else None
            for _, p in self.right_frames:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    im = np.zeros((self.right_cinfo.height,
                                   self.right_cinfo.width, 3), np.uint8)
                if self.publish_full:
                    self._right_cache_full.append(im)
                self._right_cache_proc.append(
                    cv2.resize(im, (self.proc_W, self.proc_H),
                               interpolation=cv2.INTER_AREA)
                )
            elapsed = time.monotonic() - t0
            self.get_logger().info(
                f"Preload done in {elapsed:.1f} s "
                f"(proc {len(self._left_cache_proc)} left, "
                f"{len(self._right_cache_proc)} right; "
                f"full {'cached' if self.publish_full else 'skipped'})"
            )
        else:
            self._left_cache_proc = None
            self._right_cache_proc = None
            self._left_cache_full = None
            self._right_cache_full = None

        # ── Replay state ────────────────────────────────────────────────
        self.idx = 0
        self.tick = 0
        self.n = min(len(self.frames), len(self.right_frames))
        if len(self.right_frames) != len(self.frames):
            self.get_logger().warn(
                f"Right has {len(self.right_frames)} frames, left has "
                f"{len(self.frames)}. Replaying min={self.n}."
            )
        self.rate = float(args.rate)
        if self.rate <= 0:
            sys.exit("--rate must be > 0")
        self.period = 1.0 / self.rate
        self.on_end = args.on_end

        self.get_logger().info(
            f"Replay: {self.n} frames @ {self.rate:.2f} Hz, "
            f"on_end={self.on_end}"
        )

        self.timer = self.create_timer(self.period, self._on_tick)
        self._t_start = time.monotonic()

    # ----------------------------------------------------------------
    def _read_pair(self, i: int):
        """Returns (left_proc, right_proc, left_full_or_None, right_full_or_None)."""
        if self.preload:
            l_proc = self._left_cache_proc[i]
            r_proc = self._right_cache_proc[i]
            l_full = self._left_cache_full[i] if self.publish_full else None
            r_full = self._right_cache_full[i] if self.publish_full else None
        else:
            _, lpath = self.frames[i]
            _, rpath = self.right_frames[i]
            l_full = cv2.imread(str(lpath), cv2.IMREAD_COLOR)
            r_full = cv2.imread(str(rpath), cv2.IMREAD_COLOR)
            if l_full is None:
                l_full = np.zeros((self.full_H, self.full_W, 3), np.uint8)
            if r_full is None:
                r_full = np.zeros((self.right_cinfo.height,
                                   self.right_cinfo.width, 3), np.uint8)
            l_proc = cv2.resize(l_full, (self.proc_W, self.proc_H),
                                interpolation=cv2.INTER_AREA)
            r_proc = cv2.resize(r_full, (self.proc_W, self.proc_H),
                                interpolation=cv2.INTER_AREA)
            if not self.publish_full:
                l_full = None
                r_full = None
        return l_proc, r_proc, l_full, r_full

    # ----------------------------------------------------------------
    def _on_tick(self):
        if self.idx >= self.n:
            if self.on_end == "loop":
                self.idx = 0
                self.get_logger().info(
                    f"End of {self.n} frames; looping. "
                    f"({self.tick} published total)"
                )
            elif self.on_end == "hold_last":
                self.idx = self.n - 1
            elif self.on_end == "stop":
                if self.timer is not None:
                    self.timer.cancel()
                    self.timer = None
                self.get_logger().info(
                    f"End after {self.tick} frames; stopping."
                )
                return
            else:
                sys.exit(f"Unknown --on_end: {self.on_end}")

        l_proc, r_proc, l_full, r_full = self._read_pair(self.idx)
        now = self.get_clock().now().to_msg()

        # Build headers
        lhdr = Header(); lhdr.stamp = now; lhdr.frame_id = self.args.frame_id
        rhdr = Header(); rhdr.stamp = now
        rhdr.frame_id = self.args.right_frame_id

        # Left proc-res messages
        left_rgb_rect = numpy_to_image_msg(l_proc, "bgr8", lhdr)
        left_ci_rect = CameraInfo()
        left_ci_rect.header = lhdr
        left_ci_rect.height = self.proc_H
        left_ci_rect.width  = self.proc_W
        left_ci_rect.k = self._left_k_proc
        left_ci_rect.p = self._left_p_proc
        left_ci_rect.d = list(self.left_cinfo.d)
        left_ci_rect.r = list(self.left_cinfo.r)
        left_ci_rect.distortion_model = self.left_cinfo.distortion_model

        # Right proc-res messages
        right_rgb_rect = numpy_to_image_msg(r_proc, "bgr8", rhdr)
        right_ci_rect = CameraInfo()
        right_ci_rect.header = rhdr
        right_ci_rect.height = self.proc_H
        right_ci_rect.width  = self.proc_W
        right_ci_rect.k = self._right_k_proc
        right_ci_rect.p = self._right_p_proc
        right_ci_rect.d = list(self.right_cinfo.d)
        right_ci_rect.r = list(self.right_cinfo.r)
        right_ci_rect.distortion_model = self.right_cinfo.distortion_model

        # Publish back-to-back
        # Left
        self.pub_left_rgb_rect.publish(left_rgb_rect)
        self.pub_left_cinfo_rect.publish(left_ci_rect)
        self.pub_left_cinfo_alias.publish(left_ci_rect)
        # Right
        self.pub_right_rgb_rect.publish(right_rgb_rect)
        self.pub_right_cinfo_rect.publish(right_ci_rect)
        self.pub_right_cinfo_alias.publish(right_ci_rect)

        # Mask: publish every tick, matching the offline publisher.
        # FP at 720p maxes out at ~10 fps on RTX 5090 (per NVIDIA's
        # benchmarks), so it's downstream-bound regardless. The per-tick
        # mask cost is negligible compared to FP's processing time, and
        # republishing every frame removes any startup race against
        # FP's TRT engine load (which can take several seconds).
        mask_msg = numpy_to_image_msg(self.mask_arr, "mono8", lhdr)
        self.pub_mask.publish(mask_msg)

        # Optional full-res
        if self.publish_full:
            l_full_msg = numpy_to_image_msg(l_full, "bgr8", lhdr)
            l_ci_full = CameraInfo()
            l_ci_full.header = lhdr
            l_ci_full.height = self.full_H
            l_ci_full.width  = self.full_W
            l_ci_full.k = list(self.left_cinfo.k)
            l_ci_full.p = list(self.left_cinfo.p)
            l_ci_full.d = list(self.left_cinfo.d)
            l_ci_full.r = list(self.left_cinfo.r)
            l_ci_full.distortion_model = self.left_cinfo.distortion_model

            r_full_msg = numpy_to_image_msg(r_full, "bgr8", rhdr)
            r_ci_full = CameraInfo()
            r_ci_full.header = rhdr
            r_ci_full.height = self.right_cinfo.height
            r_ci_full.width  = self.right_cinfo.width
            r_ci_full.k = list(self.right_cinfo.k)
            r_ci_full.p = list(self.right_cinfo.p)
            r_ci_full.d = list(self.right_cinfo.d)
            r_ci_full.r = list(self.right_cinfo.r)
            r_ci_full.distortion_model = self.right_cinfo.distortion_model

            self.pub_left_rgb_full.publish(l_full_msg)
            self.pub_left_cinfo_full.publish(l_ci_full)
            self.pub_right_rgb_full.publish(r_full_msg)
            self.pub_right_cinfo_full.publish(r_ci_full)

        self.idx += 1
        self.tick += 1

        if self.tick == 1 or self.tick % 60 == 0:
            elapsed = time.monotonic() - self._t_start
            actual_hz = self.tick / max(elapsed, 1e-6)
            self.get_logger().info(
                f"Tick {self.tick} (idx {self.idx-1}/{self.n}): "
                f"actual {actual_hz:.2f} Hz vs target "
                f"{self.rate:.2f} Hz"
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--right_image_dir", required=True)
    p.add_argument("--camera_yaml", required=True)
    p.add_argument("--right_camera_yaml", required=True)
    p.add_argument("--mask_path", required=True)
    p.add_argument("--frame_id", default="zed_left_camera_optical_frame")
    p.add_argument("--right_frame_id",
                   default="zed_right_camera_optical_frame")
    # Topic defaults match fp_offline_publisher.py defaults exactly so
    # the offline FP graph and ESS launch consume them unchanged.
    p.add_argument("--left_rgb_topic_rect",   default="/left/image_rect")
    p.add_argument("--left_cinfo_topic_rect", default="/left/camera_info_rect")
    p.add_argument("--left_cinfo_topic_alias", default="/left/camera_info",
                   help="ESS subscribes to bare /left/camera_info; FS uses "
                        "the _rect suffix. Publish on both for backend "
                        "compatibility.")
    p.add_argument("--right_rgb_topic_rect",   default="/right/image_rect")
    p.add_argument("--right_cinfo_topic_rect",
                   default="/right/camera_info_rect")
    p.add_argument("--right_cinfo_topic_alias", default="/right/camera_info")
    p.add_argument("--mask_topic", default="/left/segmentation")
    # Optional full-res for recorder overlay
    p.add_argument("--left_rgb_topic_full",    default="/left/image_full")
    p.add_argument("--left_cinfo_topic_full",  default="/left/camera_info_full")
    p.add_argument("--right_rgb_topic_full",   default="/right/image_full")
    p.add_argument("--right_cinfo_topic_full",
                   default="/right/camera_info_full")
    p.add_argument("--publish_full", type=int, default=1,
                   help="Also publish full-res images (for recorder overlay). "
                        "Default 1.")
    # Resolutions
    p.add_argument("--proc_width",  type=int, default=960)
    p.add_argument("--proc_height", type=int, default=576)
    # Replay
    p.add_argument("--rate", type=float, default=30.0)
    p.add_argument("--on_end", choices=["loop", "hold_last", "stop"],
                   default="loop")
    p.add_argument("--preload_ram", default="false", choices=["true", "false"],
                   help="Preload all frames to RAM (required for 30 Hz).")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = LiveSourceSim(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()