#!/usr/bin/env python3
"""
fp_offline_publisher.py  (one-frame-at-a-time version)

Walks PNG frames and publishes RGB / camera_info / depth / segmentation to
the topics Isaac ROS FoundationPose's selector subscribes to. After each
frame, it WAITS for the recorder to write the matching pose .npy on disk
before publishing the next frame. This removes any race between
publish-rate and FP's processing rate.

How it knows when the next frame can be published:
  - The recorder writes pose_cam_to_link0/000NNN.npy for each pose received.
  - We pre-count files in that directory before publishing frame i, then
    poll until the count increases (or timeout fires).

Usage:
    python fp_offline_publisher.py \\
        --image_dir   ".../images/left" \\
        --depth_dir   ".../depth_fs/depth" \\
        --camera_yaml ".../camera.left.image.camera_info_4.yaml" \\
        --mask_path   ".../mask_left_0.png" \\
        --recorder_out_dir ~/FoundationPose_assets/poses_disp01/
"""

import argparse
import re
import sys
import threading
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
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def list_frames(image_dir: Path, side: str = "left"):
    """List camera_image_<side>_N.png files in numeric order."""
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
        assert arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3
        msg.step = arr.shape[1] * 3
    elif encoding == "mono8":
        assert arr.dtype == np.uint8 and arr.ndim == 2
        msg.step = arr.shape[1]
    elif encoding == "32FC1":
        assert arr.dtype == np.float32 and arr.ndim == 2
        msg.step = arr.shape[1] * 4
    else:
        sys.exit(f"Unsupported encoding {encoding}")
    msg.data = arr.tobytes()
    return msg


# ──────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────


class OneAtATimePublisher(Node):
    def __init__(self, args):
        super().__init__("fp_offline_publisher")
        self.args = args

        self.image_dir = Path(args.image_dir)
        self.depth_dir = Path(args.depth_dir) if args.depth_dir else None
        self.do_publish_depth = self.depth_dir is not None
        self.mask_path = Path(args.mask_path)
        self.recorder_link0_dir = (
            Path(args.recorder_out_dir) / "pose_cam_to_link0"
        )
        self.recorder_link0_dir.mkdir(parents=True, exist_ok=True)

        check_paths = [(self.image_dir, "image_dir"),
                       (self.mask_path, "mask_path"),
                       (Path(args.camera_yaml), "camera_yaml")]
        if self.do_publish_depth:
            check_paths.append((self.depth_dir, "depth_dir"))
        for p, label in check_paths:
            if not p.exists():
                sys.exit(f"{label} not found: {p}")

        # Camera info
        self.cinfo_template = load_camera_info_from_yaml(
            Path(args.camera_yaml), args.frame_id
        )
        self.full_W = self.cinfo_template.width
        self.full_H = self.cinfo_template.height
        self.get_logger().info(
            f"Original camera info: {self.full_W}x{self.full_H}, "
            f"fx={self.cinfo_template.k[0]:.2f}"
        )

        # Decide processing resolution (the size FS and FP consume).
        # Defaults to FS native: 960x576. The publisher publishes images
        # in BOTH full-res (on _full topics) and processing-res (on _rect
        # topics) so that the recorder can do full-res overlay drawing
        # while FS/FP run at the smaller, faster size.
        self.proc_W = int(args.proc_width)
        self.proc_H = int(args.proc_height)
        # Independent x/y scale factors (publisher does NON-isotropic
        # resize when the aspect ratio of full vs proc differs).
        self.scale_x = self.proc_W / self.full_W
        self.scale_y = self.proc_H / self.full_H
        self.get_logger().info(
            f"Dual-publish: full {self.full_W}x{self.full_H} -> "
            f"proc {self.proc_W}x{self.proc_H}  "
            f"(scale_x={self.scale_x:.4f}, scale_y={self.scale_y:.4f})"
        )

        # Pre-build BOTH full-res and processing-res CameraInfo lists.
        # K_full / P_full come straight from the YAML.
        K_full = np.array(self.cinfo_template.k, dtype=np.float64).reshape(3, 3)
        self._k_full = [float(x) for x in K_full.flatten()]
        P_full = np.array(self.cinfo_template.p, dtype=np.float64).reshape(3, 4)
        self._p_full = [float(x) for x in P_full.flatten()]
        self._d_full = list(self.cinfo_template.d)
        self._r_full = list(self.cinfo_template.r)
        self._dist_model = self.cinfo_template.distortion_model

        # K_proc: scale fx, cx by scale_x; fy, cy by scale_y.
        K_proc = K_full.copy()
        K_proc[0, 0] *= self.scale_x   # fx
        K_proc[0, 2] *= self.scale_x   # cx
        K_proc[1, 1] *= self.scale_y   # fy
        K_proc[1, 2] *= self.scale_y   # cy
        self._k_proc = [float(x) for x in K_proc.flatten()]
        # P (3x4): same idea. P[0,3] is Tx (= -fx*baseline for right cam,
        # 0 for left). For left rectified Tx=0 so multiply is fine; for
        # right cam we still want Tx scaled by scale_x to match fx scaling.
        P_proc = P_full.copy()
        P_proc[0, 0] *= self.scale_x   # fx
        P_proc[0, 2] *= self.scale_x   # cx
        P_proc[0, 3] *= self.scale_x   # Tx (scales with fx)
        P_proc[1, 1] *= self.scale_y   # fy
        P_proc[1, 2] *= self.scale_y   # cy
        self._p_proc = [float(x) for x in P_proc.flatten()]
        # D and R don't scale with image size. Distortion model unchanged.

        # Frame list
        self.frames = list_frames(self.image_dir)
        if not self.frames:
            sys.exit(f"No camera_image_left_N.png files in {self.image_dir}")
        self.get_logger().info(
            f"Found {len(self.frames)} frames "
            f"(first={self.frames[0][0]}, last={self.frames[-1][0]})"
        )

        # Mask (binarised, mono8). Resized to (mask_width, mask_height)
        # which mimics real-world deployment: segmentation model runs on the
        # downscaled image, not at full resolution.
        mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            sys.exit(f"Failed to read mask: {self.mask_path}")
        mask_bin = (mask > 127).astype(np.uint8) * 255
        mask_target = (int(args.mask_width), int(args.mask_height))
        if mask_target != mask_bin.shape[1::-1]:
            mask_bin = cv2.resize(mask_bin, mask_target,
                                  interpolation=cv2.INTER_NEAREST)
        self.mask_arr = mask_bin
        self.get_logger().info(
            f"Loaded mask -> {self.mask_arr.shape}, "
            f"foreground pixels: {(self.mask_arr > 0).sum()}"
        )

        # Publishers — RELIABLE + VOLATILE matches Isaac ROS selector
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # Left publishers: proc-res always; full-res only if --publish_full
        self.publish_full = bool(args.publish_full)
        self.pub_rgb_rect   = self.create_publisher(
            Image,      args.rgb_topic_rect,   sensor_qos)
        self.pub_cinfo_rect = self.create_publisher(
            CameraInfo, args.cinfo_topic_rect, sensor_qos)
        # ESS subscribes to bare /left/camera_info; FS subscribes to
        # /left/camera_info_rect. Publish on both so the same publisher
        # works with either backend with no flag changes.
        self.pub_cinfo_alias = self.create_publisher(
            CameraInfo, args.cinfo_topic_alias, sensor_qos)
        if self.publish_full:
            self.pub_rgb_full   = self.create_publisher(
                Image,      args.rgb_topic_full,   sensor_qos)
            self.pub_cinfo_full = self.create_publisher(
                CameraInfo, args.cinfo_topic_full, sensor_qos)
            self.get_logger().info(
                f"Full-res publishing ON: {args.rgb_topic_full} + "
                f"{args.cinfo_topic_full}"
            )
        else:
            self.pub_rgb_full   = None
            self.pub_cinfo_full = None
            self.get_logger().info(
                "Full-res publishing OFF (recorder won't have full-res "
                "image for overlay). Pass --publish_full to enable."
            )
        if self.do_publish_depth:
            self.pub_depth = self.create_publisher(Image,  args.depth_topic, sensor_qos)
        else:
            self.pub_depth = None
            self.get_logger().info(
                "No --depth_dir provided; skipping depth publishing. "
                "Make sure another node (e.g. stereo_depth_saver) is "
                "publishing on /left/depth_image."
            )
        self.pub_mask  = self.create_publisher(Image,      args.mask_topic,  sensor_qos)

        # ── Optional right camera publishers ────────────────────────────
        self.do_right = bool(args.right_image_dir)
        if self.do_right:
            if not args.right_camera_yaml:
                sys.exit("--right_image_dir requires --right_camera_yaml")
            self.right_image_dir = Path(args.right_image_dir)
            if not self.right_image_dir.is_dir():
                sys.exit(f"right_image_dir not found: {self.right_image_dir}")
            self.right_frames = list_frames(self.right_image_dir, side="right")
            if not self.right_frames:
                sys.exit(f"No camera_image_right_N.png in "
                         f"{self.right_image_dir}")
            if len(self.right_frames) != len(self.frames):
                self.get_logger().warn(
                    f"Right has {len(self.right_frames)} frames, left has "
                    f"{len(self.frames)}. Will publish min of the two."
                )
            # Load right camera info and pre-compute proc-res versions
            self.right_cinfo = load_camera_info_from_yaml(
                Path(args.right_camera_yaml), args.right_frame_id
            )
            # Right K full from YAML
            R_K_full = np.array(self.right_cinfo.k,
                                dtype=np.float64).reshape(3, 3)
            R_P_full = np.array(self.right_cinfo.p,
                                dtype=np.float64).reshape(3, 4)
            # Right K_proc with same x/y scales (assumes right cam has
            # same full-res W/H as left, which is true for ZED stereo)
            R_K_proc = R_K_full.copy()
            R_K_proc[0, 0] *= self.scale_x
            R_K_proc[0, 2] *= self.scale_x
            R_K_proc[1, 1] *= self.scale_y
            R_K_proc[1, 2] *= self.scale_y
            self._right_k_proc = [float(x) for x in R_K_proc.flatten()]
            R_P_proc = R_P_full.copy()
            R_P_proc[0, 0] *= self.scale_x
            R_P_proc[0, 2] *= self.scale_x
            R_P_proc[0, 3] *= self.scale_x
            R_P_proc[1, 1] *= self.scale_y
            R_P_proc[1, 2] *= self.scale_y
            self._right_p_proc = [float(x) for x in R_P_proc.flatten()]

            # Publishers: right always gets _rect; _full conditional on
            # the same --publish_full flag as left.
            self.pub_right_rgb_rect   = self.create_publisher(
                Image,      args.right_rgb_topic_rect,   sensor_qos)
            self.pub_right_cinfo_rect = self.create_publisher(
                CameraInfo, args.right_cinfo_topic_rect, sensor_qos)
            # ESS-compat alias for right (same as left)
            self.pub_right_cinfo_alias = self.create_publisher(
                CameraInfo, args.right_cinfo_topic_alias, sensor_qos)
            if self.publish_full:
                self.pub_right_rgb_full   = self.create_publisher(
                    Image,      args.right_rgb_topic_full,   sensor_qos)
                self.pub_right_cinfo_full = self.create_publisher(
                    CameraInfo, args.right_cinfo_topic_full, sensor_qos)
            else:
                self.pub_right_rgb_full   = None
                self.pub_right_cinfo_full = None
            self.get_logger().info(
                f"Right camera ON: {len(self.right_frames)} frames, "
                f"{self.right_cinfo.width}x{self.right_cinfo.height}, "
                f"fx_full={self.right_cinfo.k[0]:.2f}; "
                f"publishing on {args.right_rgb_topic_rect} "
                f"({self.proc_W}x{self.proc_H})"
                + (f" + {args.right_rgb_topic_full} (full)"
                   if self.publish_full else "")
            )
        else:
            self.get_logger().info("Right camera OFF "
                                   "(pass --right_image_dir to enable)")

        # Worker thread runs the publish loop; main thread spins for ROS callbacks
        self._worker = threading.Thread(target=self._publish_loop, daemon=True)
        self._worker.start()

    # ──────────────────────────────────────────────────────────────────
    def _count_pose_files(self) -> int:
        return sum(1 for _ in self.recorder_link0_dir.glob("*.npy"))

    def _wait_for_recorder(self, prev_count: int) -> None:
        """Block until a new pose file appears in the recorder dir."""
        while rclpy.ok():
            if self._count_pose_files() > prev_count:
                return
            time.sleep(0.02)

    # ──────────────────────────────────────────────────────────────────
    def _publish_loop(self):
        # Wipe stale .npy files from previous runs FIRST, before any sleep
        # or publishing. Cleans both left (pose_cam_to_link0) and right
        # (pose_right_to_link0) pose dirs if they exist. Doesn't touch
        # overlays/ — the recorder manages that.
        recorder_root = self.recorder_link0_dir.parent
        for d in (self.recorder_link0_dir,
                  recorder_root / "pose_right_to_link0"):
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file() and f.suffix == ".npy":
                        f.unlink()
        ts = recorder_root / "timestamps.csv"
        if ts.is_file():
            ts.unlink()

        # Now give subscribers (including downstream tools) time to register.
        time.sleep(2.0)

        successful = 0
        prev_count = 0

        for i, (fnum, img_path) in enumerate(self.frames):
            if not rclpy.ok():
                return

            depth = None
            if self.do_publish_depth:
                depth_path = self.depth_dir / f"{fnum:06d}.npy"
                if not depth_path.exists():
                    self.get_logger().warn(f"Skip frame {fnum}: missing {depth_path}")
                    continue

            # ── Stage 1: Read all source data from disk in parallel ──────
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                self.get_logger().warn(f"Skip frame {fnum}: failed to read image")
                continue
            if self.do_publish_depth:
                depth = np.load(depth_path).astype(np.float32)
                if depth.shape[:2] != img_bgr.shape[:2]:
                    self.get_logger().warn(
                        f"Skip frame {fnum}: depth {depth.shape} != image "
                        f"{img_bgr.shape[:2]}"
                    )
                    continue
            # Read right image too if enabled (so we have everything in
            # memory before any publish happens)
            rimg = None
            if self.do_right and i < len(self.right_frames):
                rfnum, rpath = self.right_frames[i]
                rimg = cv2.imread(str(rpath), cv2.IMREAD_COLOR)
                if rimg is None:
                    self.get_logger().warn(
                        f"Right frame {rfnum}: failed to read image"
                    )

            # ── Stage 2: Build messages for BOTH full-res and proc-res ────
            # img_bgr stays full-res. img_proc is the downscaled copy that
            # FS and FP consume.
            img_proc = cv2.resize(img_bgr, (self.proc_W, self.proc_H),
                                  interpolation=cv2.INTER_LINEAR)
            depth_proc = None
            if depth is not None:
                depth_proc = cv2.resize(depth, (self.proc_W, self.proc_H),
                                        interpolation=cv2.INTER_NEAREST)

            rimg_proc = None
            if rimg is not None:
                rimg_proc = cv2.resize(rimg, (self.proc_W, self.proc_H),
                                       interpolation=cv2.INTER_LINEAR)

            now = self.get_clock().now().to_msg()
            hdr = Header(); hdr.stamp = now; hdr.frame_id = self.args.frame_id

            # Left full-res messages (only if publishing full-res)
            left_rgb_full_msg = None
            left_ci_full = None
            if self.publish_full:
                left_rgb_full_msg = numpy_to_image_msg(img_bgr, "bgr8", hdr)
                left_ci_full = CameraInfo()
                left_ci_full.header = hdr
                left_ci_full.height = self.full_H
                left_ci_full.width  = self.full_W
                left_ci_full.k = self._k_full
                left_ci_full.p = self._p_full
                left_ci_full.d = self._d_full
                left_ci_full.r = self._r_full
                left_ci_full.distortion_model = self._dist_model

            # Left proc-res messages
            left_rgb_rect_msg = numpy_to_image_msg(img_proc, "bgr8", hdr)
            left_depth_msg = (numpy_to_image_msg(depth_proc, "32FC1", hdr)
                              if depth_proc is not None else None)
            left_mask_msg  = numpy_to_image_msg(self.mask_arr, "mono8", hdr)
            left_ci_rect = CameraInfo()
            left_ci_rect.header = hdr
            left_ci_rect.height = self.proc_H
            left_ci_rect.width  = self.proc_W
            left_ci_rect.k = self._k_proc
            left_ci_rect.p = self._p_proc
            left_ci_rect.d = self._d_full   # D doesn't scale with image size
            left_ci_rect.r = self._r_full
            left_ci_rect.distortion_model = self._dist_model

            # Right messages (full + rect) if right is enabled
            right_rgb_full_msg = right_ci_full = None
            right_rgb_rect_msg = right_ci_rect = None
            if rimg is not None:
                rhdr = Header()
                rhdr.stamp = now
                rhdr.frame_id = self.args.right_frame_id
                # Full-res (conditional)
                if self.publish_full:
                    right_rgb_full_msg = numpy_to_image_msg(rimg, "bgr8", rhdr)
                    right_ci_full = CameraInfo()
                    right_ci_full.header = rhdr
                    right_ci_full.height = self.right_cinfo.height
                    right_ci_full.width  = self.right_cinfo.width
                    right_ci_full.k = self.right_cinfo.k
                    right_ci_full.p = self.right_cinfo.p
                    right_ci_full.d = self.right_cinfo.d
                    right_ci_full.r = self.right_cinfo.r
                    right_ci_full.distortion_model = self.right_cinfo.distortion_model
                # Proc-res
                right_rgb_rect_msg = numpy_to_image_msg(rimg_proc, "bgr8", rhdr)
                right_ci_rect = CameraInfo()
                right_ci_rect.header = rhdr
                right_ci_rect.height = self.proc_H
                right_ci_rect.width  = self.proc_W
                right_ci_rect.k = self._right_k_proc
                right_ci_rect.p = self._right_p_proc
                right_ci_rect.d = list(self.right_cinfo.d)
                right_ci_rect.r = list(self.right_cinfo.r)
                right_ci_rect.distortion_model = self.right_cinfo.distortion_model

            # ── Stage 3: Publish everything back-to-back, no work between ─
            # Left
            if self.publish_full:
                self.pub_rgb_full.publish(left_rgb_full_msg)
                self.pub_cinfo_full.publish(left_ci_full)
            self.pub_rgb_rect.publish(left_rgb_rect_msg)
            self.pub_cinfo_rect.publish(left_ci_rect)
            self.pub_cinfo_alias.publish(left_ci_rect)
            if left_depth_msg is not None:
                self.pub_depth.publish(left_depth_msg)
            self.pub_mask.publish(left_mask_msg)
            # Right
            if right_rgb_rect_msg is not None:
                if self.publish_full and right_rgb_full_msg is not None:
                    self.pub_right_rgb_full.publish(right_rgb_full_msg)
                    self.pub_right_cinfo_full.publish(right_ci_full)
                self.pub_right_rgb_rect.publish(right_rgb_rect_msg)
                self.pub_right_cinfo_rect.publish(right_ci_rect)
                self.pub_right_cinfo_alias.publish(right_ci_rect)

            # Wait until the recorder writes the matching .npy
            self._wait_for_recorder(prev_count)
            prev_count += 1
            successful += 1
            if i == 0 or i % 25 == 0 or i == len(self.frames) - 1:
                self.get_logger().info(
                    f"Frame {fnum} ({i+1}/{len(self.frames)}): OK"
                )

        self.get_logger().info(
            f"DONE. Published {len(self.frames)} frames, "
            f"recorded {successful}."
        )
        time.sleep(2.0)
        rclpy.shutdown()


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--depth_dir", default="",
                   help="Optional. Path to pre-computed depth .npy files "
                        "(NVlabs FoundationStereo format). When set, "
                        "publishes /left/depth_image from disk. When empty, "
                        "skips depth publishing entirely (useful when depth "
                        "is generated live by an Isaac ROS stereo node).")
    p.add_argument("--camera_yaml", required=True)
    p.add_argument("--mask_path", required=True)
    p.add_argument("--recorder_out_dir", required=True,
                   help="Same --out_dir you pass to fp_pose_recorder.py")
    p.add_argument("--frame_id", default="zed_left_camera_optical_frame")
    p.add_argument("--rgb_topic_full",   default="/left/image_full",
                   help="Full-resolution left image topic (for recorder "
                        "overlay drawing).")
    p.add_argument("--cinfo_topic_full", default="/left/camera_info_full",
                   help="Full-resolution left camera_info topic.")
    p.add_argument("--rgb_topic_rect",   default="/left/image_rect",
                   help="Processing-resolution (proc_W x proc_H) left image "
                        "topic. Consumed by FoundationStereo and FoundationPose.")
    p.add_argument("--cinfo_topic_rect", default="/left/camera_info_rect",
                   help="Processing-resolution left camera_info topic.")
    p.add_argument("--cinfo_topic_alias", default="/left/camera_info",
                   help="ESS-compat alias for left camera_info. ESS subscribes "
                        "to bare /left/camera_info; FS uses _rect suffix. "
                        "We publish on both for backend-agnostic compatibility.")
    p.add_argument("--depth_topic", default="/left/depth_image_full",
                   help="Topic for depth published from --depth_dir (only "
                        "used when --depth_dir is set; for live-FS pipelines "
                        "the stereo_depth_saver publishes /left/depth_image).")
    p.add_argument("--mask_topic",  default="/left/segmentation",
                   help="Segmentation mask topic. Mask is pre-resized to "
                        "(--mask_width, --mask_height) which should match "
                        "(--proc_width, --proc_height) for FP.")
    # Right (optional)
    p.add_argument("--right_image_dir", default="",
                   help="If set, also publish right images.")
    p.add_argument("--right_camera_yaml", default="",
                   help="Right camera_info yaml. Required with "
                        "--right_image_dir.")
    p.add_argument("--right_frame_id", default="zed_right_camera_optical_frame")
    p.add_argument("--right_rgb_topic_full",   default="/right/image_full")
    p.add_argument("--right_cinfo_topic_full", default="/right/camera_info_full")
    p.add_argument("--right_rgb_topic_rect",   default="/right/image_rect")
    p.add_argument("--right_cinfo_topic_rect", default="/right/camera_info_rect")
    p.add_argument("--right_cinfo_topic_alias", default="/right/camera_info",
                   help="ESS-compat alias for right camera_info (see "
                        "--cinfo_topic_alias).")
    p.add_argument("--proc_width", type=int, default=960,
                   help="Width of the processing-resolution stream sent to "
                        "FoundationStereo and FoundationPose. Default 960 "
                        "(FoundationStereo native width).")
    p.add_argument("--proc_height", type=int, default=576,
                   help="Height of the processing-resolution stream. "
                        "Default 576 (FoundationStereo native height).")
    p.add_argument("--publish_full", type=int, default=1,
                   help="Publish full-resolution images on _full topics for "
                        "the recorder's overlay drawing. Default 1 (on). "
                        "Pass 0 if you don't need overlay (saves serialization "
                        "and DDS bandwidth).")
    p.add_argument("--mask_width", type=int, default=960,
                   help="Width to resize the segmentation mask to. Should "
                        "match --proc_width. Default 960.")
    p.add_argument("--mask_height", type=int, default=576,
                   help="Height to resize the segmentation mask to. Should "
                        "match --proc_height. Default 576.")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = OneAtATimePublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()