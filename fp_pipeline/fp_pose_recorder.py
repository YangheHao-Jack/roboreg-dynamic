#!/usr/bin/env python3
"""
fp_pose_recorder.py

Subscribes to FoundationPose pose output. Two modes:

  1. Counter mode (default): count poses, log rate every second.
     Lightweight, no disk I/O. Used to measure FP throughput.

  2. Save mode (--save_npy): write per-frame T_cam_to_link0 .npy +
     timestamps.csv. Optional stereo mirror to right camera and
     wireframe+axes overlay PNGs.

Topics consumed
    /pose_estimation/pose_matrix_output  (TensorList, 4x4 col-major fp32; init)
    /tracking/pose_matrix_output         (TensorList, 4x4 col-major fp32; track)
    /pose_estimation/output, /tracking/output  (Detection3DArray, debug only)
    --stamp_source_topic                 (CameraInfo, for bag-frame stamps)
    + image/cinfo topics if --save_overlay

Outputs (only with --save_npy)
    out_dir/pose_cam_to_link0/000NNN.npy      (always with --save_npy)
    out_dir/timestamps.csv                    (always with --save_npy)
    out_dir/pose_right_to_link0/000NNN.npy    (with --stereo_npy)
    out_dir/overlays/overlay_000NNN.png       (with --save_overlay)
    out_dir/overlays_right/overlay_000NNN.png (with --save_overlay --enable_right)

Usage (counter mode)
    python fp_pose_recorder.py \\
        --offset_npy ~/FoundationPose_assets/lbr_med7_baked_offset.npy

Usage (save mode)
    python fp_pose_recorder.py \\
        --save_npy \\
        --offset_npy ~/FoundationPose_assets/lbr_med7_baked_offset.npy \\
        --out_dir    ~/FoundationPose_assets/poses_disp01/

Usage (save mode + stereo + overlay)
    python fp_pose_recorder.py \\
        --save_npy \\
        --offset_npy ~/FoundationPose_assets/lbr_med7_baked_offset.npy \\
        --out_dir    ~/FoundationPose_assets/poses_disp01/ \\
        --stereo_npy /home/jack/.../HT_right_to_left.npy \\
        --save_overlay --enable_right \\
        --mesh_obj   ~/FoundationPose_assets/lbr_med7_baked.obj
"""

import argparse
import csv
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSDurabilityPolicy, QoSHistoryPolicy)

try:
    from isaac_ros_tensor_list_interfaces.msg import TensorList
except ImportError:
    sys.exit("isaac_ros_tensor_list_interfaces not found. "
             "Source ROS2 + run 'isaac-ros activate'.")

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped


SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10)
STAMP_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=50)


# ── Helpers ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--offset_npy", required=True,
                   help="Path to lbr_med7_baked_offset.npy from the bake script.")
    p.add_argument("--save_npy", action="store_true",
                   help="Save per-frame T_cam_to_link0 .npy + timestamps.csv "
                        "to --out_dir. Off by default (counter-only mode).")
    p.add_argument("--out_dir", default="",
                   help="Where to save per-frame pose .npy files. "
                        "Required when --save_npy is set.")
    p.add_argument("--matrix_topic", default="/pose_estimation/pose_matrix_output")
    p.add_argument("--detection_topic", default="/output")
    p.add_argument("--save_overlay", action="store_true",
                   help="Generate overlay PNG each frame. Requires --save_npy.")
    p.add_argument("--stereo_npy", default="",
                   help="Path to HT_right_to_left.npy. If set with --save_npy, "
                        "also save mirrored T_right_to_link0 poses.")
    p.add_argument("--left_image_topic",  default="/left/image_rect")
    p.add_argument("--left_cinfo_topic",  default="/left/camera_info_rect")
    p.add_argument("--right_image_topic", default="/right/image_rect")
    p.add_argument("--right_cinfo_topic", default="/right/camera_info_rect")
    p.add_argument("--use_bag_frame_index", type=int, default=0,
                   help="If 1: name pose .npy files by the bag's frame index "
                        "(counted from --stamp_source_topic), so dropped "
                        "frames create visible gaps. If 0 (default): "
                        "sequential (000000.npy, 000001.npy, ...).")
    p.add_argument("--stamp_source_topic", default="/left/camera_info_rect",
                   help="High-rate CameraInfo topic to track per-pose bag stamps.")
    p.add_argument("--enable_right", action="store_true",
                   help="Subscribe to right camera + produce right overlays. "
                        "Requires --stereo_npy.")
    p.add_argument("--mesh_obj", default="",
                   help="Path to centered .obj. Required for overlays.")
    p.add_argument("--axis_length", type=float, default=0.1)
    p.add_argument("--mesh_subsample", type=int, default=2)
    p.add_argument("--init_only", action="store_true",
                   help="One-shot init mode: wait for first FP pose, "
                        "latched-publish /pose_init (PoseStamped), kill "
                        "FP+ESS+depth_saver via PID files in --pid_dir, "
                        "then keep spinning. Other modes (--save_npy, "
                        "--save_overlay) are ignored.")
    p.add_argument("--pid_dir", default="/tmp/fp_init_pids",
                   help="Directory holding {fp,ess,depth_saver}.pid files "
                        "to terminate after init. Used only with --init_only.")
    p.add_argument("--pose_init_topic", default="/pose_init",
                   help="Topic for latched PoseStamped publish (init mode).")
    p.add_argument("--pose_init_frame_id", default="zed_left_camera_optical_frame")
    return p.parse_args()


def tensorlist_to_4x4(msg: TensorList):
    """First 4x4 from a TensorList. FP publishes column-major fp32."""
    if not msg.tensors:
        return None
    arr = np.frombuffer(bytes(msg.tensors[0].data), dtype=np.float32)
    if arr.size < 16:
        return None
    return arr[:16].reshape(4, 4, order="F").astype(np.float64)


def project_points(P3, K):
    depths = P3[2]
    valid = depths > 1e-6
    uv_h = K @ P3
    uv = np.full((2, P3.shape[1]), -1.0)
    uv[:, valid] = uv_h[:2, valid] / uv_h[2:3, valid]
    return uv, depths


def stamp_u64(stamp):
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)


def img_msg_to_bgr(msg):
    """ROS Image -> BGR ndarray. Returns None for unsupported encodings."""
    import cv2
    h, w = msg.height, msg.width
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "bgr8":
        return raw.reshape(h, w, 3)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgra8":
        return raw.reshape(h, w, 4)[:, :, :3]
    if msg.encoding == "rgba8":
        return cv2.cvtColor(raw.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    return None


def draw_axes(img, T, K, length):
    import cv2
    pts = np.array([[0, 0, 0, 1],
                    [length, 0, 0, 1],
                    [0, length, 0, 1],
                    [0, 0, length, 1]]).T
    P_cam = (T @ pts)[:3]
    uv, depths = project_points(P_cam, K)
    if depths[0] <= 0:
        return
    H, W = img.shape[:2]
    o = uv[:, 0].astype(int)
    if not (0 <= o[0] < W and 0 <= o[1] < H):
        return
    for k, color in [(1, (0, 0, 255)),    # X red
                     (2, (0, 255, 0)),    # Y green
                     (3, (255, 0, 0))]:   # Z blue
        if depths[k] <= 0:
            continue
        p = uv[:, k].astype(int)
        cv2.line(img, tuple(o), tuple(p), color, 3, cv2.LINE_AA)


def draw_mesh_wireframe(img, verts, edges, T, K, color=(0, 220, 0)):
    import cv2
    Vh = np.hstack([verts, np.ones((len(verts), 1))]).T
    Vc = (T @ Vh)[:3]
    uv, depths = project_points(Vc, K)
    H, W = img.shape[:2]
    for a, b in edges:
        if depths[a] <= 0 or depths[b] <= 0:
            continue
        pa = uv[:, a]; pb = uv[:, b]
        if (pa[0] < 0 or pa[0] >= W or pa[1] < 0 or pa[1] >= H or
                pb[0] < 0 or pb[0] >= W or pb[1] < 0 or pb[1] >= H):
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])),
                 (int(pb[0]), int(pb[1])), color, 1, cv2.LINE_AA)


def matrix_to_pose_stamped(T, frame_id, stamp):
    """4x4 matrix -> PoseStamped (quaternion via Shepperd's method)."""
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    R = T[:3, :3]
    t = T[:3, 3]
    msg.pose.position.x = float(t[0])
    msg.pose.position.y = float(t[1])
    msg.pose.position.z = float(t[2])
    # Stable rotation -> quaternion (Shepperd 1978)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = (tr + 1.0) ** 0.5 * 2.0  # = 4*qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0  # = 4*qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0  # = 4*qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0  # = 4*qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def kill_pid_file(pid_path: Path, label: str, logger):
    """Read pid_path, send SIGTERM (then SIGKILL after 2s if still alive)."""
    import os
    import signal
    import time as _time
    if not pid_path.exists():
        logger.warn(f"PID file missing: {pid_path} ({label} not killed)")
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except Exception as e:
        logger.warn(f"Bad PID in {pid_path}: {e}")
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info(f"Sent SIGTERM to {label} (pid {pid})")
    except ProcessLookupError:
        logger.info(f"{label} (pid {pid}) already exited")
        return True
    except PermissionError:
        logger.error(f"Cannot kill {label} (pid {pid}): permission denied")
        return False
    # Wait briefly, then SIGKILL if still running
    for _ in range(20):
        _time.sleep(0.1)
        try:
            os.kill(pid, 0)  # check
        except ProcessLookupError:
            logger.info(f"{label} terminated cleanly")
            return True
    try:
        os.kill(pid, signal.SIGKILL)
        logger.warn(f"Force-killed {label} (pid {pid})")
    except ProcessLookupError:
        pass
    return True


def pkill_pattern(pattern: str, label: str, logger):
    """SIGTERM all processes whose cmdline matches `pattern` (substring)."""
    import os
    import signal
    import time as _time
    # Find PIDs by reading /proc — avoids depending on `pkill` binary
    matches = []
    try:
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            cmdline_path = f"/proc/{pid_str}/cmdline"
            try:
                with open(cmdline_path, "rb") as f:
                    cmdline = f.read().replace(b"\x00", b" ").decode(
                        "utf-8", errors="replace")
            except (FileNotFoundError, PermissionError):
                continue
            if pattern in cmdline and int(pid_str) != os.getpid():
                matches.append(int(pid_str))
    except Exception as e:
        logger.warn(f"pkill scan failed: {e}")
        return False
    if not matches:
        logger.warn(f"No process matched '{pattern}' for {label}")
        return False
    for pid in matches:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to {label} (pid {pid}) [match '{pattern}']")
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.error(f"Cannot kill pid {pid}: permission denied")
    # Brief grace period, then SIGKILL stragglers
    for _ in range(20):
        _time.sleep(0.1)
        alive = []
        for pid in matches:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            logger.info(f"{label}: all matched processes terminated")
            return True
        matches = alive
    for pid in matches:
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warn(f"Force-killed {label} (pid {pid})")
        except ProcessLookupError:
            pass
    return True


def wipe_dir(path: Path, ext: str) -> int:
    n = 0
    if path.exists():
        for f in path.iterdir():
            if f.is_file() and f.suffix == ext:
                f.unlink()
                n += 1
    return n


# ── Node ─────────────────────────────────────────────────────────────
class PoseRecorder(Node):

    def __init__(self, args):
        super().__init__("fp_pose_recorder")
        self.args = args
        self.init_only = bool(args.init_only)
        self.save_npy = bool(args.save_npy)
        self._init_done = False

        # init_only: no per-frame npy; default out_dir for FP_init.npy + overlays.
        if self.init_only:
            self.save_npy = False
            if not args.out_dir:
                args.out_dir = str(Path.home() / "fp_init_debug")
                self.get_logger().info(
                    f"--init_only: defaulting --out_dir to {args.out_dir}")

        if self.save_npy and not args.out_dir:
            sys.exit("--save_npy requires --out_dir")
        if args.save_overlay and not (self.save_npy or self.init_only):
            sys.exit("--save_overlay requires --save_npy or --init_only")
        if args.save_overlay and not args.out_dir:
            sys.exit("--save_overlay requires --out_dir")
        if args.stereo_npy and not (self.save_npy or self.init_only):
            sys.exit("--stereo_npy requires --save_npy or --init_only")
        if args.enable_right and not args.stereo_npy:
            sys.exit("--enable_right requires --stereo_npy")

        # ── Offset (mesh-bake → link0) — ALWAYS required ────────────
        # link0 sits at -bbox_center in the centered (baked) mesh frame.
        offset = np.load(args.offset_npy).astype(np.float64).flatten()
        if offset.shape != (3,):
            sys.exit(f"offset_npy expected shape (3,), got {offset.shape}")
        self.offset = offset
        self.T_centered_to_link0 = np.eye(4)
        self.T_centered_to_link0[:3, 3] = -offset
        self.T_link0_to_centered = np.eye(4)
        self.T_link0_to_centered[:3, 3] = +offset
        self.get_logger().info(
            f"Loaded offset {offset} (||={np.linalg.norm(offset):.4f} m)")

        # ── Output dir + CSV (only in save mode) ────────────────────
        self.out_dir = None
        self.ts_csv = None
        self.ts_writer = None
        # out_dir is created if any save mode wants it
        if args.out_dir:
            self.out_dir = Path(args.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.save_npy:
            pose_dir = self.out_dir / "pose_cam_to_link0"
            pose_dir.mkdir(parents=True, exist_ok=True)
            wiped = wipe_dir(pose_dir, ".npy")
            if wiped:
                self.get_logger().info(f"Wiped {wiped} stale .npy files")
            self.ts_csv = open(self.out_dir / "timestamps.csv", "w", newline="")
            self.ts_writer = csv.writer(self.ts_csv)
            self.ts_writer.writerow(["frame_index", "stamp_sec", "stamp_nsec"])
            self.get_logger().info(f"Save mode: writing to {self.out_dir}")
        elif self.init_only and args.save_overlay:
            self.get_logger().info(
                f"init_only debug overlays will be saved under {self.out_dir}")
        elif not self.init_only:
            self.get_logger().info(
                "Counter-only mode (no .npy save). Pass --save_npy to enable.")

        # ── Optional stereo mirror ──────────────────────────────────
        self.do_mirror = bool(args.stereo_npy) and (self.save_npy or self.init_only)
        if self.do_mirror:
            self._init_stereo(args)
        elif self.save_npy or self.init_only:
            self.get_logger().info("Stereo mirror OFF "
                                   "(pass --stereo_npy to enable)")

        # ── Optional overlay ────────────────────────────────────────
        self.do_overlay = bool(args.save_overlay) and (self.save_npy or self.init_only)
        self.do_right_overlay = False  # set inside _init_overlay if asked
        if self.do_overlay:
            if not args.mesh_obj:
                sys.exit("--save_overlay requires --mesh_obj")
            self._init_overlay(args)
        elif self.save_npy:
            self.get_logger().info("Overlay OFF "
                                   "(pass --save_overlay to enable)")

        # ── Stamp tracker ───────────────────────────────────────────
        # FP's output stamp is unreliable (always the first input frame's stamp);
        # we use the latest CameraInfo stamp as the real bag time per pose,
        # and increment a per-frame counter on each CameraInfo tick.
        self._latest_image_stamp = None
        self._bag_frame_counter = -1     # not yet seen
        self._last_saved_bag_frame = -1  # for dedup in _on_matrix
        self.create_subscription(
            CameraInfo, args.stamp_source_topic,
            self._on_stamp_source, STAMP_QOS)
        self.use_bag_frame_index = bool(args.use_bag_frame_index)
        self.get_logger().info(
            f"Stamp source: {args.stamp_source_topic} "
            f"(use_bag_frame_index={self.use_bag_frame_index})")

        self.frame_idx = 0

        # Rate counter (logged every second)
        self._frames_total = 0
        self._frames_last_window = 0
        import time as _time
        self._t0 = _time.monotonic()
        self._t_last = self._t0
        self.create_timer(1.0, self._log_rate)

        # ── init_only: latched pose publisher + PID setup ───────────
        self._pose_init_pub = None
        if self.init_only:
            latched_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1)
            self._pose_init_pub = self.create_publisher(
                PoseStamped, args.pose_init_topic, latched_qos)
            self._pid_dir = Path(args.pid_dir)
            self.get_logger().info(
                f"INIT-ONLY MODE: waiting for first pose, will latched-publish "
                f"on {args.pose_init_topic}, then kill PIDs from {self._pid_dir}")

        # Frame 0 from /pose_estimation; frames 1..N from /tracking.
        # init_only only needs the estimator (frame 0).
        self.create_subscription(
            TensorList, args.matrix_topic, self._on_matrix, 10)
        if not self.init_only:
            self.create_subscription(
                TensorList, "/tracking/pose_matrix_output",
                self._on_matrix, 10)
            self.create_subscription(
                Detection3DArray, args.detection_topic, self._on_detection, 10)
            self.create_subscription(
                Detection3DArray, "/tracking/output", self._on_detection, 10)
            self.get_logger().info(
                f"Listening on {args.matrix_topic} + /tracking/pose_matrix_output")
        else:
            self.get_logger().info(
                f"Listening on {args.matrix_topic} (init_only)")
        if self.save_npy:
            self.get_logger().info(f"Saving to {self.out_dir}")

    # ── Stereo init ─────────────────────────────────────────────────
    def _init_stereo(self, args):
        H = np.load(args.stereo_npy, allow_pickle=True)
        if H.ndim == 3:
            H = H[0]
        H = H.reshape(4, 4).astype(np.float64)

        # ROS2 (X-fwd, Y-left, Z-up) -> OpenCV (X-right, Y-down, Z-fwd)
        R_cl2opt = np.array([[ 0, -1,  0],
                             [ 0,  0, -1],
                             [ 1,  0,  0]], dtype=np.float64)
        H_cv = np.eye(4)
        H_cv[:3, :3] = R_cl2opt @ H[:3, :3] @ R_cl2opt.T
        H_cv[:3, 3]  = R_cl2opt @ H[:3, 3]
        self.H_left_to_right_cv = np.linalg.inv(H_cv)
        baseline = float(np.linalg.norm(self.H_left_to_right_cv[:3, 3]))

        self.right_dir = self.out_dir / "pose_right_to_link0"
        if not self.init_only:
            self.right_dir.mkdir(parents=True, exist_ok=True)
            wipe_dir(self.right_dir, ".npy")
        self.get_logger().info(
            f"Stereo mirror ON: |left->right|={baseline:.4f} m"
            + (f"; out={self.right_dir}" if not self.init_only else ""))

    # ── Overlay init ────────────────────────────────────────────────
    def _init_overlay(self, args):
        import cv2
        import trimesh
        self._cv2 = cv2

        # Mesh
        m = trimesh.load(args.mesh_obj, force="mesh")
        self.verts = np.asarray(m.vertices, dtype=np.float64)
        edges = np.asarray(m.edges_unique, dtype=np.int64)
        if args.mesh_subsample > 1:
            edges = edges[::args.mesh_subsample]
        self.edges = edges

        # Image ring buffers + lock
        self._lock = threading.Lock()
        self.left_buf  = deque(maxlen=30)   # (stamp_u64, bgr)
        self.right_buf = deque(maxlen=30)
        self.K_left = None
        self.K_right = None
        # Poses received before any image arrived — drawn later
        self._pending = deque(maxlen=20)

        # Left subs + dir
        self.create_subscription(Image, args.left_image_topic,
                                 self._on_left_image, SENSOR_QOS)
        self.create_subscription(CameraInfo, args.left_cinfo_topic,
                                 self._on_left_cinfo, SENSOR_QOS)
        self.overlay_dir = self.out_dir / "overlays"
        if not self.init_only:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)
            wipe_dir(self.overlay_dir, ".png")
        self.get_logger().info(
            f"Overlay LEFT ON: {args.left_image_topic} + "
            f"{args.left_cinfo_topic}; mesh={len(self.verts)}v "
            f"{len(self.edges)}e")

        # Right (optional)
        self.do_right_overlay = bool(args.enable_right)
        if self.do_right_overlay:
            if not self.do_mirror:
                sys.exit("--enable_right requires --stereo_npy")
            self.create_subscription(Image, args.right_image_topic,
                                     self._on_right_image, SENSOR_QOS)
            self.create_subscription(CameraInfo, args.right_cinfo_topic,
                                     self._on_right_cinfo, SENSOR_QOS)
            self.overlay_dir_right = self.out_dir / "overlays_right"
            if not self.init_only:
                self.overlay_dir_right.mkdir(parents=True, exist_ok=True)
                wipe_dir(self.overlay_dir_right, ".png")
            self.get_logger().info(
                f"Overlay RIGHT ON: {args.right_image_topic} + "
                f"{args.right_cinfo_topic}")

    # ── Image / camera_info callbacks ───────────────────────────────
    def _on_left_image(self, msg):
        bgr = img_msg_to_bgr(msg)
        if bgr is None:
            return
        with self._lock:
            self.left_buf.append((stamp_u64(msg.header.stamp), bgr))
            pending = list(self._pending)
            self._pending.clear()
            K_left = self.K_left.copy() if self.K_left is not None else None
            K_right = self.K_right.copy() if self.K_right is not None else None
            bgr_for_pending = bgr.copy()
            bgr_right_for_pending = (self.right_buf[-1][1].copy()
                                     if self.right_buf else None)
        # Render queued overlays best-effort with this newly-arrived image.
        for (idx, T_cen, T_l, T_r_cen, T_r_l) in pending:
            if K_left is not None:
                self._draw_and_save(idx, bgr_for_pending.copy(),
                                    T_cen, T_l, K_left, self.overlay_dir)
            if (self.do_right_overlay and bgr_right_for_pending is not None
                    and K_right is not None and T_r_l is not None):
                self._draw_and_save(idx, bgr_right_for_pending.copy(),
                                    T_r_cen, T_r_l, K_right,
                                    self.overlay_dir_right)

    def _on_right_image(self, msg):
        bgr = img_msg_to_bgr(msg)
        if bgr is None:
            return
        with self._lock:
            self.right_buf.append((stamp_u64(msg.header.stamp), bgr))

    def _on_left_cinfo(self, msg):
        with self._lock:
            self.K_left = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_right_cinfo(self, msg):
        with self._lock:
            self.K_right = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def _closest_image(buf, target_stamp):
        if not buf:
            return None, None
        if target_stamp is None:
            return buf[-1][1], 0
        best = min(buf, key=lambda kv: abs(kv[0] - target_stamp))
        return best[1].copy(), abs(best[0] - target_stamp)

    # ── Stamp source ────────────────────────────────────────────────
    def _on_stamp_source(self, msg):
        self._latest_image_stamp = msg.header.stamp
        self._bag_frame_counter += 1

    # ── Pose handler ────────────────────────────────────────────────
    def _on_matrix(self, msg: TensorList):
        T_cam_centered = tensorlist_to_4x4(msg)
        if T_cam_centered is None:
            self.get_logger().warn("Empty TensorList; skipping")
            return

        T_cam_link0 = T_cam_centered @ self.T_centered_to_link0

        # Always: track stats
        self._frames_total += 1
        self._frames_last_window += 1

        # ── init_only: latched-publish + kill, then idle forever ────
        if self.init_only:
            if self._init_done:
                return  # ignore subsequent poses (shouldn't arrive — FP killed)
            self._do_init_handoff(T_cam_centered, T_cam_link0)
            self._init_done = True
            return

        # Filename: bag-frame index (preserves gaps) or sequential. Both
        # FP topics can fire for the same bag frame — use the first only.
        if self.use_bag_frame_index and self._bag_frame_counter >= 0:
            if self._bag_frame_counter == self._last_saved_bag_frame:
                return
            self._last_saved_bag_frame = self._bag_frame_counter
            stem = f"{self._bag_frame_counter:06d}"
        else:
            stem = f"{self.frame_idx:06d}"

        # Counter-only mode: log and return.
        if not self.save_npy:
            if self.frame_idx % 50 == 0 or self.frame_idx < 5:
                t = T_cam_link0[:3, 3]
                self.get_logger().info(
                    f"Frame {self.frame_idx}: cam->link0 t=["
                    f"{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m")
            self.frame_idx += 1
            return

        np.save(self.out_dir / "pose_cam_to_link0" / f"{stem}.npy",
                T_cam_link0)

        # Optional: mirror to right-camera frame using stereo extrinsic
        T_right_to_link0 = None
        T_right_to_centered = None
        if self.do_mirror:
            T_right_to_link0 = self.H_left_to_right_cv @ T_cam_link0
            np.save(self.right_dir / f"{stem}.npy", T_right_to_link0)
            T_right_to_centered = self.H_left_to_right_cv @ T_cam_centered

        # Timestamp — prefer the latest tracked input image stamp
        # (FP's output stamp is unreliably stale)
        idx_for_csv = int(stem)
        stamp = getattr(msg, "header", None)
        if self._latest_image_stamp is not None:
            self.ts_writer.writerow([idx_for_csv,
                                     self._latest_image_stamp.sec,
                                     self._latest_image_stamp.nanosec])
        elif stamp is not None:
            self.ts_writer.writerow([idx_for_csv,
                                     stamp.stamp.sec, stamp.stamp.nanosec])
        else:
            now = self.get_clock().now().to_msg()
            self.ts_writer.writerow([idx_for_csv, now.sec, now.nanosec])
        self.ts_csv.flush()

        # Overlay (synchronous in same callback)
        if self.do_overlay:
            if self._latest_image_stamp is not None:
                target = (int(self._latest_image_stamp.sec) * 10**9
                          + int(self._latest_image_stamp.nanosec))
            elif stamp is not None:
                target = stamp_u64(stamp.stamp)
            else:
                target = None
            with self._lock:
                bgr_left, _ = self._closest_image(self.left_buf, target)
                K_left = (self.K_left.copy()
                          if self.K_left is not None else None)
                if self.do_right_overlay:
                    bgr_right, _ = self._closest_image(self.right_buf, target)
                    K_right = (self.K_right.copy()
                               if self.K_right is not None else None)
                else:
                    bgr_right, K_right = None, None

            if bgr_left is not None and K_left is not None:
                self._draw_and_save(int(stem), bgr_left,
                                    T_cam_centered, T_cam_link0,
                                    K_left, self.overlay_dir)
            elif K_left is not None:
                # No image yet — defer until one arrives
                with self._lock:
                    self._pending.append((
                        int(stem),
                        T_cam_centered.copy(),
                        T_cam_link0.copy(),
                        T_right_to_centered.copy()
                            if T_right_to_centered is not None else None,
                        T_right_to_link0.copy()
                            if T_right_to_link0 is not None else None,
                    ))
            if (self.do_right_overlay and bgr_right is not None
                    and K_right is not None and T_right_to_link0 is not None):
                self._draw_and_save(int(stem), bgr_right,
                                    T_right_to_centered, T_right_to_link0,
                                    K_right, self.overlay_dir_right)

        if self.frame_idx % 50 == 0 or self.frame_idx < 5:
            t = T_cam_link0[:3, 3]
            self.get_logger().info(
                f"Frame {self.frame_idx}: cam->link0 t=["
                f"{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m")
        self.frame_idx += 1

    def _draw_and_save(self, idx, bgr, T_centered, T_link0, K, overlay_dir):
        cv2 = self._cv2
        draw_mesh_wireframe(bgr, self.verts, self.edges, T_centered, K,
                            color=(0, 220, 0))
        draw_axes(bgr, T_link0, K, self.args.axis_length)
        t = T_link0[:3, 3]
        cv2.putText(bgr, f"frame {idx:04d}  "
                         f"t=[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlay_dir / f"overlay_{idx:06d}.png"), bgr)

    def _log_rate(self):
        # Suppress periodic counter logs after init_only handoff is done.
        # The 5s exit window doesn't need to be noisy.
        if self.init_only and self._init_done:
            return
        import time as _time
        now = _time.monotonic()
        dt = max(1e-6, now - self._t_last)
        rate_1s = self._frames_last_window / dt
        rate_avg = self._frames_total / max(1e-6, now - self._t0)
        self.get_logger().info(
            f"[count] N={self._frames_total}  "
            f"rate_1s={rate_1s:.1f}  rate_avg={rate_avg:.1f} Hz")
        self._t_last = now
        self._frames_last_window = 0

    def _do_init_handoff(self, T_cam_centered, T_cam_link0):
        """Render optional debug overlay, latched-publish /pose_init,
        then kill FP/ESS/depth_saver. Idempotent."""
        # Convert to IPCAI's custom frame: invert to T_link0_to_cam, then
        # apply opencv_to_ros2_transform (frame swap + row permutation).
        # See track.py for the source of this convention.
        H_left_to_base_opencv = np.linalg.inv(T_cam_link0)

        T_cv_to_ros = np.array([
            [0,  0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0,  0, 0, 1]], dtype=np.float64)
        H_ros2 = T_cv_to_ros @ H_left_to_base_opencv @ T_cv_to_ros.T
        H_temp = H_ros2.copy()
        H_temp[1, :] *= -1
        H_temp[2, :] *= -1
        # Row permute [0,1,2] -> [1,2,0]
        H_custom = H_temp.copy()
        H_custom[0, :] = H_temp[1, :]
        H_custom[1, :] = H_temp[2, :]
        H_custom[2, :] = H_temp[0, :]

        # Saved file and published topic use the same converted pose.
        npy_path = self.out_dir / "FP_init.npy"
        np.save(str(npy_path), H_custom.astype(np.float32))
        self.get_logger().info(
            f"INIT: saved H_left_to_base_custom (IPCAI convention) -> "
            f"{npy_path}")

        if self._latest_image_stamp is not None:
            stamp = self._latest_image_stamp
        else:
            stamp = self.get_clock().now().to_msg()

        pose = matrix_to_pose_stamped(
            H_custom, self.args.pose_init_frame_id, stamp)
        self._pose_init_pub.publish(pose)
        t = H_custom[:3, 3]
        self.get_logger().info(
            f"INIT: published /pose_init (IPCAI convention) "
            f"t=[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m")

        # Overlay BEFORE kill so the ring buffer still has fresh images.
        if self.do_overlay:
            self._render_init_overlay(T_cam_centered, T_cam_link0)

        # Kill upstream: depth_saver via PID file; FP/ESS via cmdline pkill.
        log = self.get_logger()
        kill_pid_file(self._pid_dir / "depth_saver.pid", "depth_saver", log)
        pkill_pattern("foundationpose_container", "FP", log)
        pkill_pattern("disparity_container",      "ESS", log)

        log.info("INIT: handoff complete. /pose_init is latched. "
                 "Recorder will exit in 5s (giving late subscribers a window).")

        # Self-shutdown after 5s so late subscribers can pick up the latch.
        def _self_shutdown():
            self.get_logger().info(
                "INIT: 5s window elapsed; shutting down recorder.")
            try:
                rclpy.shutdown()
            except Exception:
                pass
        self._shutdown_timer = self.create_timer(5.0, _self_shutdown)

    def _render_init_overlay(self, T_cam_centered, T_cam_link0):
        """One-shot overlay render for init debug. Saves PNG(s) at the
        top of out_dir."""
        log = self.get_logger()
        with self._lock:
            bgr_left = (self.left_buf[-1][1].copy()
                        if self.left_buf else None)
            K_left = (self.K_left.copy()
                      if self.K_left is not None else None)
            bgr_right = None
            K_right = None
            if self.do_right_overlay:
                bgr_right = (self.right_buf[-1][1].copy()
                             if self.right_buf else None)
                K_right = (self.K_right.copy()
                           if self.K_right is not None else None)

        if bgr_left is None:
            log.warn("init overlay: no left image cached — overlay skipped. "
                     "(images may not have started flowing yet)")
            return
        if K_left is None:
            log.warn("init overlay: no left camera_info cached — skipped")
            return

        # LEFT
        left_path = self.out_dir / "init_overlay_left.png"
        cv2 = self._cv2
        draw_mesh_wireframe(bgr_left, self.verts, self.edges,
                            T_cam_centered, K_left, color=(0, 220, 0))
        draw_axes(bgr_left, T_cam_link0, K_left, self.args.axis_length)
        t = T_cam_link0[:3, 3]
        cv2.putText(bgr_left, f"INIT  t=[{t[0]:+.3f}, "
                              f"{t[1]:+.3f}, {t[2]:+.3f}] m",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(left_path), bgr_left)
        log.info(f"init overlay LEFT  -> {left_path}")

        # RIGHT (if asked)
        if self.do_right_overlay:
            if bgr_right is None or K_right is None:
                log.warn("init overlay: right image/cinfo not cached — "
                         "right overlay skipped")
                return
            T_right_centered = self.H_left_to_right_cv @ T_cam_centered
            T_right_link0    = self.H_left_to_right_cv @ T_cam_link0
            right_path = self.out_dir / "init_overlay_right.png"
            draw_mesh_wireframe(bgr_right, self.verts, self.edges,
                                T_right_centered, K_right, color=(0, 220, 0))
            draw_axes(bgr_right, T_right_link0, K_right,
                      self.args.axis_length)
            t = T_right_link0[:3, 3]
            cv2.putText(bgr_right, f"INIT  t=[{t[0]:+.3f}, "
                                   f"{t[1]:+.3f}, {t[2]:+.3f}] m",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(right_path), bgr_right)
            log.info(f"init overlay RIGHT -> {right_path}")

    def _on_detection(self, msg: Detection3DArray):
        if not msg.detections or self.frame_idx > 1:
            return
        det = msg.detections[0]
        c = det.bbox.center.position
        self.get_logger().info(
            f"[/output] det bbox center @ ({c.x:+.3f}, {c.y:+.3f}, "
            f"{c.z:+.3f}) m, size=({det.bbox.size.x:.3f}, "
            f"{det.bbox.size.y:.3f}, {det.bbox.size.z:.3f})")

    def destroy_node(self):
        import time as _time
        elapsed = _time.monotonic() - self._t0
        rate = self._frames_total / max(1e-6, elapsed)
        self.get_logger().info(
            f"[final] N={self._frames_total}  elapsed={elapsed:.1f}s  "
            f"avg={rate:.2f} Hz")
        if hasattr(self, "ts_csv") and self.ts_csv is not None:
            self.ts_csv.close()
        super().destroy_node()


def main():
    args = parse_args()
    rclpy.init()
    node = PoseRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()