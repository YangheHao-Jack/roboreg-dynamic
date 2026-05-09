#!/usr/bin/env python3
"""
qpf2_receiver.py

TCP server for the QPF2 wire format. Decodes incoming H.264 NAL units
via PyAV (NVDEC), GPU-resizes to 960x576 via torch, and publishes per
frame, all stamped identically:

    /left/image_rect            (sensor_msgs/Image, rgb8)
    /left/camera_info_rect      (sensor_msgs/CameraInfo, FP-side)
    /left/camera_info           (sensor_msgs/CameraInfo, ESS-side, same content)
    /left/segmentation          (sensor_msgs/Image, mono8)
    + right counterparts (no segmentation)

Wire format (24-byte header, big-endian):
     4  magic "QPF2"
     1  cameraId    (50 = left, 51 = right)
     1  flags       bit0=keyframe, bit1=has_csd_prepended
     2  reserved
     8  timestamp_us
     2  width
     2  height
     4  payloadLen
   Payload: H.264 Annex-B NAL units (with optional SPS/PPS prepended).

Output resolution (960x576) matches the ESS model output so that
FoundationPose's depth_to_pointcloud sees image, camera_info, and
depth at the same dimension. Camera_info K is rescaled accordingly.

Usage:
    # Either provide YAML files...
    python qpf2_receiver.py \\
        --left_yaml  camera.left.yaml \\
        --right_yaml camera.right.yaml \\
        [--mask_path mask.png] \\
        [--start_signal_file /tmp/qpf2_start]

    # ...or have caminfo come from ROS topics (e.g. from `ros2 bag play`):
    python qpf2_receiver.py \\
        --left_camera_info_topic  /zed/zed_node/left/camera_info \\
        --right_camera_info_topic /zed/zed_node/right/camera_info
"""

import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

try:
    import av
except ImportError:
    sys.exit("PyAV not installed: pip install av --break-system-packages")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch not installed: pip install torch --break-system-packages")

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy,
)
from sensor_msgs.msg import Image, CameraInfo


# ── Wire-format constants ────────────────────────────────────────────
MAGIC = b"QPF2"
HEADER_SIZE = 24
FLAG_KEYFRAME = 0x01
FLAG_HAS_CSD = 0x02

# ── Output resolution (matches ESS ess.engine output) ────────────────
OUT_W = 960
OUT_H = 576


# ── Helpers ──────────────────────────────────────────────────────────
def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes; return None on disconnect."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _rescale_K_P(K_in, P_in, src_w, src_h):
    """Return (K_out, P_out) rescaled from (src_w, src_h) to (OUT_W, OUT_H)."""
    sx = OUT_W / float(src_w)
    sy = OUT_H / float(src_h)

    K = [float(x) for x in K_in]
    K[0] *= sx; K[1] *= sx; K[2] *= sx
    K[3] *= sy; K[4] *= sy; K[5] *= sy

    if P_in is not None and len(P_in) == 12:
        P = [float(x) for x in P_in]
        P[0] *= sx; P[1] *= sx; P[2] *= sx; P[3] *= sx
        P[4] *= sy; P[5] *= sy; P[6] *= sy; P[7] *= sy
    else:
        P = [K[0], K[1], K[2], 0.0,
             K[3], K[4], K[5], 0.0,
             K[6], K[7], K[8], 0.0]
    return K, P


def load_camera_info(yaml_path: str, frame_id: str) -> CameraInfo:
    """Load camera intrinsics from YAML and return a CameraInfo with
    K and P rescaled to (OUT_W, OUT_H)."""
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)

    K, P = _rescale_K_P(d["k"], d.get("p"), int(d["width"]), int(d["height"]))

    msg = CameraInfo()
    msg.width = OUT_W
    msg.height = OUT_H
    msg.distortion_model = d.get("distortion_model", "plumb_bob")
    msg.k = K
    msg.p = P
    msg.d = [float(x) for x in d.get("d", [0.0] * 5)]
    msg.r = [float(x) for x in d.get(
        "r", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])]
    msg.binning_x = int(d.get("binning_x", 0))
    msg.binning_y = int(d.get("binning_y", 0))
    msg.header.frame_id = frame_id
    return msg


def template_from_camera_info(src: CameraInfo, frame_id: str) -> CameraInfo:
    """Take an incoming CameraInfo (e.g. from a bag topic), return a
    template with K and P rescaled to (OUT_W, OUT_H)."""
    K, P = _rescale_K_P(src.k, src.p, int(src.width), int(src.height))

    msg = CameraInfo()
    msg.width = OUT_W
    msg.height = OUT_H
    msg.distortion_model = src.distortion_model if src.distortion_model else "plumb_bob"
    msg.k = K
    msg.p = P
    msg.d = [float(x) for x in src.d] if len(src.d) > 0 else [0.0] * 5
    msg.r = ([float(x) for x in src.r] if len(src.r) == 9
             else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    msg.binning_x = int(src.binning_x)
    msg.binning_y = int(src.binning_y)
    msg.header.frame_id = frame_id
    return msg


def load_mask(path: str) -> np.ndarray:
    """Load a mono PNG mask, resize to (OUT_W, OUT_H), binarise."""
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if raw.shape != (OUT_H, OUT_W):
        raw = cv2.resize(raw, (OUT_W, OUT_H), cv2.INTER_NEAREST)
    return ((raw > 127).astype(np.uint8) * 255)


# ── Decoder ──────────────────────────────────────────────────────────
class CameraDecoder:
    """One H.264 decoder per camera stream (h264_cuvid for NVDEC)."""

    def __init__(self, name: str, log):
        self.name = name
        self.log = log
        try:
            self._codec = av.CodecContext.create("h264_cuvid", "r")
            log.info(f"[{name}] decoder: h264_cuvid (NVDEC)")
        except Exception as e:
            log.warn(f"[{name}] h264_cuvid unavailable ({e}); falling back to CPU")
            self._codec = av.CodecContext.create("h264", "r")
        self.frames_in = 0
        self.frames_out = 0

    def decode(self, payload: bytes):
        """Yield decoded HxWx3 uint8 RGB arrays (zero or more per call)."""
        self.frames_in += 1
        try:
            frames = self._codec.decode(av.Packet(payload))
        except av.AVError as e:
            self.log.warn(f"[{self.name}] decode error: {e}")
            return
        for frame in frames:
            try:
                rgb = frame.to_ndarray(format="rgb24")
            except Exception as e:
                self.log.warn(f"[{self.name}] to_ndarray failed: {e}")
                continue
            self.frames_out += 1
            yield rgb


# ── Receiver node ────────────────────────────────────────────────────
class QPF2Receiver(Node):

    LEFT_ID = 50
    RIGHT_ID = 51
    LEFT_FRAME_ID = "zed_left_camera_optical_frame"
    RIGHT_FRAME_ID = "zed_right_camera_optical_frame"

    def __init__(self, args):
        super().__init__("qpf2_receiver")
        self.args = args

        # Gate file: drop frames until this exists. Start with a clean
        # slate (delete any leftover from a previous run).
        self._gate_path = Path(args.start_signal_file) if args.start_signal_file else None
        if self._gate_path and self._gate_path.exists():
            self._gate_path.unlink()
        self._gate_open = self._gate_path is None

        self._frame_id = {
            self.LEFT_ID:  self.LEFT_FRAME_ID,
            self.RIGHT_ID: self.RIGHT_FRAME_ID,
        }

        # ── Camera info templates (K rescaled to OUT_W x OUT_H) ─────
        # Either preload from YAML files, or wait for first message on a
        # ROS topic. Mixing the two is fine (e.g. left from YAML, right
        # from topic), but normally you pick one mode for both.
        self._cinfo = {self.LEFT_ID: None, self.RIGHT_ID: None}
        if args.left_yaml:
            self._cinfo[self.LEFT_ID] = load_camera_info(
                args.left_yaml, self.LEFT_FRAME_ID)
        if args.right_yaml:
            self._cinfo[self.RIGHT_ID] = load_camera_info(
                args.right_yaml, self.RIGHT_FRAME_ID)
        for cid, ci in self._cinfo.items():
            if ci is not None:
                self.get_logger().info(
                    f"cam{cid}: K rescaled to {ci.width}x{ci.height} (yaml)")

        # ── Mask (left camera only) ─────────────────────────────────
        self._mask = None
        if args.mask_path:
            self._mask = load_mask(args.mask_path)
            self.get_logger().info(
                f"Mask loaded {args.mask_path}: shape={self._mask.shape}, "
                f"foreground={(self._mask > 0).sum()} px")

        # ── Publishers ──────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self._pub_image = {
            self.LEFT_ID:  self.create_publisher(Image, "/left/image_rect",  qos),
            self.RIGHT_ID: self.create_publisher(Image, "/right/image_rect", qos),
        }
        # Publish camera_info to BOTH topics: ESS subscribes to bare,
        # FP's selector subscribes to _rect. Same content either way.
        self._pub_cinfo_rect = {
            self.LEFT_ID:  self.create_publisher(
                CameraInfo, "/left/camera_info_rect",  qos),
            self.RIGHT_ID: self.create_publisher(
                CameraInfo, "/right/camera_info_rect", qos),
        }
        self._pub_cinfo = {
            self.LEFT_ID:  self.create_publisher(
                CameraInfo, "/left/camera_info",  qos),
            self.RIGHT_ID: self.create_publisher(
                CameraInfo, "/right/camera_info", qos),
        }
        self._pub_mask = self.create_publisher(
            Image, "/left/segmentation", qos) if self._mask is not None else None

        # ── Camera_info subscriptions (only for sides without YAML) ─
        if args.left_camera_info_topic and self._cinfo[self.LEFT_ID] is None:
            self.create_subscription(
                CameraInfo, args.left_camera_info_topic,
                lambda m: self._on_cinfo(self.LEFT_ID, m), qos)
            self.get_logger().info(
                f"cam{self.LEFT_ID}: subscribing to "
                f"{args.left_camera_info_topic}")
        if args.right_camera_info_topic and self._cinfo[self.RIGHT_ID] is None:
            self.create_subscription(
                CameraInfo, args.right_camera_info_topic,
                lambda m: self._on_cinfo(self.RIGHT_ID, m), qos)
            self.get_logger().info(
                f"cam{self.RIGHT_ID}: subscribing to "
                f"{args.right_camera_info_topic}")

        if (self._cinfo[self.LEFT_ID] is None and not args.left_camera_info_topic) \
           or (self._cinfo[self.RIGHT_ID] is None and not args.right_camera_info_topic):
            sys.exit("Must provide --left_yaml/--right_yaml or "
                     "--left_camera_info_topic/--right_camera_info_topic for both sides.")

        # ── GPU device for resize ───────────────────────────────────
        if not torch.cuda.is_available():
            sys.exit("CUDA not available; cannot run GPU resize.")
        self._device = torch.device("cuda:0")

        # ── Decoders ────────────────────────────────────────────────
        self._decoders = {
            cid: CameraDecoder(f"cam{cid}", self.get_logger())
            for cid in (self.LEFT_ID, self.RIGHT_ID)
        }
        self._pub_count = {self.LEFT_ID: 0, self.RIGHT_ID: 0}
        self._last_log = time.monotonic()

        # ── Server thread ───────────────────────────────────────────
        self._stop = threading.Event()
        self._server_thread = threading.Thread(
            target=self._server_loop, name="qpf2_server", daemon=True)
        self._server_thread.start()
        self.get_logger().info(
            f"qpf2_receiver listening on {args.host}:{args.port}"
            + (f" (gated on {self._gate_path})" if self._gate_path else ""))

    def _on_cinfo(self, cam_id: int, msg: CameraInfo):
        """First CameraInfo received per camera fills the template."""
        if self._cinfo[cam_id] is not None:
            return
        if msg.width <= 0 or msg.height <= 0:
            return  # ignore garbage
        tmpl = template_from_camera_info(msg, self._frame_id[cam_id])
        self._cinfo[cam_id] = tmpl
        self.get_logger().info(
            f"cam{cam_id}: K rescaled from {int(msg.width)}x{int(msg.height)} "
            f"to {tmpl.width}x{tmpl.height} (topic)")

    # ── Network ─────────────────────────────────────────────────────
    def _server_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.args.host, self.args.port))
        srv.listen(1)
        srv.settimeout(1.0)

        while not self._stop.is_set():
            try:
                client, addr = srv.accept()
            except socket.timeout:
                continue
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
            self.get_logger().info(f"Source connected from {addr}")
            try:
                self._handle_client(client)
            except (BrokenPipeError, ConnectionResetError) as e:
                self.get_logger().warn(f"Source disconnected: {e}")
            except Exception as e:
                self.get_logger().error(f"Client handler error: {e}")
            finally:
                try:
                    client.close()
                except Exception:
                    pass
        srv.close()

    def _handle_client(self, client: socket.socket):
        while not self._stop.is_set():
            header = recv_exact(client, HEADER_SIZE)
            if header is None:
                return

            magic, cam_id, flags, _, ts_us, w, h, plen = struct.unpack(
                ">4sBB2sQHHI", header)
            if magic != MAGIC:
                self.get_logger().error(f"Bad magic {magic!r}; dropping")
                return
            if cam_id not in self._decoders:
                self.get_logger().warn(f"Unknown cam_id {cam_id}")
                if plen and recv_exact(client, plen) is None:
                    return
                continue
            if not (0 < plen <= 64 << 20):
                self.get_logger().error(f"Bad payload length {plen}")
                return

            payload = recv_exact(client, plen)
            if payload is None:
                return
            self._on_frame(cam_id, ts_us, payload)

    # ── Per-frame processing ────────────────────────────────────────
    def _on_frame(self, cam_id: int, ts_us: int, payload: bytes):
        # Check gate file (cheap: stat once per frame)
        if not self._gate_open and self._gate_path.exists():
            self._gate_open = True
            self.get_logger().info(
                f"Gate file {self._gate_path} detected — publishing.")

        for rgb in self._decoders[cam_id].decode(payload):
            if not self._gate_open:
                continue  # decode (warming codec context) but drop
            if self._cinfo[cam_id] is None:
                continue  # caminfo template not yet received
            self._publish(cam_id, ts_us, rgb)

        # Periodic decoder stats (every 5s)
        now = time.monotonic()
        if now - self._last_log >= 5.0:
            self._last_log = now
            stats = " | ".join(
                f"cam{cid}: in={d.frames_in} out={d.frames_out} "
                f"pub={self._pub_count[cid]}"
                for cid, d in self._decoders.items())
            self.get_logger().info(stats)

    def _publish(self, cam_id: int, ts_us: int, rgb: np.ndarray):
        # Resize 1920x1080 -> OUT_W x OUT_H on GPU (bicubic, antialias).
        t = torch.from_numpy(rgb).to(self._device).float()
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(OUT_H, OUT_W),
                          mode="bicubic", align_corners=False, antialias=True)
        rgb_out = (t.squeeze(0).permute(1, 2, 0)
                   .clamp(0, 255).byte().cpu().numpy())

        sec = int(ts_us // 1_000_000)
        nsec = int((ts_us % 1_000_000) * 1000)

        # Image
        img = Image()
        img.header.stamp.sec = sec
        img.header.stamp.nanosec = nsec
        img.header.frame_id = self._frame_id[cam_id]
        img.height = OUT_H
        img.width = OUT_W
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = OUT_W * 3
        img.data = rgb_out.tobytes()
        self._pub_image[cam_id].publish(img)

        # CameraInfo (same content on both _rect and bare topics)
        ci = CameraInfo()
        ci.header = img.header
        tmpl = self._cinfo[cam_id]
        ci.width = tmpl.width
        ci.height = tmpl.height
        ci.distortion_model = tmpl.distortion_model
        ci.k = list(tmpl.k)
        ci.p = list(tmpl.p)
        ci.d = list(tmpl.d)
        ci.r = list(tmpl.r)
        ci.binning_x = tmpl.binning_x
        ci.binning_y = tmpl.binning_y
        self._pub_cinfo_rect[cam_id].publish(ci)
        self._pub_cinfo[cam_id].publish(ci)

        # Mask (left only)
        if cam_id == self.LEFT_ID and self._pub_mask is not None:
            mask = Image()
            mask.header = img.header
            mask.height = OUT_H
            mask.width = OUT_W
            mask.encoding = "mono8"
            mask.is_bigendian = 0
            mask.step = OUT_W
            mask.data = self._mask.tobytes()
            self._pub_mask.publish(mask)

        self._pub_count[cam_id] += 1

    def shutdown(self):
        self._stop.set()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7040)
    p.add_argument("--left_yaml", default="",
                   help="Camera intrinsics YAML for left. If empty, use "
                        "--left_camera_info_topic instead.")
    p.add_argument("--right_yaml", default="",
                   help="Camera intrinsics YAML for right. If empty, use "
                        "--right_camera_info_topic instead.")
    p.add_argument("--left_camera_info_topic", default="",
                   help="ROS topic to subscribe for left CameraInfo "
                        "(used when --left_yaml is empty).")
    p.add_argument("--right_camera_info_topic", default="",
                   help="ROS topic to subscribe for right CameraInfo "
                        "(used when --right_yaml is empty).")
    p.add_argument("--mask_path", default="",
                   help="Optional segmentation mask PNG. If empty, no "
                        "/left/segmentation topic is published.")
    p.add_argument("--start_signal_file", default="",
                   help="If set, drop all frames until this file exists. "
                        "Use to gate live recording: e.g. `touch /tmp/qpf2_start` "
                        "from another terminal to begin publishing.")
    args = p.parse_args()
    if not args.mask_path:
        args.mask_path = None
    if not args.start_signal_file:
        args.start_signal_file = None

    rclpy.init()
    node = QPF2Receiver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()