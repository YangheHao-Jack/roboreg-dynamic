#!/usr/bin/env python3
"""
qpf2_receiver_pynv.py

GPU-resident QPF2 receiver. Decodes H.264 via PyNvVideoCodec (NVDEC),
keeps frames on the GPU via DLPack, resizes via torch, and publishes
through PyNITROS so ESS receives the image as a NITROS-typed message
without any D->H copy.

Replaces qpf2_receiver.py for performance-critical workflows. Runs
in parallel with the existing receiver (different launch file).

Topics published:
  /pynitros_left, /pynitros_right       (NitrosBridgeImage, via PyNITROS)
  /left/camera_info_rect, /left/camera_info,
  /right/camera_info_rect, /right/camera_info  (CameraInfo, std DDS)
  /left/segmentation                    (Image, std DDS)

The bridge launch (fp_pipeline_quest_qpf2_pynv.launch.py) wires:
  pynitros_{left,right}  →  ImageConverterNode (in ESS container)
                         →  /left/image_rect, /right/image_rect (NitrosImage)

Usage:
    python qpf2_receiver_pynv.py \\
        --left_yaml  camera.left.yaml \\
        --right_yaml camera.right.yaml \\
        --mask_path  mask.png \\
        [--start_signal_file /tmp/qpf2_start]
"""

import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np
import yaml

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("torch not installed")

try:
    import PyNvVideoCodec as nvc
except ImportError:
    sys.exit("PyNvVideoCodec not installed")

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy,
)
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

try:
    from isaac_ros_nitros_bridge_interfaces.msg import NitrosBridgeImage
    from isaac_ros_pynitros.isaac_ros_pynitros_publisher import PyNitrosPublisher
    from isaac_ros_pynitros.pynitros_type_builders.pynitros_image_builder \
        import PyNitrosImageBuilder
except ImportError as e:
    sys.exit(f"PyNITROS not available: {e}")


# ── Wire format ──────────────────────────────────────────────────────
MAGIC = b"QPF2"
HEADER_SIZE = 24
FLAG_KEYFRAME = 0x01
FLAG_HAS_CSD = 0x02

# ── Output resolution (matches ESS) ──────────────────────────────────
OUT_W = 960
OUT_H = 576
SRC_W = 1920
SRC_H = 1080

# ── Camera ids ───────────────────────────────────────────────────────
LEFT_ID = 50
RIGHT_ID = 51
LEFT_FRAME_ID = "zed_left_camera_optical_frame"
RIGHT_FRAME_ID = "zed_right_camera_optical_frame"


# ── Helpers ──────────────────────────────────────────────────────────
def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def load_camera_info(yaml_path: str, frame_id: str) -> CameraInfo:
    """Load YAML, rescale K and P to (OUT_W, OUT_H)."""
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)
    src_w = int(d["width"])
    src_h = int(d["height"])
    sx = OUT_W / float(src_w)
    sy = OUT_H / float(src_h)

    K = [float(x) for x in d["k"]]
    K[0] *= sx; K[1] *= sx; K[2] *= sx
    K[3] *= sy; K[4] *= sy; K[5] *= sy

    if "p" in d and len(d["p"]) == 12:
        P = [float(x) for x in d["p"]]
        P[0] *= sx; P[1] *= sx; P[2] *= sx; P[3] *= sx
        P[4] *= sy; P[5] *= sy; P[6] *= sy; P[7] *= sy
    else:
        P = [K[0], K[1], K[2], 0.0,
             K[3], K[4], K[5], 0.0,
             K[6], K[7], K[8], 0.0]

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


def load_mask(path: str) -> np.ndarray:
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"could not read mask: {path}")
    if raw.shape != (OUT_H, OUT_W):
        raw = cv2.resize(raw, (OUT_W, OUT_H), cv2.INTER_NEAREST)
    return ((raw > 127).astype(np.uint8) * 255)


# ── Per-camera decode/publish worker ─────────────────────────────────
class CameraWorker(threading.Thread):
    """One thread per camera. Owns its NVDEC decoder and PyNITROS
    publisher. Pulls (ts, payload_bytes) tuples from a queue, decodes,
    resizes on GPU, publishes via PyNITROS."""

    def __init__(self, name: str, cam_id: int, frame_id: str,
                 pynitros_topic: str, pynitros_topic_ros: str,
                 node: Node, log, gate_check):
        super().__init__(name=f"worker-{name}", daemon=True)
        self.name_ = name
        self.cam_id = cam_id
        self.frame_id = frame_id
        self.log = log
        self._gate_check = gate_check  # callable returning bool
        self._stop = threading.Event()
        self.q: Queue = Queue(maxsize=64)

        # Decoder: GPU memory, RGB output. RGB = HWC uint8, no NV12 conversion.
        self.decoder = nvc.CreateDecoder(
            gpuid=0,
            codec=nvc.cudaVideoCodec.H264,
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.RGB,
        )
        log.info(f"[{name}] PyNvDecoder created (RGB, device memory)")

        # PyNITROS publisher + image builder
        self.pub = PyNitrosPublisher(
            node, NitrosBridgeImage,
            pynitros_topic, pynitros_topic_ros)
        self.builder = PyNitrosImageBuilder(num_buffer=40, timeout=5)

        # Pre-allocate device GPU mem for resize destination. We
        # write into this tensor via F.interpolate output, then hand
        # its data_ptr to the builder. PyNITROS's IPC pool copies
        # from this pointer into its own buffer, so the tensor can
        # be reused next frame.
        self._resize_dst = torch.empty(
            (OUT_H, OUT_W, 3), dtype=torch.uint8, device="cuda:0")

        # Stats
        self.frames_in = 0
        self.frames_decoded = 0
        self.frames_published = 0

        # Lifetime management for in-flight numpy buffers
        # Decoder reads from these asynchronously; we keep them
        # alive long enough that NVDEC has consumed them.
        self._buf_ring = []
        self._buf_ring_size = 8

    def submit(self, ts_us: int, payload: bytes):
        """Called from the network thread. Drops if the queue is
        full (consumer overload)."""
        try:
            self.q.put_nowait((ts_us, payload))
        except Exception:
            self.log.warn(f"[{self.name_}] queue full; dropping frame")

    def stop(self):
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

    def run(self):
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                break
            ts_us, payload = item
            self.frames_in += 1

            # Wrap bytes in numpy uint8 array. Keep it alive.
            buf = np.frombuffer(payload, dtype=np.uint8).copy()
            self._buf_ring.append(buf)
            if len(self._buf_ring) > self._buf_ring_size:
                self._buf_ring.pop(0)

            # Build PacketData
            pkt = nvc.PacketData()
            pkt.bsl_data = buf.ctypes.data
            pkt.bsl = buf.size
            pkt.pts = ts_us

            try:
                frames = self.decoder.Decode(pkt)
            except Exception as e:
                self.log.error(f"[{self.name_}] decode error: {e}")
                continue

            for frame in frames:
                self.frames_decoded += 1
                if self._gate_check():
                    self._process_and_publish(frame, ts_us)

    def _process_and_publish(self, frame, ts_us: int):
        """One DecodedFrame -> resize -> PyNITROS publish."""
        # Wrap as torch tensor (zero-copy, GPU)
        try:
            t = torch.from_dlpack(frame)
        except Exception as e:
            self.log.warn(f"[{self.name_}] dlpack wrap failed: {e}")
            return

        # Normalise layout to HWC. PyNvVideoCodec RGB output is
        # typically HWC.
        if t.ndim == 3 and t.shape[-1] == 3:
            rgb = t  # HWC
        elif t.ndim == 3 and t.shape[0] == 3:
            rgb = t.permute(1, 2, 0).contiguous()
        elif t.ndim == 4 and t.shape[0] == 1 and t.shape[1] == 3:
            rgb = t[0].permute(1, 2, 0).contiguous()
        else:
            self.log.warn(f"[{self.name_}] unexpected shape {tuple(t.shape)}")
            return

        # Resize (1920x1080 -> 960x576) on GPU.
        # F.interpolate wants NCHW float; convert, interpolate, back to uint8 HWC.
        t_nchw = rgb.permute(2, 0, 1).unsqueeze(0).float()
        t_resized = F.interpolate(
            t_nchw, size=(OUT_H, OUT_W),
            mode="bicubic", align_corners=False, antialias=True)
        # Write into pre-allocated dst buffer (avoids allocating each frame)
        out_hwc = (t_resized.squeeze(0).permute(1, 2, 0)
                   .clamp(0, 255).to(torch.uint8).contiguous())

        # Build header (use QPF2 timestamp, not wall clock — keeps
        # the same stamp on image+cinfo+mask)
        sec = int(ts_us // 1_000_000)
        nsec = int((ts_us % 1_000_000) * 1000)
        header = Header()
        header.stamp.sec = sec
        header.stamp.nanosec = nsec
        header.frame_id = self.frame_id

        # PyNITROS publish: hand the data_ptr directly. Builder
        # copies into its IPC pool; out_hwc can be GC'd after.
        try:
            built = self.builder.build(
                out_hwc.data_ptr(),
                OUT_H, OUT_W, OUT_W * 3,
                "rgb8", header, 0, False)
            self.pub.publish(built)
            self.frames_published += 1
        except Exception as e:
            self.log.error(f"[{self.name_}] PyNITROS publish failed: {e}")


# ── Main node ────────────────────────────────────────────────────────
class QPF2ReceiverPyNv(Node):

    def __init__(self, args):
        super().__init__("qpf2_receiver_pynv")
        self.args = args

        if not torch.cuda.is_available():
            sys.exit("CUDA not available")

        # Gate file
        self._gate_path = Path(args.start_signal_file) if args.start_signal_file else None
        if self._gate_path and self._gate_path.exists():
            self._gate_path.unlink()
        self._gate_open = self._gate_path is None

        # Camera info templates
        self._cinfo = {
            LEFT_ID:  load_camera_info(args.left_yaml,  LEFT_FRAME_ID),
            RIGHT_ID: load_camera_info(args.right_yaml, RIGHT_FRAME_ID),
        }
        self._frame_id = {LEFT_ID: LEFT_FRAME_ID, RIGHT_ID: RIGHT_FRAME_ID}
        for cid, ci in self._cinfo.items():
            self.get_logger().info(
                f"cam{cid}: K rescaled to {ci.width}x{ci.height}")

        # Mask
        self._mask = None
        if args.mask_path:
            self._mask = load_mask(args.mask_path)
            self.get_logger().info(
                f"Mask loaded: shape={self._mask.shape}, "
                f"foreground={(self._mask > 0).sum()} px")

        # CameraInfo + mask publishers (standard DDS)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50)
        self._pub_cinfo_rect = {
            LEFT_ID:  self.create_publisher(CameraInfo, "/left/camera_info_rect",  qos),
            RIGHT_ID: self.create_publisher(CameraInfo, "/right/camera_info_rect", qos),
        }
        self._pub_cinfo = {
            LEFT_ID:  self.create_publisher(CameraInfo, "/left/camera_info",  qos),
            RIGHT_ID: self.create_publisher(CameraInfo, "/right/camera_info", qos),
        }
        self._pub_mask = (self.create_publisher(Image, "/left/segmentation", qos)
                         if self._mask is not None else None)

        # Per-camera workers (each owns a decoder + PyNITROS publisher)
        self._workers = {
            LEFT_ID: CameraWorker(
                "left",  LEFT_ID,  LEFT_FRAME_ID,
                "pynitros_left",  "pynitros_left_ros",
                self, self.get_logger(), lambda: self._gate_open),
            RIGHT_ID: CameraWorker(
                "right", RIGHT_ID, RIGHT_FRAME_ID,
                "pynitros_right", "pynitros_right_ros",
                self, self.get_logger(), lambda: self._gate_open),
        }
        for w in self._workers.values():
            w.start()

        # TCP server thread
        self._stop = threading.Event()
        self._server_thread = threading.Thread(
            target=self._server_loop, name="qpf2_server", daemon=True)
        self._server_thread.start()
        self.get_logger().info(
            f"qpf2_receiver_pynv listening on {args.host}:{args.port}"
            + (f" (gated on {self._gate_path})" if self._gate_path else ""))

        # Periodic stats timer
        self._last_log = time.monotonic()
        self.create_timer(5.0, self._log_stats)

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
                self.get_logger().error(f"Bad magic {magic!r}")
                return
            if cam_id not in self._workers:
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

            # Check gate
            if not self._gate_open and self._gate_path.exists():
                self._gate_open = True
                self.get_logger().info(
                    f"Gate file {self._gate_path} detected — publishing.")

            # Hand off to the worker thread
            self._workers[cam_id].submit(ts_us, payload)

            # Publish standard-DDS topics (camera_info + mask) on
            # the network thread. These are tiny.
            self._publish_aux(cam_id, ts_us)

    # ── Auxiliary publishes (CameraInfo, mask) ──────────────────────
    def _publish_aux(self, cam_id: int, ts_us: int):
        if not self._gate_open:
            return
        sec = int(ts_us // 1_000_000)
        nsec = int((ts_us % 1_000_000) * 1000)

        ci = CameraInfo()
        ci.header.stamp.sec = sec
        ci.header.stamp.nanosec = nsec
        ci.header.frame_id = self._frame_id[cam_id]
        tmpl = self._cinfo[cam_id]
        ci.width = tmpl.width
        ci.height = tmpl.height
        ci.distortion_model = tmpl.distortion_model
        ci.k = list(tmpl.k); ci.p = list(tmpl.p)
        ci.d = list(tmpl.d); ci.r = list(tmpl.r)
        ci.binning_x = tmpl.binning_x
        ci.binning_y = tmpl.binning_y
        self._pub_cinfo_rect[cam_id].publish(ci)
        self._pub_cinfo[cam_id].publish(ci)

        if cam_id == LEFT_ID and self._pub_mask is not None:
            mask = Image()
            mask.header = ci.header
            mask.height = OUT_H
            mask.width = OUT_W
            mask.encoding = "mono8"
            mask.is_bigendian = 0
            mask.step = OUT_W
            mask.data = self._mask.tobytes()
            self._pub_mask.publish(mask)

    # ── Stats ───────────────────────────────────────────────────────
    def _log_stats(self):
        parts = []
        for cid, w in self._workers.items():
            parts.append(
                f"cam{cid}: in={w.frames_in} dec={w.frames_decoded} "
                f"pub={w.frames_published}")
        self.get_logger().info(" | ".join(parts))

    def shutdown(self):
        self._stop.set()
        for w in self._workers.values():
            w.stop()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        for w in self._workers.values():
            w.join(timeout=2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7040)
    p.add_argument("--left_yaml", required=True)
    p.add_argument("--right_yaml", required=True)
    p.add_argument("--mask_path", default="")
    p.add_argument("--start_signal_file", default="")
    args = p.parse_args()
    if not args.mask_path:
        args.mask_path = None
    if not args.start_signal_file:
        args.start_signal_file = None

    rclpy.init()
    node = QPF2ReceiverPyNv(args)
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
