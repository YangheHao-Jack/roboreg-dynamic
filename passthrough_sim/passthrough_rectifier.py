#!/usr/bin/env python3
"""
passthrough_rectifier.py

Clean, single-purpose stereo rectifier for the "passthrough on ROS topics"
simulation. It reproduces the real deployment boundary exactly: the headset
receiver publishes H.264 CompressedImage to ROS topics, and everything
downstream processes that stream. This node sits at that boundary --- it

    1. subscribes to the H.264 CompressedImage stereo pair,
    2. NVDEC-decodes each access unit to the engine-native NV12 surface
       (no library CSC, no clone — nothing on the SMs),
    3. rectifies + letterboxes IN THE NV12 DOMAIN: one composed grid_sample
       per plane (Y, UV) straight from source resolution to 960x576,
    4. colour-converts YCbCr->RGB at OUTPUT resolution (0.55 MP, not the
       decoder's ~3.7 MP — this was the bulk of the codec path's SM tax),
    5. republishes rectified Image + CameraInfo, stamped with the original
       capture time.

The JPEG/nvJPEG path was removed: the producer is H.264-only now, and nvJPEG
ran on the SMs — exactly the contention this node is designed to avoid.
--raw-input (rgb8 Image in, no decode) is kept as the test path; it shares
the same composed-grid resampler.

Inputs (all overridable):
    --left-image-topic    sensor_msgs/CompressedImage  /xr/image_left/compressed
    --right-image-topic   sensor_msgs/CompressedImage  /xr/image_right/compressed
    --left-caminfo-topic  sensor_msgs/CameraInfo       /xr/image_left/camera_info
    --right-caminfo-topic sensor_msgs/CameraInfo       /xr/image_right/camera_info
    --extrinsics-topic    geometry_msgs/PoseStamped    /xr/baseline

Outputs (stamped with the input capture time):
    /left/image_rect,        /right/image_rect          sensor_msgs/Image (rgb8, 960x576)
    /left/camera_info_rect,  /right/camera_info_rect    sensor_msgs/CameraInfo (rectified)
    /left/camera_info,       /right/camera_info         (same content; ESS-side)

Usage:
    python3 passthrough_rectifier.py
    # or override topics, e.g. for a different receiver namespace:
    python3 passthrough_rectifier.py --left-image-topic /quest/image_left/compressed ...
"""

import argparse
import queue
import statistics
import struct
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch not installed: pip install torch --break-system-packages")

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time as RosTime


# ── Output resolution (matches the ESS engine output) ────────────────
OUT_W = 960
OUT_H = 576

# ── Quest Zenoh wire protocol (--zenoh-input; ported verbatim from
#    quest_zenoh_receiver.py — keep the two in sync with the APK) ──────
# Frame attachment, set by ZenohFrameStreamer.kt (16 bytes LE):
#   u64 stamp_us | u16 W | u16 H | u8 codec_id (0=jpeg,1=h264)
#   | u8 flags (bit0=keyframe) | u16 reserved
ZATTACH_FORMAT = "<QHHBBH"
ZATTACH_SIZE = struct.calcsize(ZATTACH_FORMAT)          # 16
ZCODEC_JPEG, ZCODEC_H264 = 0, 1
# Calibration payload (QCAL):
ZCAL_MAGIC = b"QCAL"
ZCAL_HEADER_FORMAT = "<4sBBH"
ZCAL_HEADER_SIZE = struct.calcsize(ZCAL_HEADER_FORMAT)  # 8
ZCAL_CAM_FORMAT = "<BBHHH4f5f3f4f"
ZCAL_CAM_SIZE = struct.calcsize(ZCAL_CAM_FORMAT)        # 72


class CamStats:
    """Per-camera wire statistics for --zenoh-input, ported verbatim from
    quest_zenoh_receiver.CamStats so the report line stays comparable across
    the two ingest paths. 'arrival p50' is the MEDIAN INTER-ARRIVAL GAP (not
    a latency); 'net_jit' is the stdev of (arrival deltas - producer-stamp
    deltas) — network jitter, immune to the headset/PC clock-epoch offset."""

    def __init__(self, name: str, window: int = 600):
        self.name = name
        self.frames_total = 0
        self.recv_ns: deque = deque(maxlen=window)
        self.stamp_us: deque = deque(maxlen=window)
        self.payload_size: deque = deque(maxlen=window)

    def update(self, recv_ns: int, stamp_us: int, payload_bytes: int):
        self.frames_total += 1
        self.recv_ns.append(recv_ns)
        self.stamp_us.append(stamp_us)
        self.payload_size.append(payload_bytes)

    def report(self):
        if len(self.recv_ns) < 2:
            return None
        recv = list(self.recv_ns)
        stamp = list(self.stamp_us)
        size = list(self.payload_size)

        arr_ms   = [(recv[i+1] - recv[i]) / 1e6 for i in range(len(recv) - 1)]
        stamp_ms = [(stamp[i+1] - stamp[i]) / 1e3 for i in range(len(stamp) - 1)]
        net_jit  = [a - s for a, s in zip(arr_ms, stamp_ms)]

        span_s = (recv[-1] - recv[0]) / 1e9
        fps    = (len(recv) - 1) / span_s if span_s > 0 else 0.0
        arr_p50 = sorted(arr_ms)[len(arr_ms) // 2]
        arr_jit = statistics.stdev(arr_ms) if len(arr_ms) > 1 else 0.0
        net_jit_ms = statistics.stdev(net_jit) if len(net_jit) > 1 else 0.0
        size_kb = statistics.mean(size) / 1024
        mbps    = (sum(size) * 8) / (span_s * 1e6) if span_s > 0 else 0.0

        return (f"[{self.name}|h264] frames={self.frames_total:>5d} "
                f"fps={fps:5.1f} arrival p50={arr_p50:5.1f}ms "
                f"arr_jit={arr_jit:4.1f}ms net_jit={net_jit_ms:4.1f}ms "
                f"size={size_kb:5.1f}KB bw={mbps:5.1f}Mbps")

# ── Camera identifiers (kept numeric so the rectification math below is
#    byte-for-byte the same as the validated implementation) ──────────
LEFT_ID = 50
RIGHT_ID = 51


# ── Intrinsics rescale (uniform fit + letterbox) ─────────────────────
def _rescale_K_P(K_in, P_in, src_w, src_h):
    """Return (K, P) rescaled from (src_w, src_h) to (OUT_W, OUT_H) via
    uniform fit-and-letterbox (preserves aspect; pads the short axis). cx,cy
    are shifted by the padding offset so the optical centre stays on the
    actual content within the padded canvas --- required to keep ESS / FP
    geometry correct."""
    s = min(OUT_W / float(src_w), OUT_H / float(src_h))
    inner_w = int(round(src_w * s))
    inner_h = int(round(src_h * s))
    pad_x = (OUT_W - inner_w) // 2
    pad_y = (OUT_H - inner_h) // 2

    K = [float(x) for x in K_in]
    K[0] *= s; K[4] *= s
    K[2] = K[2] * s + pad_x
    K[5] = K[5] * s + pad_y
    K[1] *= s; K[3] *= s
    K[6] *= s; K[7] *= s

    if P_in is not None and len(P_in) == 12:
        P = [float(x) for x in P_in]
        P[0] *= s; P[1] *= s
        P[2] = P[2] * s + pad_x
        P[3] *= s
        P[4] *= s; P[5] *= s
        P[6] = P[6] * s + pad_y
        P[7] *= s
    else:
        P = [K[0], K[1], K[2], 0.0,
             K[3], K[4], K[5], 0.0,
             K[6], K[7], K[8], 0.0]
    return K, P


def template_from_camera_info(src: CameraInfo, frame_id: str) -> CameraInfo:
    """Incoming CameraInfo -> a template with K and P rescaled to
    (OUT_W, OUT_H). _maybe_build_rectification later overwrites K/P/R with the
    rectified geometry once both sides + extrinsics are known."""
    K, P = _rescale_K_P(src.k, src.p, int(src.width), int(src.height))
    msg = CameraInfo()
    msg.width = OUT_W
    msg.height = OUT_H
    msg.distortion_model = src.distortion_model or "plumb_bob"
    msg.k = K
    msg.p = P
    msg.d = [float(x) for x in src.d] if len(src.d) > 0 else [0.0] * 5
    msg.r = ([float(x) for x in src.r] if len(src.r) == 9
             else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    msg.binning_x = int(src.binning_x)
    msg.binning_y = int(src.binning_y)
    msg.header.frame_id = frame_id
    return msg


class PassthroughRectifier(Node):
    LEFT_ID = LEFT_ID
    RIGHT_ID = RIGHT_ID

    def __init__(self, args):
        super().__init__("passthrough_rectifier")

        if not torch.cuda.is_available():
            sys.exit("CUDA not available; this node needs a GPU.")
        self._device = torch.device("cuda:0")

        self._frame_id = {
            self.LEFT_ID:  args.left_frame_id,
            self.RIGHT_ID: args.right_frame_id,
        }

        # State filled by callbacks; rectification builds once both
        # caminfo templates AND the extrinsics have arrived.
        self._cinfo = {self.LEFT_ID: None, self.RIGHT_ID: None}
        self._src_dims = {self.LEFT_ID: None, self.RIGHT_ID: None}  # (w, h)
        self._stereo_T = None
        self._stereo_T_sub = None       # ROS-mode extrinsics sub (None in
                                        # --zenoh-input; the handler's
                                        # unsubscribe guard checks this)
        self._rectify_grids = None      # per-cam (1,OUT_H,OUT_W,2) SOURCE-space
        self._grid_warned = False
        self._dims_warned = {self.LEFT_ID: False, self.RIGHT_ID: False}
        self._pub_count = {self.LEFT_ID: 0, self.RIGHT_ID: 0}
        # Frame accounting (the passthrough receiver's in/out/drop check).
        # recv = image msgs arrived; pub = rectified frames published;
        # drop = arrived but not published (caminfo not ready / decode fail).
        self._recv_count = {self.LEFT_ID: 0, self.RIGHT_ID: 0}
        self._drop_count = {self.LEFT_ID: 0, self.RIGHT_ID: 0}
        self._stat_t0 = time.monotonic()
        self._stat_last_pub = {self.LEFT_ID: 0, self.RIGHT_ID: 0}

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Must match the bridge's raw image publishers: ~22 MB rgb8 frames go
        # BEST_EFFORT + depth 1 (freshest-frame, drop-the-rest) so neither side
        # blocks. caminfo/extrinsics stay RELIABLE on `qos`.
        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publishers ───────────────────────────────────────────────
        self._pub_image = {
            self.LEFT_ID:  self.create_publisher(Image, args.left_rect_topic, qos),
            self.RIGHT_ID: self.create_publisher(Image, args.right_rect_topic, qos),
        }
        self._pub_cinfo_rect = {
            self.LEFT_ID:  self.create_publisher(
                CameraInfo, args.left_caminfo_rect_topic, qos),
            self.RIGHT_ID: self.create_publisher(
                CameraInfo, args.right_caminfo_rect_topic, qos),
        }
        self._pub_cinfo = {
            self.LEFT_ID:  self.create_publisher(
                CameraInfo, args.left_caminfo_out_topic, qos),
            self.RIGHT_ID: self.create_publisher(
                CameraInfo, args.right_caminfo_out_topic, qos),
        }

        # ── H.264 decoder (NVDEC, one persistent stream per eye) ─────
        # output="nv12": NVDEC hands back the engine-native NV12 surface —
        # no library CSC kernel, no clone. The zero-copy contract (consume
        # before next decode) is satisfied by _publish_msgs's blocking D2H.
        self._h264_dec = {}
        if not getattr(args, "raw_input", False):
            from gpu_h264_codec import GpuH264Decoder
            dev = str(self._device)
            self._h264_dec = {
                self.LEFT_ID:  GpuH264Decoder(device=dev, output="nv12"),
                self.RIGHT_ID: GpuH264Decoder(device=dev, output="nv12"),
            }

        # ── Subscriptions ────────────────────────────────────────────
        self._zenoh = None
        if getattr(args, "zenoh_input", False):
            # Direct headset ingest: no ROS input subscriptions at all; the
            # Zenoh workers feed the SAME _process_h264 core. CameraInfo +
            # baseline come from the QCAL calibration key instead of topics.
            self._setup_zenoh_input(args)
        elif getattr(args, "raw_input", False):
            self.create_subscription(
                Image, args.left_image_topic,
                lambda m: self._on_raw(self.LEFT_ID, m), img_qos)
            self.create_subscription(
                Image, args.right_image_topic,
                lambda m: self._on_raw(self.RIGHT_ID, m), img_qos)
            self.get_logger().info(
                f"image in:  L='{args.left_image_topic}' "
                f"R='{args.right_image_topic}' (RAW rgb8 Image, no decode, "
                f"BEST_EFFORT depth=1)")
        else:
            self.create_subscription(
                CompressedImage, args.left_image_topic,
                lambda m: self._on_h264(self.LEFT_ID, m), qos)
            self.create_subscription(
                CompressedImage, args.right_image_topic,
                lambda m: self._on_h264(self.RIGHT_ID, m), qos)
            self.get_logger().info(
                f"image in:  L='{args.left_image_topic}' "
                f"R='{args.right_image_topic}' "
                f"(H264 CompressedImage, NVDEC decode, NV12-domain rectify)")

        if not getattr(args, "zenoh_input", False):
            self.create_subscription(
                CameraInfo, args.left_caminfo_topic,
                lambda m: self._on_cinfo(self.LEFT_ID, m), qos)
            self.create_subscription(
                CameraInfo, args.right_caminfo_topic,
                lambda m: self._on_cinfo(self.RIGHT_ID, m), qos)
            self.get_logger().info(
                f"caminfo in: L='{args.left_caminfo_topic}' "
                f"R='{args.right_caminfo_topic}'")

            self._stereo_T_sub = self.create_subscription(
                PoseStamped, args.extrinsics_topic,
                self._on_stereo_extrinsics, qos)
            self.get_logger().info(
                f"Waiting for stereo extrinsics on '{args.extrinsics_topic}' "
                f"(PoseStamped, right eye in left-eye frame)")
        self.get_logger().info(
            f"image out: L='{args.left_rect_topic}' R='{args.right_rect_topic}' "
            f"(rgb8 {OUT_W}x{OUT_H}); rectification builds once both caminfo "
            f"templates + extrinsics arrive.")

        # Periodic in/out/drop log (every 5 s).
        self.create_timer(5.0, self._log_stats)

    # ── Geometry setup ───────────────────────────────────────────────
    # ── Zenoh direct-headset ingest (--zenoh-input) ──────────────────
    def _setup_zenoh_input(self, args):
        """Subscribe to the headset's Zenoh keys directly (no receiver
        process, no receiver->rectifier ROS hop). Zenoh callbacks fire on
        multiple runtime threads, and each eye's H.264 stream must reach its
        NVDEC decoder serialized and in order — so each camera gets a small
        drop-oldest FIFO drained by ONE dedicated worker thread that calls
        the same _process_h264 core the ROS path uses. Per-camera state
        (decoder, grid, template, counters, publishers) is touched only by
        that camera's worker, so no locks are needed past the queues.
        Calibration arrives on the QCAL key and is fed through the existing
        _on_cinfo/_on_stereo_extrinsics handlers — identical grid build."""
        import zenoh
        prefix = args.zenoh_key_prefix.rstrip('/')
        self._use_producer_stamp = bool(args.use_producer_stamp)
        self._zcal_done = False
        self._zcodec_warned = False
        self._zstop = False

        # Optional /xr republish (bag recording / WebXR compat): forward the
        # compressed bytes + 1 Hz CameraInfo/baseline like the receiver did.
        self._xr_pub = None
        if args.republish_xr:
            ns = args.xr_namespace.rstrip('/')
            xr_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST, depth=10)
            self._xr_pub = {
                self.LEFT_ID: self.create_publisher(
                    CompressedImage, f"{ns}/image_left/compressed", xr_qos),
                self.RIGHT_ID: self.create_publisher(
                    CompressedImage, f"{ns}/image_right/compressed", xr_qos),
            }
            self._xr_cinfo_pub = {
                self.LEFT_ID: self.create_publisher(
                    CameraInfo, f"{ns}/image_left/camera_info", xr_qos),
                self.RIGHT_ID: self.create_publisher(
                    CameraInfo, f"{ns}/image_right/camera_info", xr_qos),
            }
            self._xr_baseline_pub = self.create_publisher(
                PoseStamped, f"{ns}/baseline", xr_qos)
            self._xr_cal_msgs = None          # (ci_l, ci_r, baseline_ps)
            self.create_timer(1.0, self._republish_xr_calibration)
            self.get_logger().info(
                f"--republish-xr: forwarding compressed bytes + calibration "
                f"on {ns}/* for recording/back-compat")

        # Per-camera FIFOs: small + drop-oldest bounds latency if a worker
        # stalls; the stream resyncs at the next IDR (the APK sends SPS/PPS
        # on every IDR).
        self._zq = {self.LEFT_ID: queue.Queue(maxsize=8),
                    self.RIGHT_ID: queue.Queue(maxsize=8)}
        self._zstats = {self.LEFT_ID: CamStats("L"),
                        self.RIGHT_ID: CamStats("R")}

        if args.zenoh_endpoint:
            joined = ",".join(f'"{e}"' for e in args.zenoh_endpoint)
            cfg = zenoh.Config.from_json5(
                '{ mode: "peer", connect: { endpoints: [ ' + joined + ' ] } }')
        else:
            cfg = zenoh.Config()
        self.get_logger().info(
            f"[zenoh] opening session "
            f"(endpoints={args.zenoh_endpoint or '[scouting]'})")
        self._zenoh = zenoh.open(cfg)
        self._zsubs = [
            self._zenoh.declare_subscriber(
                f"{prefix}/image_left/compressed",
                lambda s: self._zenoh_enqueue(s, self.LEFT_ID)),
            self._zenoh.declare_subscriber(
                f"{prefix}/image_right/compressed",
                lambda s: self._zenoh_enqueue(s, self.RIGHT_ID)),
            self._zenoh.declare_subscriber(
                f"{prefix}/calibration", self._on_zenoh_calibration),
        ]
        self._zworkers = [
            threading.Thread(target=self._zenoh_worker, args=(cam,),
                             name=f"zenoh-{cam}", daemon=True)
            for cam in (self.LEFT_ID, self.RIGHT_ID)
        ]
        for w in self._zworkers:
            w.start()
        self.get_logger().info(
            f"image in:  zenoh '{prefix}/image_left|right/compressed' "
            f"(direct headset ingest, NVDEC decode, NV12-domain rectify; "
            f"stamps={'producer' if self._use_producer_stamp else 'arrival'})")

    def _zenoh_enqueue(self, sample, cam_id: int):
        """Zenoh callback (any runtime thread): copy bytes out, hand to this
        camera's worker. Drop-oldest on overflow."""
        try:
            item = (time.monotonic_ns(),          # wire-arrival stamp (stats)
                    bytes(sample.payload),
                    bytes(sample.attachment)
                    if sample.attachment is not None else b"")
        except Exception:
            return
        q = self._zq[cam_id]
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _zenoh_worker(self, cam_id: int):
        """One per camera — the only thread that touches this camera's
        decoder/grid/publishers. Initialise CUDA on this thread, then drain
        the FIFO into the shared _process_h264 core."""
        try:
            _ = torch.zeros(1, device=self._device)
        except Exception as e:
            self.get_logger().error(f"[zenoh-{cam_id}] CUDA init: {e}")
        q = self._zq[cam_id]
        while not self._zstop:
            try:
                recv_ns, payload, attachment = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._zstop or not rclpy.ok():
                break          # rclpy's SIGINT handler may beat our finally
            stamp_us = 0
            codec_id = ZCODEC_H264
            if len(attachment) >= ZATTACH_SIZE:
                stamp_us, _w, _h, codec_id, _flags, _ = struct.unpack(
                    ZATTACH_FORMAT, attachment[:ZATTACH_SIZE])
            self._zstats[cam_id].update(recv_ns, stamp_us, len(payload))
            if codec_id != ZCODEC_H264:
                if not self._zcodec_warned:
                    self._zcodec_warned = True
                    self.get_logger().error(
                        f"zenoh stream codec_id={codec_id} is not H.264 — "
                        f"this rectifier decodes H.264 only; set the APK to "
                        f"the H.264 codec")
                self._drop_count[cam_id] += 1
                self._recv_count[cam_id] += 1
                continue
            if self._use_producer_stamp and stamp_us > 0:
                ts = RosTime(sec=stamp_us // 1_000_000,
                             nanosec=(stamp_us % 1_000_000) * 1000)
            else:
                ts = self.get_clock().now().to_msg()
            try:
                self._process_h264(cam_id, ts, payload)
            except Exception as e:
                if self._zstop or not rclpy.ok():
                    break      # shutdown race, not a real error
                self.get_logger().error(f"[zenoh-{cam_id}] {e}")
                self._drop_count[cam_id] += 1
            if self._xr_pub is not None:
                m = CompressedImage()
                m.header.stamp = ts
                m.header.frame_id = self._frame_id[cam_id]
                m.format = "h264"
                m.data = payload
                self._xr_pub[cam_id].publish(m)

    def _on_zenoh_calibration(self, sample):
        """QCAL payload -> CameraInfo pair + baseline PoseStamped, fed through
        the EXISTING _on_cinfo/_on_stereo_extrinsics handlers so the grid
        build is byte-identical to the ROS-topic path. Parsed once. (Ported
        from quest_zenoh_receiver._on_calibration.)"""
        if self._zcal_done:
            return
        try:
            payload = bytes(sample.payload)
        except Exception:
            return
        if len(payload) < ZCAL_HEADER_SIZE:
            return
        magic, _ver, ncam, _ = struct.unpack(
            ZCAL_HEADER_FORMAT, payload[:ZCAL_HEADER_SIZE])
        if magic != ZCAL_MAGIC or ncam < 2:
            self.get_logger().warn(
                f"zenoh calibration: bad header (magic={magic!r} ncam={ncam})")
            return
        cams = {}
        off = ZCAL_HEADER_SIZE
        for _ in range(ncam):
            if off + ZCAL_CAM_SIZE > len(payload):
                break
            (cam_id, dist_count, w, h, _res,
             fx, fy, cx, cy,
             k1, k2, k3, p1, p2,
             tx, ty, tz,
             _qx, _qy, _qz, _qw) = struct.unpack(
                ZCAL_CAM_FORMAT, payload[off:off + ZCAL_CAM_SIZE])
            off += ZCAL_CAM_SIZE
            cams[cam_id] = dict(
                w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy,
                dist=[k1, k2, k3, p1, p2] if dist_count >= 5 else [0.0] * 5,
                t=(tx, ty, tz))
        if self.LEFT_ID not in cams or self.RIGHT_ID not in cams:
            self.get_logger().warn(
                f"zenoh calibration: missing eye(s); got ids {sorted(cams)}")
            return

        def _ci(cam, frame_id):
            ci = CameraInfo()
            ci.width = int(cam["w"]); ci.height = int(cam["h"])
            ci.distortion_model = "plumb_bob"
            ci.d = [float(x) for x in cam["dist"]]
            fx, fy = float(cam["fx"]), float(cam["fy"])
            cx, cy = float(cam["cx"]), float(cam["cy"])
            ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            ci.header.frame_id = frame_id
            return ci

        ci_l = _ci(cams[self.LEFT_ID], self._frame_id[self.LEFT_ID])
        ci_r = _ci(cams[self.RIGHT_ID], self._frame_id[self.RIGHT_ID])
        tl, tr = cams[self.LEFT_ID]["t"], cams[self.RIGHT_ID]["t"]
        ps = PoseStamped()
        ps.header.frame_id = self._frame_id[self.LEFT_ID]
        ps.pose.position.x = float(tr[0] - tl[0])
        ps.pose.position.y = float(tr[1] - tl[1])
        ps.pose.position.z = float(tr[2] - tl[2])
        ps.pose.orientation.w = 1.0
        self._zcal_done = True
        self.get_logger().info(
            f"zenoh calibration: L fx={cams[self.LEFT_ID]['fx']:.1f} "
            f"R fx={cams[self.RIGHT_ID]['fx']:.1f} "
            f"{cams[self.LEFT_ID]['w']}x{cams[self.LEFT_ID]['h']} -> "
            f"feeding the standard caminfo/extrinsics path")
        self._on_cinfo(self.LEFT_ID, ci_l)
        self._on_cinfo(self.RIGHT_ID, ci_r)
        self._on_stereo_extrinsics(ps)
        if self._xr_pub is not None:
            self._xr_cal_msgs = (ci_l, ci_r, ps)

    def _republish_xr_calibration(self):
        if self._xr_pub is None or self._xr_cal_msgs is None:
            return
        ci_l, ci_r, ps = self._xr_cal_msgs
        now = self.get_clock().now().to_msg()
        ci_l.header.stamp = now; ci_r.header.stamp = now
        ps.header.stamp = now
        self._xr_cinfo_pub[self.LEFT_ID].publish(ci_l)
        self._xr_cinfo_pub[self.RIGHT_ID].publish(ci_r)
        self._xr_baseline_pub.publish(ps)

    def shutdown_zenoh(self):
        if self._zenoh is None:
            return
        self._zstop = True
        for s in getattr(self, "_zsubs", []):
            try:
                s.undeclare()
            except Exception:
                pass
        try:
            self._zenoh.close()
        except Exception:
            pass
        for w in getattr(self, "_zworkers", []):
            try:
                w.join(timeout=1.0)
            except Exception:
                pass


    def _on_stereo_extrinsics(self, msg: PoseStamped):
        """First PoseStamped populates self._stereo_T (4x4 right-in-left).
        Drops the subscription afterwards. xr_ros_bridge publishes this every
        frame on /xr/baseline; for Quest 3 / OpenXR the translation lands at
        ~(0, +/-0.063, 0) --- _maybe_build_rectification uses the L2 norm so the
        axis choice doesn't matter."""
        if self._stereo_T is not None:
            return
        try:
            p = msg.pose.position
            q = msg.pose.orientation
            w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
            norm = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
            w, x, y, z = w / norm, x / norm, y / norm, z / norm
            R = np.array([
                [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
                [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
                [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ], dtype=np.float64)
            HT = np.eye(4, dtype=np.float64)
            HT[:3, :3] = R
            HT[:3, 3] = [float(p.x), float(p.y), float(p.z)]
        except Exception as e:
            self.get_logger().warn(f"extrinsics: failed to parse PoseStamped: {e}")
            return

        baseline_m = float(np.linalg.norm(HT[:3, 3]))
        if not (1e-3 < baseline_m < 1.0):
            self.get_logger().warn(
                f"extrinsics: implausible baseline {baseline_m * 1000:.2f} mm "
                f"--- ignoring this msg")
            return

        self._stereo_T = HT
        self.get_logger().info(
            f"Stereo extrinsics received: "
            f"translation=({HT[0,3]:+.4f}, {HT[1,3]:+.4f}, {HT[2,3]:+.4f}) m, "
            f"baseline = {baseline_m * 1000:.1f} mm")
        if self._stereo_T_sub is not None:
            try:
                self.destroy_subscription(self._stereo_T_sub)
            except Exception:
                pass
            self._stereo_T_sub = None
        self._maybe_build_rectification()

    def _on_cinfo(self, cam_id: int, msg: CameraInfo):
        """First CameraInfo per camera fills the template."""
        if self._cinfo[cam_id] is not None:
            return
        if msg.width <= 0 or msg.height <= 0:
            return
        tmpl = template_from_camera_info(msg, self._frame_id[cam_id])
        self._cinfo[cam_id] = tmpl
        src_w, src_h = int(msg.width), int(msg.height)
        self._src_dims[cam_id] = (src_w, src_h)
        ls = min(OUT_W / float(src_w), OUT_H / float(src_h))
        liw, lih = int(round(src_w * ls)), int(round(src_h * ls))
        lpx, lpy = (OUT_W - liw) // 2, (OUT_H - lih) // 2
        self.get_logger().info(
            f"cam{cam_id}: K rescaled from {src_w}x{src_h} to "
            f"{tmpl.width}x{tmpl.height} via letterbox "
            f"(active {liw}x{lih}, pad x={lpx} y={lpy}, scale={ls:.3f})")
        self._maybe_build_rectification()

    def _maybe_build_rectification(self):
        """Build stereo rectification maps once both caminfo templates are
        populated AND extrinsics arrived. Overwrites each template's K/P/R with
        the rectified geometry ESS/FoundationPose expect, and stores per-eye
        grid_sample LUTs that _publish applies per frame."""
        if self._stereo_T is None:
            return
        if self._rectify_grids is not None:
            return
        if self._cinfo[self.LEFT_ID] is None or self._cinfo[self.RIGHT_ID] is None:
            return

        K_l = np.array(self._cinfo[self.LEFT_ID].k, dtype=np.float64).reshape(3, 3)
        K_r = np.array(self._cinfo[self.RIGHT_ID].k, dtype=np.float64).reshape(3, 3)

        # Parallel-axis stereo (R = I); only the rendered FOV is asymmetric.
        # OpenCV stereoRectify wants T = origin_left - origin_right, i.e.
        # negative along +X for a right-of-left rig, so P_right[0,3] comes out
        # as -fx*baseline (ROS standard) and downstream reads T = -P[0,3]/fx.
        baseline = float(np.linalg.norm(self._stereo_T[:3, 3]))
        R = np.eye(3)
        T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        image_size = (OUT_W, OUT_H)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_l, dist, K_r, dist, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        map_lx, map_ly = cv2.initUndistortRectifyMap(
            K_l, dist, R1, P1, image_size, cv2.CV_32FC1)
        map_rx, map_ry = cv2.initUndistortRectifyMap(
            K_r, dist, R2, P2, image_size, cv2.CV_32FC1)

        def _maps_to_src_grid(mapx, mapy, src_w, src_h):
            # The cv2 maps give, per OUTPUT pixel, the source location in the
            # LETTERBOXED canvas. Compose with the inverse letterbox so the
            # grid samples the ORIGINAL source directly: one grid_sample from
            # (src_h, src_w) to (OUT_H, OUT_W), no resize, no pad. The inverse
            # x = (x' - pad)/s mirrors the K rescale convention exactly
            # (x' = s*x + pad), so the image resampling and the published K
            # now share one convention. (The old resize+pad+grid_sample chain
            # carried a ~0.5*(1-s)/s px offset from align_corners=False
            # bilinear vs the K convention — sub-pixel, now gone.)
            # Normalised grid coords are resolution-independent, so the SAME
            # grid samples both the full-res Y plane and the half-res UV
            # plane of an NV12 frame (the UV fetch lands on chroma centres,
            # i.e. an implicit bilinear chroma upsample matching the old
            # nv12_to_rgb_chw interpolation).
            s = min(OUT_W / float(src_w), OUT_H / float(src_h))
            inner_w = int(round(src_w * s))
            inner_h = int(round(src_h * s))
            pad_x = (OUT_W - inner_w) // 2
            pad_y = (OUT_H - inner_h) // 2
            sx = (mapx - pad_x) / s
            sy = (mapy - pad_y) / s
            gx = (2.0 * sx + 1.0) / float(src_w) - 1.0
            gy = (2.0 * sy + 1.0) / float(src_h) - 1.0
            grid = np.stack([gx, gy], axis=-1)            # (H, W, 2)
            return (torch.from_numpy(grid).to(self._device)
                    .float().unsqueeze(0))                # (1, H, W, 2)

        self._rectify_grids = {
            self.LEFT_ID:  _maps_to_src_grid(map_lx, map_ly,
                                             *self._src_dims[self.LEFT_ID]),
            self.RIGHT_ID: _maps_to_src_grid(map_rx, map_ry,
                                             *self._src_dims[self.RIGHT_ID]),
        }

        for cam_id, P_new, R_new in (
            (self.LEFT_ID, P1, R1), (self.RIGHT_ID, P2, R2),
        ):
            ci = self._cinfo[cam_id]
            ci.k = [float(x) for x in P_new[:3, :3].flatten()]
            ci.p = [float(x) for x in P_new.flatten()]
            ci.r = [float(x) for x in R_new.flatten()]
            ci.d = [0.0] * 5

        fx, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
        Tx_px = P2[0, 3]
        T_meters = -Tx_px / fx
        ok = "OK" if T_meters > 0 else "BAD (NEGATIVE - sign convention error)"
        self.get_logger().info(
            f"Stereo rectification ready: K_rect(fx={fx:.2f}, cx={cx:.2f}, "
            f"cy={cy:.2f}), P_right[0,3]={Tx_px:.2f} px, "
            f"T_baseline = {T_meters * 1000:.2f} mm "
            f"(expected ~ +{baseline * 1000:.2f} mm) [{ok}]")

    # ── Per-frame: decode -> rectify -> publish ──────────────────────
    def _grids_ready(self, cam_id: int) -> bool:
        """Composed grids exist (caminfo x2 + extrinsics arrived). Until then
        frames are dropped (counted); grids build within ~1 frame because the
        bridge publishes /xr/baseline every frame."""
        if self._rectify_grids is not None:
            return True
        if not self._grid_warned:
            self.get_logger().info(
                "rectification grids not built yet (waiting for both caminfo "
                "+ extrinsics) — dropping frames until ready")
            self._grid_warned = True
        return False

    def _check_dims(self, cam_id: int, w: int, h: int) -> bool:
        """The composed grid is built from caminfo dims; a stream at any other
        resolution would be silently mis-sampled, so verify and drop."""
        if (w, h) == self._src_dims[cam_id]:
            return True
        if not self._dims_warned[cam_id]:
            self.get_logger().error(
                f"cam{cam_id}: frame {w}x{h} != caminfo "
                f"{self._src_dims[cam_id]} — dropping (grid built from "
                f"caminfo dims)")
            self._dims_warned[cam_id] = True
        return False

    def _on_h264(self, cam_id: int, msg: CompressedImage):
        """ROS transport wrapper: unwrap the CompressedImage and hand the
        bytes to the transport-agnostic core (shared with --zenoh-input)."""
        self._process_h264(cam_id, msg.header.stamp, bytes(msg.data))

    def _process_h264(self, cam_id: int, ts, payload: bytes):
        """NVDEC-decode the H.264 access unit to the ENGINE-NATIVE NV12
        surface (one persistent decoder per eye; no library CSC, no clone),
        rectify + letterbox IN THE NV12 DOMAIN (one grid_sample per plane,
        straight from source res to 960x576), then colour-convert at OUTPUT
        resolution. Total SM work per frame: two small grid_samples + a
        0.55 MP CSC — vs the old path's full-res library CSC + clone + float
        cast + resize + pad + grid_sample.

        decode() returns None during lock-on (until the first IDR) -> drop.
        The NV12 tensor VIEWS the decoder's reused surface pool; the blocking
        D2H at the end of _publish_msgs materialises everything that read it
        before this camera's next decode (the zero-copy contract — each
        camera's frames are processed serially, by the rclpy executor in ROS
        mode or by that camera's dedicated worker in --zenoh-input mode)."""
        self._recv_count[cam_id] += 1
        if self._cinfo[cam_id] is None or not self._grids_ready(cam_id):
            self._drop_count[cam_id] += 1
            return
        nv12 = self._h264_dec[cam_id].decode(payload)
        if nv12 is None:
            self._drop_count[cam_id] += 1
            return                                   # NVDEC warmup / no frame yet
        h32, W = int(nv12.shape[0]), int(nv12.shape[1])
        H = (h32 * 2) // 3
        if not self._check_dims(cam_id, W, H):
            self._drop_count[cam_id] += 1
            return

        grid = self._rectify_grids[cam_id]
        # Y: (H,W) -> rectified (OUT_H,OUT_W), sampled at source res.
        y = nv12[:H].to(torch.float32).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        y_r = F.grid_sample(y, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        # UV: (H/2,W/2,2) -> (1,2,H/2,W/2); the SAME normalised grid samples
        # the half-res plane (implicit bilinear chroma upsample).
        uv = (nv12[H:].reshape(H // 2, W // 2, 2)
              .permute(2, 0, 1).unsqueeze(0).to(torch.float32))    # (1,2,h,w)
        uv_r = F.grid_sample(uv, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
        # Letterbox pad zones sample outside the source -> Y=Cb=Cr=0, which
        # the studio-swing CSC maps slightly off pure black; clamp them.
        # (Mask from the grid: any coordinate outside [-1,1] is padding.)
        # CSC at OUTPUT resolution (0.55 MP) — this was the codec tax.
        from gpu_h264_codec import ycbcr_planes_to_rgb
        rgb = ycbcr_planes_to_rgb(y_r[0, 0], uv_r[0, 0], uv_r[0, 1])
        oob = ((grid[0, ..., 0].abs() > 1.0) |
               (grid[0, ..., 1].abs() > 1.0))
        rgb[:, oob] = 0.0
        rgb_out = (rgb.round().to(torch.uint8).permute(1, 2, 0)
                   .contiguous().cpu().numpy())                    # single D2H
        self._publish_msgs(cam_id, ts, rgb_out)

    def _on_raw(self, cam_id: int, msg: "Image"):
        """TEST path: raw rgb8 Image in, no decode. Upload -> one composed
        grid_sample (source res -> rectified 960x576), stamped with the
        input's capture time (same as the H.264 path)."""
        self._recv_count[cam_id] += 1
        if self._cinfo[cam_id] is None or not self._grids_ready(cam_id):
            self._drop_count[cam_id] += 1
            return
        ts = msg.header.stamp
        try:
            h = int(msg.height); w = int(msg.width)
            arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(h, w, 3)
            chw = np.ascontiguousarray(arr.transpose(2, 0, 1))   # (3,H,W) rgb8
            rgb = torch.from_numpy(chw).to(self._device, non_blocking=True)
        except Exception as e:
            self.get_logger().warn(f"cam{cam_id}: raw reshape failed: {e}")
            self._drop_count[cam_id] += 1
            return
        if not self._check_dims(cam_id, w, h):
            self._drop_count[cam_id] += 1
            return
        t = rgb.unsqueeze(0).float()                             # (1,3,H,W)
        t = F.grid_sample(t, self._rectify_grids[cam_id], mode="bilinear",
                          padding_mode="zeros", align_corners=False)
        rgb_out = (t[0].clamp(0, 255).round().to(torch.uint8)
                   .permute(1, 2, 0).contiguous().cpu().numpy())  # single D2H
        self._publish_msgs(cam_id, ts, rgb_out)

    def _publish_msgs(self, cam_id: int, ts, rgb_out: "np.ndarray"):
        """rgb_out: (OUT_H,OUT_W,3) uint8 numpy (already on the host — the
        .cpu() in the callers is the blocking D2H that also enforces the
        NV12 zero-copy contract). Publish Image + CameraInfo."""
        img = Image()
        img.header.stamp = ts
        img.header.frame_id = self._frame_id[cam_id]
        img.height = OUT_H
        img.width = OUT_W
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = OUT_W * 3
        img.data = rgb_out.tobytes()
        self._pub_image[cam_id].publish(img)

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

        self._pub_count[cam_id] += 1


    def _log_stats(self):
        """Periodic in/out/drop report for the passthrough receiver."""
        now = time.monotonic()
        dt = now - self._stat_t0
        if dt <= 0:
            return
        parts = []
        for cam_id, name in ((self.LEFT_ID, "cam50"), (self.RIGHT_ID, "cam51")):
            recv = self._recv_count[cam_id]
            pub = self._pub_count[cam_id]
            drop = self._drop_count[cam_id]
            rate = (pub - self._stat_last_pub[cam_id]) / dt
            self._stat_last_pub[cam_id] = pub
            pct = (100.0 * drop / recv) if recv else 0.0
            parts.append(f"{name}: in={recv} out={pub} drop={drop} "
                         f"({pct:.0f}%) | {rate:.1f}pub/s")
        self.get_logger().info(" | ".join(parts))
        # In --zenoh-input mode, also emit the receiver-style wire stats so
        # the two ingest paths stay directly comparable in logs.
        if self._zenoh is not None:
            for cam_id in (self.LEFT_ID, self.RIGHT_ID):
                line = self._zstats[cam_id].report()
                if line:
                    self.get_logger().info(line)
        self._stat_t0 = now


def build_parser():
    p = argparse.ArgumentParser(
        description="H.264 stereo rectifier (passthrough-on-ROS-topics "
                    "simulation boundary). NVDEC decode + NV12-domain "
                    "rectification.")
    # Inputs
    p.add_argument("--left-image-topic", default="/xr/image_left/compressed")
    p.add_argument("--right-image-topic", default="/xr/image_right/compressed")
    p.add_argument("--zenoh-input", action="store_true",
                   help="DEPLOYMENT: ingest the headset's Zenoh stream "
                        "directly (no receiver process, no receiver->"
                        "rectifier ROS hop). Subscribes "
                        "<prefix>/image_left|right/compressed + "
                        "<prefix>/calibration; CameraInfo + baseline come "
                        "from the QCAL key. All ROS input topics are "
                        "ignored in this mode; outputs are unchanged.")
    p.add_argument("--zenoh-endpoint", action="append", default=[],
                   help="Zenoh endpoint(s); default = multicast scouting.")
    p.add_argument("--zenoh-key-prefix", default="quest",
                   help="Zenoh key prefix (default 'quest').")
    p.add_argument("--use-producer-stamp", action="store_true",
                   help="(--zenoh-input) Stamp outputs with the headset's "
                        "capture time from the frame attachment instead of "
                        "arrival time. Requires Quest/PC clock sync (NTP) "
                        "for absolute latency numbers to be meaningful.")
    p.add_argument("--republish-xr", action="store_true",
                   help="(--zenoh-input) Also forward the compressed bytes "
                        "on <ns>/image_*/compressed + CameraInfo/baseline "
                        "at 1 Hz, replicating the receiver's /xr topics for "
                        "bag recording / WebXR replay compatibility.")
    p.add_argument("--xr-namespace", default="/xr",
                   help="Namespace for --republish-xr topics.")
    p.add_argument("--raw-input", action="store_true",
                   help="TEST: subscribe to raw rgb8 sensor_msgs/Image instead "
                        "of H.264 CompressedImage (no decode). Matches the "
                        "producer's --raw-images. Auto-strips a trailing "
                        "'/compressed' from the image topics.")
    p.add_argument("--left-caminfo-topic", default="/xr/image_left/camera_info")
    p.add_argument("--right-caminfo-topic", default="/xr/image_right/camera_info")
    p.add_argument("--extrinsics-topic", default="/xr/baseline")
    # Outputs
    p.add_argument("--left-rect-topic", default="/left/image_rect")
    p.add_argument("--right-rect-topic", default="/right/image_rect")
    p.add_argument("--left-caminfo-rect-topic", default="/left/camera_info_rect")
    p.add_argument("--right-caminfo-rect-topic", default="/right/camera_info_rect")
    p.add_argument("--left-caminfo-out-topic", default="/left/camera_info")
    p.add_argument("--right-caminfo-out-topic", default="/right/camera_info")
    # Frame ids. MUST match what downstream selectors expect: FoundationPose
    # matches (stamp, frame_id) across image/depth/segmentation/camera_info,
    # and stereo_depth_saver / fp_pose_recorder both default to
    # zed_left_camera_optical_frame. Diverging here silently stops FP firing.
    p.add_argument("--left-frame-id", default="zed_left_camera_optical_frame")
    p.add_argument("--right-frame-id", default="zed_right_camera_optical_frame")
    return p


def main():
    args = build_parser().parse_args()
    if args.zenoh_input and args.raw_input:
        sys.exit("--zenoh-input and --raw-input are mutually exclusive")
    if args.raw_input:
        # Raw Image topics drop the '/compressed' suffix the bridge omits.
        if args.left_image_topic.endswith("/compressed"):
            args.left_image_topic = args.left_image_topic[:-len("/compressed")]
        if args.right_image_topic.endswith("/compressed"):
            args.right_image_topic = args.right_image_topic[:-len("/compressed")]
    rclpy.init()
    node = PassthroughRectifier(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_zenoh()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()