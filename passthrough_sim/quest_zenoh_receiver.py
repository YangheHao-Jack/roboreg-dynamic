#!/usr/bin/env python3
"""
quest_zenoh_receiver.py
=======================
Subscribes to the Quest 3 Zenoh H.264 topics and:

  1. Republishes them as ROS2 topics. Two modes (--ros-format):
       - 'passthrough' (default): sensor_msgs/CompressedImage on
         /xr/image_{left,right}/compressed with the source bytes UNCHANGED,
         format "h264". No transcode — the rectifier does the single GPU
         (NVDEC) decode downstream, the "decode once, on the GPU" path.
       - 'raw': decode once here and publish rgb8 sensor_msgs/Image on
         /xr/image_{left,right} (no /compressed) — feed a rectifier running
         --raw-input.

  2. Optionally (--view, default on) shows cv2 windows for a live check.

  3. Optionally (--record-bag DIR) records the received WIRE stream to a
     rosbag2 (mcap): source-codec CompressedImage bytes + CameraInfo x2 +
     baseline — replays into the rectifier's H.264 path like the live topics.

Decoding is serialized per camera on a dedicated worker thread (PyAV's
CodecContext is not thread-safe), fed in order from the Zenoh callback.

Attachment wire format (set by ZenohFrameStreamer.kt, 16 bytes LE):
    u64 stamp_us  | u16 W | u16 H | u8 codec_id (0=jpeg,1=h264)
    | u8 flags (bit0=keyframe) | u16 reserved

Quick start
-----------
    python3 quest_zenoh_receiver.py            # cv2 + ROS
    python3 quest_zenoh_receiver.py --no-view  # ROS only (headless)
    python3 quest_zenoh_receiver.py --no-ros   # cv2 only

Viewing in rqt
--------------
    ros2 run rqt_image_view rqt_image_view /xr/image_left
"""

import argparse
import os
import queue
import statistics
import struct
import threading
import time
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np
import zenoh

try:
    import av  # PyAV — H.264 decode (cv2 viewer + JPEG preview re-encode)
except ImportError:
    av = None  # type: ignore

try:
    import torch  # NVDEC decode path (GpuTranscoder)
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image
    from geometry_msgs.msg import PoseStamped
    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False

ATTACHMENT_FORMAT = "<QHHBBH"   # stamp_us | W | H | codec_id | flags | reserved
ATTACHMENT_SIZE = struct.calcsize(ATTACHMENT_FORMAT)  # 16

# Calibration wire format (set by ZenohFrameStreamer.publishCalibration, LE):
#   header  : 4s "QCAL" | u8 version | u8 numCameras | u16 reserved      (8)
#   per cam : u8 id | u8 distCount | u16 W | u16 H | u16 reserved
#             | 4f fx,fy,cx,cy | 5f k1..k5 | 3f tx,ty,tz | 4f qx,qy,qz,qw (72)
CAL_MAGIC = b"QCAL"
CAL_HEADER_FORMAT = "<4sBBH"
CAL_HEADER_SIZE = struct.calcsize(CAL_HEADER_FORMAT)   # 8
CAL_CAM_FORMAT = "<BBHHH4f5f3f4f"
CAL_CAM_SIZE = struct.calcsize(CAL_CAM_FORMAT)         # 72

CODEC_JPEG = 0
CODEC_H264 = 1
CODEC_NAMES = {CODEC_JPEG: "jpeg", CODEC_H264: "h264"}


# --------------------------------------------------------------------------
#  Per-camera stats
# --------------------------------------------------------------------------

class CamStats:
    def __init__(self, name: str, window: int = 600):
        self.name = name
        self.frames_total = 0
        self.recv_ns: deque = deque(maxlen=window)
        self.stamp_us: deque = deque(maxlen=window)
        self.payload_size: deque = deque(maxlen=window)
        self.codec_seen = "—"

    def update(self, recv_ns: int, stamp_us: int, payload_bytes: int, codec_name: str):
        self.frames_total += 1
        self.recv_ns.append(recv_ns)
        self.stamp_us.append(stamp_us)
        self.payload_size.append(payload_bytes)
        self.codec_seen = codec_name

    def report(self) -> Optional[str]:
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

        return (f"[{self.name}|{self.codec_seen:>4}] frames={self.frames_total:>5d} "
                f"fps={fps:5.1f} arrival p50={arr_p50:5.1f}ms "
                f"arr_jit={arr_jit:4.1f}ms net_jit={net_jit_ms:4.1f}ms "
                f"size={size_kb:5.1f}KB bw={mbps:5.1f}Mbps")


# --------------------------------------------------------------------------
#  Decoders for the cv2 viewer
# --------------------------------------------------------------------------

class H264Decoder:
    """PyAV H.264 decoder, one per camera."""
    def __init__(self):
        if av is None:
            raise RuntimeError("PyAV not installed. pip install av")
        self.codec = av.CodecContext.create("h264", "r")
        self.codec.options = {"flags": "low_delay", "thread_type": "slice"}

    def decode(self, nal_bytes: bytes) -> list:
        out = []
        for pkt in self.codec.parse(nal_bytes):
            try:
                for frame in self.codec.decode(pkt):
                    out.append(frame.to_ndarray(format="bgr24"))
            except av.error.InvalidDataError:
                pass
        return out


class Decoders:
    """Per-camera decode dispatch (PyAV for H.264, OpenCV for JPEG). Used for
    both the cv2 viewer and the JPEG preview re-encode. Runs on the Zenoh
    callback thread, which delivers each camera's frames in order — so the
    H.264 decoder sees a continuous stream (no mid-GOP drops)."""
    def __init__(self):
        self._h264: Dict[str, H264Decoder] = {}

    def decode(self, label: str, codec_id: int, payload: bytes) -> list:
        if codec_id == CODEC_JPEG:
            arr = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return [img] if img is not None else []
        if codec_id == CODEC_H264:
            dec = self._h264.get(label)
            if dec is None:
                dec = H264Decoder()
                self._h264[label] = dec
                print(f"[decoder] H.264 decoder created for cam {label}")
            return dec.decode(payload)
        return []


# --------------------------------------------------------------------------
#  All-GPU H.264 -> JPEG transcoder (used only in --gpu-decode mode).
#  NVDEC decode + nvJPEG encode; the frame never leaves the GPU until the
#  final JPEG byte download. Mirrors stereo_pipeline_export.py's
#  NVDEC decode via PyNvVideoCodec. One instance per camera,
#  created and used ONLY on that camera's worker thread (owns CUDA context).
# --------------------------------------------------------------------------

def _nv12_to_rgb_chw_float(nv12, height: int, width: int):
    """NV12 CUDA uint8 (H*3/2, W) -> RGB CHW float[0,1] CUDA, BT.601 limited
    range (same as stereo_pipeline_export.py)."""
    import torch.nn.functional as F
    Y  = nv12[:height].float()
    UV = nv12[height:].view(height // 2, width // 2, 2).float()
    UV_full = F.interpolate(
        UV.permute(2, 0, 1).unsqueeze(0),
        size=(height, width), mode='bilinear', align_corners=False
    ).squeeze(0)
    U, V = UV_full[0], UV_full[1]
    Y_n = (Y - 16.0)  / 219.0
    U_n = (U - 128.0) / 224.0
    V_n = (V - 128.0) / 224.0
    R = (Y_n + 1.402    * V_n).clamp_(0.0, 1.0)
    G = (Y_n - 0.344136 * U_n - 0.714136 * V_n).clamp_(0.0, 1.0)
    B = (Y_n + 1.772    * U_n).clamp_(0.0, 1.0)
    return torch.stack([R, G, B], dim=0)


class GpuTranscoder:
    """Per-camera NVDEC decode (viewer / raw mode). Constructed and called on
    a single worker thread."""
    def __init__(self, label: str):
        self.label = label
        self._dec = None
        self._nvc = None

    def _ensure(self):
        if self._dec is not None:
            return
        import PyNvVideoCodec as nvc
        self._nvc = nvc
        self._dec = nvc.CreateDecoder(
            gpuid=0, codec=nvc.cudaVideoCodec.H264,
            cudacontext=0, cudastream=0, usedevicememory=True,
        )

    def decode_rgb(self, payload: bytes):
        """H.264 Annex-B packet -> RGB CHW uint8 CUDA tensor (or None on
        parser warmup / decode failure)."""
        self._ensure()
        arr = np.frombuffer(payload, dtype=np.uint8)
        pd = self._nvc.PacketData()
        pd.bsl_data = arr.ctypes.data
        pd.bsl = int(arr.size)
        try:
            frames = self._dec.Decode(pd) or []
        except Exception as e:
            print(f"[gpu-decode] NVDEC error ({self.label}): {e}")
            return None
        if not frames:
            return None
        nv12 = torch.from_dlpack(frames[-1]).clone()
        h32, w = nv12.shape
        height = (h32 * 2) // 3
        rgb_f = _nv12_to_rgb_chw_float(nv12, height, w)
        return (rgb_f * 255.0).round().clamp_(0, 255).to(torch.uint8)

    @staticmethod
    def rgb_to_bgr_hwc(rgb_u8):
        """RGB CHW uint8 CUDA -> BGR HWC numpy (for the cv2 viewer)."""
        return rgb_u8[[2, 1, 0]].permute(1, 2, 0).contiguous().cpu().numpy()


# --------------------------------------------------------------------------
#  cv2 viewer — main thread
# --------------------------------------------------------------------------

class CvViewer:
    def __init__(self, scale: float = 0.5):
        self.scale = scale
        self.latest: Dict[str, np.ndarray] = {}
        self.latest_ts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.stop_requested = False

    def submit(self, label: str, ts_us: int, bgr: np.ndarray):
        with self.lock:
            self.latest[label] = bgr
            self.latest_ts[label] = ts_us

    def run(self):
        print("[viewer] press 'q' in any window to quit")
        while not self.stop_requested:
            with self.lock:
                snapshot = dict(self.latest)
                ts_snap = dict(self.latest_ts)
            for cam, frame in snapshot.items():
                img = frame
                if self.scale != 1.0:
                    img = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                                     interpolation=cv2.INTER_AREA)
                label = f"cam {cam}  ts={ts_snap.get(cam, 0)/1e6:.3f}s"
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(f"Quest cam {cam}", img)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                self.stop_requested = True
                break
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------
#  Receiver
# --------------------------------------------------------------------------

class QuestReceiver:
    def __init__(self, namespace: str, endpoints: list, stats_period_s: float,
                 use_producer_stamp: bool, enable_ros: bool,
                 gpu_decode: bool,
                 viewer: Optional[CvViewer], ros_format: str = "passthrough",
                 record_bag: Optional[str] = None,
                 save_calib: Optional[str] = None,
                 baseline_frame: str = "eye"):
        self.viewer = viewer
        self.use_producer_stamp = use_producer_stamp
        self._stats_period_s = stats_period_s
        self.stats_left  = CamStats("L")
        self.stats_right = CamStats("R")
        self.decoders = Decoders()
        self._last_report = time.monotonic()
        self.gpu_decode = gpu_decode
        self.ros_format = ros_format
        needs_decode = (viewer is not None) or (ros_format == "raw")
        if not needs_decode:
            print("[decode] none: passthrough relay (no decoder will run)")
        elif self.gpu_decode:
            if not _HAS_TORCH:
                raise RuntimeError(
                    "--gpu-decode needs torch + CUDA. None available.")
            print("[gpu-decode] NVDEC decode (viewer / raw mode)")
        else:
            print("[decode] CPU mode: PyAV decode (viewer / raw mode)")

        # ROS
        self.node = None
        self.pub_left = self.pub_right = None
        # Bag recording (--record-bag): writes the WIRE stream — source-codec
        # CompressedImage bytes (h264/jpeg as received, NO transcode, in every
        # --ros-format mode) + CameraInfo x2 + baseline — so the bag replays
        # into the rectifier's H.264 path exactly like the live /xr topics.
        self._bag_writer = None
        self._bag_lock = threading.Lock()
        self._bag_serialize = None
        # Calibration -> CameraInfo / baseline state (filled by _on_calibration)
        self.pub_cinfo_left = self.pub_cinfo_right = self.pub_baseline = None
        self.sub_calibration = None
        self._cal_lock = threading.Lock()
        self._have_calibration = False
        self._cinfo_left_msg = self._cinfo_right_msg = self._baseline_msg = None
        self._cinfo_frame_left = "xr_left_optical"
        self._cinfo_frame_right = "xr_right_optical"
        if enable_ros:
            if not _HAS_ROS:
                raise RuntimeError("rclpy not available. Source ROS2 or pass --no-ros.")
            rclpy.init()
            self.node = rclpy.create_node('quest_zenoh_receiver')
            qos = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            ns = namespace.rstrip('/')
            if self.ros_format == "raw":
                # Decode once, publish the decoded frame as raw rgb8 Image (no
                # JPEG re-encode here, no JPEG decode in the rectifier — the
                # "decode once, keep raw downstream" path). ~22 MB/frame, so
                # BEST_EFFORT depth=1 (freshest-frame; matches the rectifier's
                # --raw-input QoS). Topics drop the /compressed suffix.
                img_qos = QoSProfile(depth=1,
                                     reliability=ReliabilityPolicy.BEST_EFFORT,
                                     durability=DurabilityPolicy.VOLATILE)
                self.pub_left  = self.node.create_publisher(
                    Image, f"{ns}/image_left",  img_qos)
                self.pub_right = self.node.create_publisher(
                    Image, f"{ns}/image_right", img_qos)
                self.node.create_timer(self._stats_period_s, self._maybe_report)
                self.node.get_logger().info(
                    f"ros_format=raw: rgb8 Image on {ns}/image_left|right "
                    f"(decode-once, no JPEG; run rectifier with --raw-input)")
            else:
                # passthrough: forward the source bytes unchanged (h264).
                self.pub_left  = self.node.create_publisher(
                    CompressedImage, f"{ns}/image_left/compressed",  qos)
                self.pub_right = self.node.create_publisher(
                    CompressedImage, f"{ns}/image_right/compressed", qos)
                self.node.create_timer(self._stats_period_s, self._maybe_report)
                self.node.get_logger().info(
                    f"ros_format=passthrough: CompressedImage on "
                    f"{ns}/image_left|right/compressed "
                    f"(source bytes forwarded as-is; no transcode)")

            # CameraInfo + stereo baseline, built from the APK's streamed
            # calibration (quest/calibration). TRANSIENT_LOCAL latches for
            # durable subs; the 1 Hz timer covers the rectifier's VOLATILE
            # subs regardless of start order. Topics match what the rectifier
            # (--raw-input) and the IPCAI consumer's optimiser scene read.
            cinfo_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pub_cinfo_left = self.node.create_publisher(
                CameraInfo, f"{ns}/image_left/camera_info", cinfo_qos)
            self.pub_cinfo_right = self.node.create_publisher(
                CameraInfo, f"{ns}/image_right/camera_info", cinfo_qos)
            self.pub_baseline = self.node.create_publisher(
                PoseStamped, f"{ns}/baseline", cinfo_qos)
            self._cinfo_frame_left = f"{ns.strip('/')}_left_optical"
            self._cinfo_frame_right = f"{ns.strip('/')}_right_optical"
            self.node.create_timer(1.0, self._publish_caminfo)
            self.node.get_logger().info(
                f"awaiting calibration on Zenoh 'quest/calibration' -> "
                f"CameraInfo on {ns}/image_left|right/camera_info + baseline "
                f"on {ns}/baseline")

        if record_bag:
            if self.node is None:
                raise RuntimeError("--record-bag needs ROS (drop --no-ros).")
            import rosbag2_py
            from rclpy.serialization import serialize_message
            self._bag_serialize = serialize_message
            writer = rosbag2_py.SequentialWriter()
            writer.open(
                rosbag2_py.StorageOptions(uri=record_bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions(
                    input_serialization_format='cdr',
                    output_serialization_format='cdr'))
            ns = namespace.rstrip('/')
            for tid, (topic, ttype) in enumerate([
                    (f"{ns}/image_left/compressed",  'sensor_msgs/msg/CompressedImage'),
                    (f"{ns}/image_right/compressed", 'sensor_msgs/msg/CompressedImage'),
                    (f"{ns}/image_left/camera_info",  'sensor_msgs/msg/CameraInfo'),
                    (f"{ns}/image_right/camera_info", 'sensor_msgs/msg/CameraInfo'),
                    (f"{ns}/baseline", 'geometry_msgs/msg/PoseStamped')]):
                writer.create_topic(rosbag2_py.TopicMetadata(
                    id=tid, name=topic, type=ttype,
                    serialization_format='cdr'))
            self._bag_writer = writer
            self._bag_topic_left  = f"{ns}/image_left/compressed"
            self._bag_topic_right = f"{ns}/image_right/compressed"
            self._bag_topic_cinfo_l = f"{ns}/image_left/camera_info"
            self._bag_topic_cinfo_r = f"{ns}/image_right/camera_info"
            self._bag_topic_baseline = f"{ns}/baseline"
            self.node.get_logger().info(
                f"recording wire stream to bag: {record_bag} "
                f"(source-codec CompressedImage + CameraInfo + baseline)")

        # Zenoh
        if endpoints:
            joined = ",".join(f'"{e}"' for e in endpoints)
            cfg = zenoh.Config.from_json5(
                '{ mode: "peer", connect: { endpoints: [ ' + joined + ' ] } }')
        else:
            cfg = zenoh.Config()

        print(f"[zenoh] opening session (endpoints={endpoints or '[scouting]'})")
        self.zsession = zenoh.open(cfg)

        # One decode worker per camera, each fed by its own in-order FIFO.
        # The Zenoh callback can fire on multiple runtime threads; PyAV's
        # CodecContext is NOT thread-safe, so decoding directly in the
        # callback lets two threads corrupt one decoder's reference-frame
        # state (visible as corruption, worst when the main thread is idle,
        # i.e. --no-view). Funnelling each camera through a single worker
        # thread guarantees serialized, in-order decode.
        self._stop = False
        self._q_left  = queue.Queue(maxsize=120)
        self._q_right = queue.Queue(maxsize=120)
        self.sub_left  = self.zsession.declare_subscriber(
            "quest/image_left/compressed",
            lambda s: self._enqueue(s, self._q_left,  "L"))
        self.sub_right = self.zsession.declare_subscriber(
            "quest/image_right/compressed",
            lambda s: self._enqueue(s, self._q_right, "R"))
        # Calibration: the APK re-puts intrinsics + per-eye pose at 1 Hz on
        # this key. Parse the first valid one into CameraInfo + baseline; the
        # ROS timer then republishes. Only needed when we have a ROS node.
        if self.node is not None:
            self.sub_calibration = self.zsession.declare_subscriber(
                "quest/calibration", self._on_calibration)
        # --save-calib: dump the CameraInfo pair + baseline this receiver
        # already builds from QCAL to producer-format YAML files, once.
        self._save_calib_dir = (os.path.expanduser(save_calib)
                                if save_calib else None)
        self._baseline_frame = baseline_frame
        self._calib_saved = False
        self._workers = [
            threading.Thread(target=self._decode_loop,
                             args=(self._q_left,  self.stats_left,
                                   self.pub_left,  "L"),
                             name="decode-L", daemon=True),
            threading.Thread(target=self._decode_loop,
                             args=(self._q_right, self.stats_right,
                                   self.pub_right, "R"),
                             name="decode-R", daemon=True),
        ]
        for w in self._workers:
            w.start()

        if self.node is None:
            self._stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
            self._stats_thread.start()

        print("[receiver] ready.", "ROS ON." if self.node else "", "cv2 ON." if self.viewer else "")

    def _enqueue(self, sample, q: "queue.Queue", label: str):
        """Zenoh callback (may run on multiple threads). Copy the bytes out of
        the sample and hand them to this camera's worker, in order. No decode
        here — that must be single-threaded per camera."""
        recv_ns = time.monotonic_ns()
        payload = bytes(sample.payload)
        attachment = bytes(sample.attachment) if sample.attachment is not None else b""
        try:
            q.put_nowait((recv_ns, payload, attachment))
        except queue.Full:
            # Worker fell behind — drop oldest to bound latency. H.264 resyncs
            # at the next keyframe (producer sends SPS/PPS on every IDR).
            try: q.get_nowait()
            except queue.Empty: pass
            try: q.put_nowait((recv_ns, payload, attachment))
            except queue.Full: pass

    def _decode_loop(self, q: "queue.Queue", stats: CamStats,
                     publisher, label: str):
        """One per camera — the only thread that touches this camera's
        decoder (PyAV or NVDEC). Pulls frames in order and processes them."""
        transcoder = None
        if self.gpu_decode:
            # Pin the CUDA context to THIS worker thread, then build the
            # NVDEC+nvJPEG transcoder here so its decoder binds to this
            # thread's context.
            try:
                _ = torch.zeros(1, device='cuda')
            except Exception as e:
                print(f"[gpu-decode] CUDA init ({label}): {e}")
            transcoder = GpuTranscoder(label)

        while not self._stop:
            try:
                recv_ns, payload, attachment = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(recv_ns, payload, attachment,
                              stats, publisher, label, transcoder)
            except Exception as e:
                print(f"[worker {label}] {e}")

    def _process(self, recv_ns: int, payload: bytes, attachment: bytes,
                 stats: CamStats, publisher, label: str, transcoder):
        stamp_us = 0
        codec_id = CODEC_JPEG
        if len(attachment) >= ATTACHMENT_SIZE:
            stamp_us, _w, _h, codec_id, _flags, _ = struct.unpack(
                ATTACHMENT_FORMAT, attachment[:ATTACHMENT_SIZE])
        codec_name = CODEC_NAMES.get(codec_id, "jpeg")
        stats.update(recv_ns, stamp_us, len(payload), codec_name)

        if self._bag_writer is not None:
            self._bag_write_frame(label, payload, codec_name, stamp_us)

        out_bytes = None
        out_format = None
        raw_hwc = None        # HWC uint8 RGB host array, set in ros_format=="raw"

        if self.ros_format == "passthrough":
            # Thin relay: forward the encoded bytes UNCHANGED and tag the source
            # codec. No transcode — an H.264 wire stream reaches the rectifier as
            # H.264 for a single GPU (NVDEC) decode downstream. Decode here only
            # to feed the optional cv2 viewer.
            out_bytes = payload
            out_format = codec_name                       # "jpeg" or "h264"
            if self.viewer is not None:
                try:
                    if codec_id == CODEC_H264 and transcoder is not None:
                        rgb = transcoder.decode_rgb(payload)
                        if rgb is not None:
                            self.viewer.submit(
                                label, stamp_us, transcoder.rgb_to_bgr_hwc(rgb))
                    else:
                        for bgr in self.decoders.decode(label, codec_id, payload):
                            self.viewer.submit(label, stamp_us, bgr)
                except Exception as e:
                    print(f"[view] {label} {codec_name}: {e}")

        elif self.ros_format == "raw":
            # Decode once -> RGB HWC host array; publish as raw Image (no JPEG
            # re-encode, no rectifier JPEG decode). Same single H.264 decode as
            # the jpeg path; only the output stage differs.
            if codec_id == CODEC_H264 and transcoder is not None:
                try:
                    rgb = transcoder.decode_rgb(payload)      # RGB CHW CUDA
                except Exception as e:
                    print(f"[gpu-decode] {label}: {e}")
                    rgb = None
                if rgb is not None:
                    if self.viewer is not None:
                        self.viewer.submit(label, stamp_us,
                                           transcoder.rgb_to_bgr_hwc(rgb))
                    # CHW->HWC, single D2H to host rgb8.
                    raw_hwc = rgb.permute(1, 2, 0).contiguous().cpu().numpy()
            else:
                # CPU decode (PyAV H.264 or cv2 JPEG) -> BGR HWC; flip to RGB.
                try:
                    for bgr in self.decoders.decode(label, codec_id, payload):
                        if self.viewer is not None:
                            self.viewer.submit(label, stamp_us, bgr)
                        raw_hwc = np.ascontiguousarray(bgr[:, :, ::-1])
                except Exception as e:
                    print(f"[decode] {label} {codec_name}: {e}")

        if (publisher is not None and self.node is not None and rclpy.ok()
                and raw_hwc is not None):
            h, w = raw_hwc.shape[:2]
            im = Image()
            if self.use_producer_stamp and stamp_us > 0:
                im.header.stamp.sec = stamp_us // 1_000_000
                im.header.stamp.nanosec = (stamp_us % 1_000_000) * 1000
            else:
                im.header.stamp = self.node.get_clock().now().to_msg()
            im.header.frame_id = f"quest_cam_{label.lower()}"
            im.height = int(h)
            im.width = int(w)
            im.encoding = "rgb8"
            im.is_bigendian = 0
            im.step = int(w) * 3
            im.data = raw_hwc.tobytes()
            publisher.publish(im)

        elif (publisher is not None and self.node is not None
                and out_bytes is not None and out_format is not None
                and rclpy.ok()):
            msg = CompressedImage()
            if self.use_producer_stamp and stamp_us > 0:
                msg.header.stamp.sec = stamp_us // 1_000_000
                msg.header.stamp.nanosec = (stamp_us % 1_000_000) * 1000
            else:
                msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = f"quest_cam_{label.lower()}"
            msg.format = out_format
            msg.data = out_bytes
            publisher.publish(msg)

    def _bag_write_frame(self, label: str, payload: bytes,
                         codec_name: str, stamp_us: int):
        """Record the wire bytes as a source-codec CompressedImage. Same
        header-stamp policy as the live publishers; the bag receive-time is
        node-clock now (steady playback pacing). Thread-safe: the two decode
        workers and the caminfo timer share one writer under _bag_lock."""
        try:
            msg = CompressedImage()
            now = self.node.get_clock().now()
            if self.use_producer_stamp and stamp_us > 0:
                msg.header.stamp.sec = stamp_us // 1_000_000
                msg.header.stamp.nanosec = (stamp_us % 1_000_000) * 1000
            else:
                msg.header.stamp = now.to_msg()
            msg.header.frame_id = f"quest_cam_{label.lower()}"
            msg.format = codec_name
            msg.data = payload
            topic = (self._bag_topic_left if label == "L"
                     else self._bag_topic_right)
            with self._bag_lock:
                if self._bag_writer is not None:
                    self._bag_writer.write(
                        topic, self._bag_serialize(msg), now.nanoseconds)
        except Exception as e:
            print(f"[bag] frame write ({label}): {e}")

    def _bag_write_caminfo(self, ci_l, ci_r, ps):
        try:
            now = self.node.get_clock().now()
            t = now.nanoseconds
            with self._bag_lock:
                if self._bag_writer is None:
                    return
                self._bag_writer.write(
                    self._bag_topic_cinfo_l, self._bag_serialize(ci_l), t)
                self._bag_writer.write(
                    self._bag_topic_cinfo_r, self._bag_serialize(ci_r), t)
                self._bag_writer.write(
                    self._bag_topic_baseline, self._bag_serialize(ps), t)
        except Exception as e:
            print(f"[bag] caminfo write: {e}")

    @staticmethod
    def _caminfo_yaml(ci) -> str:
        """sensor_msgs/CameraInfo -> the producer's message-dump YAML format
        (binning_x / d / k / p / r / roi / frame_id — same keys, same order)."""
        def flist(vs, ind="- "):
            return "".join(f"{ind}{float(v)}\n" for v in vs)
        d = list(ci.d) if len(ci.d) else [0.0] * 5
        return (f"binning_x: {ci.binning_x}\n"
                f"binning_y: {ci.binning_y}\n"
                f"d:\n{flist(d)}"
                f"distortion_model: {ci.distortion_model or 'plumb_bob'}\n"
                f"frame_id: {ci.header.frame_id}\n"
                f"height: {ci.height}\n"
                f"k:\n{flist(ci.k)}"
                f"p:\n{flist(ci.p)}"
                f"r:\n{flist(ci.r)}"
                f"roi:\n"
                f"  do_rectify: {'true' if ci.roi.do_rectify else 'false'}\n"
                f"  height: {ci.roi.height}\n"
                f"  width: {ci.roi.width}\n"
                f"  x_offset: {ci.roi.x_offset}\n"
                f"  y_offset: {ci.roi.y_offset}\n"
                f"width: {ci.width}\n")

    def _write_calib_files(self, ci_l, ci_r, ps):
        try:
            os.makedirs(self._save_calib_dir, exist_ok=True)
            for ci, name in ((ci_l, "left"), (ci_r, "right")):
                path = os.path.join(self._save_calib_dir,
                                    f"camera_info_{name}.yaml")
                with open(path, "w") as f:
                    f.write(self._caminfo_yaml(ci))
                print(f"[calib] wrote {path}")
            # Baseline in the SAME format the consumer consumes:
            # HT_right_to_left.npy, 4x4 float64 (R from the pose quaternion,
            # t from the pose translation) — saved AS RECEIVED from QCAL,
            # no axis conversion.
            p = ps.pose.position; q = ps.pose.orientation
            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]],
                dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [p.x, p.y, p.z]
            path = os.path.join(self._save_calib_dir, "HT_right_to_left.npy")
            np.save(path, T)
            b = float(np.linalg.norm(T[:3, 3]))
            print(f"[calib] wrote {path} (baseline = {b*1000:.1f} mm, "
                  f"t = [{p.x:+.5f}, {p.y:+.5f}, {p.z:+.5f}] — as received "
                  f"from QCAL, no axis conversion)")
            self._calib_saved = True
        except Exception as e:
            print(f"[calib] save failed: {e}")

    def _on_calibration(self, sample):
        """Zenoh callback: parse a QCAL payload into CameraInfo + baseline.
        Takes the first valid calibration and ignores the rest (the
        rectifier/consumer only need it once)."""
        if self._have_calibration or self.node is None:
            return
        try:
            payload = bytes(sample.payload)
        except Exception:
            return
        if len(payload) < CAL_HEADER_SIZE:
            return
        magic, _ver, ncam, _ = struct.unpack(
            CAL_HEADER_FORMAT, payload[:CAL_HEADER_SIZE])
        if magic != CAL_MAGIC or ncam < 2:
            self.node.get_logger().warn(
                f"calibration: bad header (magic={magic!r} ncam={ncam})")
            return

        cams = {}
        off = CAL_HEADER_SIZE
        for _ in range(ncam):
            if off + CAL_CAM_SIZE > len(payload):
                break
            (cam_id, dist_count, w, h, _res,
             fx, fy, cx, cy,
             k1, k2, k3, p1, p2,
             tx, ty, tz,
             _qx, _qy, _qz, _qw) = struct.unpack(
                CAL_CAM_FORMAT, payload[off:off + CAL_CAM_SIZE])
            off += CAL_CAM_SIZE
            cams[cam_id] = dict(
                w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy,
                dist=[k1, k2, k3, p1, p2] if dist_count >= 5 else [0.0] * 5,
                t=(tx, ty, tz))

        if 50 not in cams or 51 not in cams:
            self.node.get_logger().warn(
                f"calibration: missing eye(s); got ids {sorted(cams)}")
            return

        left, right = cams[50], cams[51]
        ci_l = self._build_caminfo(left, self._cinfo_frame_left)
        ci_r = self._build_caminfo(right, self._cinfo_frame_right)

        dx = right["t"][0] - left["t"][0]
        dy = right["t"][1] - left["t"][1]
        dz = right["t"][2] - left["t"][2]
        baseline_m = (dx * dx + dy * dy + dz * dz) ** 0.5
        # Convert the QCAL (Android lens-pose) frame into the pipeline's eye
        # convention: the QCAL delta is X-dominant (+63.6 mm on x) while the
        # pipeline convention — the sim wire and the refined calibration
        # HT_right_to_left.npy — is Y-dominant. Measured on this rig:
        #   QCAL    (+0.06361, -0.00052, -0.00028)
        #   refined (+0.00002, -0.06264, +0.00003)
        # i.e. a 90° rotation about z: (x,y,z)_qcal -> (y, -x, z)_eye.
        # --baseline-frame qcal disables the conversion (raw pass-through).
        if self._baseline_frame == "eye":
            bx, by, bz = float(dy), float(-dx), float(dz)
        else:
            bx, by, bz = float(dx), float(dy), float(dz)
        ps = PoseStamped()
        ps.header.frame_id = self._cinfo_frame_left
        ps.pose.position.x = bx
        ps.pose.position.y = by
        ps.pose.position.z = bz
        # Rectifier uses ||t|| only and forces R = I, so identity orientation.
        ps.pose.orientation.x = 0.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 1.0

        with self._cal_lock:
            self._cinfo_left_msg = ci_l
            self._cinfo_right_msg = ci_r
            self._baseline_msg = ps
            self._have_calibration = True

        # ROS stereo convention: the right projection carries the baseline,
        # P_right[0,3] = -fx * B. Makes the CameraInfo pair self-contained
        # (this is also what bag_h264_source derives its baseline from —
        # the field being 0 was the '-0.0 mm' incident).
        ci_r.p[3] = -float(right["fx"]) * float(baseline_m)

        self.node.get_logger().info(
            f"calibration received: L fx={left['fx']:.1f} cx={left['cx']:.1f}, "
            f"R fx={right['fx']:.1f} cx={right['cx']:.1f}, "
            f"{left['w']}x{left['h']}, baseline={baseline_m * 1000:.1f} mm "
            f"({self._baseline_frame} frame: t=({bx:+.5f}, {by:+.5f}, "
            f"{bz:+.5f})) -> publishing CameraInfo + baseline")
        self._publish_caminfo()  # don't wait up to 1 s for the timer

    def _build_caminfo(self, cam, frame_id):
        ci = CameraInfo()
        ci.width = int(cam["w"])
        ci.height = int(cam["h"])
        ci.distortion_model = "plumb_bob"
        ci.d = [float(x) for x in cam["dist"]]
        fx, fy, cx, cy = (float(cam["fx"]), float(cam["fy"]),
                          float(cam["cx"]), float(cam["cy"]))
        ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        ci.header.frame_id = frame_id
        return ci

    def _publish_caminfo(self):
        if self.node is None or self.pub_cinfo_left is None:
            return
        with self._cal_lock:
            ci_l = self._cinfo_left_msg
            ci_r = self._cinfo_right_msg
            ps = self._baseline_msg
        if ci_l is None:
            return
        now = self.node.get_clock().now().to_msg()
        ci_l.header.stamp = now
        ci_r.header.stamp = now
        ps.header.stamp = now
        self.pub_cinfo_left.publish(ci_l)
        self.pub_cinfo_right.publish(ci_r)
        self.pub_baseline.publish(ps)
        if self._save_calib_dir and not self._calib_saved:
            self._write_calib_files(ci_l, ci_r, ps)
        if self._bag_writer is not None:
            self._bag_write_caminfo(ci_l, ci_r, ps)

    def _maybe_report(self):
        now = time.monotonic()
        if now - self._last_report < self._stats_period_s:
            return
        self._last_report = now
        for s in (self.stats_left, self.stats_right):
            line = s.report()
            if line:
                if self.node:
                    self.node.get_logger().info(line)
                else:
                    print(line)

    def _stats_loop(self):
        while not (self.viewer and self.viewer.stop_requested):
            self._maybe_report()
            time.sleep(self._stats_period_s)

    def shutdown(self):
        self._stop = True
        # Stop Zenoh first so no new frames are enqueued, then let the decode
        # workers drain/exit before we tear down rclpy (avoids "publisher's
        # context is invalid" on Ctrl-C).
        try: self.sub_left.undeclare()
        except Exception: pass
        try: self.sub_right.undeclare()
        except Exception: pass
        try:
            if self.sub_calibration is not None:
                self.sub_calibration.undeclare()
        except Exception: pass
        try: self.zsession.close()
        except Exception: pass
        for w in getattr(self, "_workers", []):
            try: w.join(timeout=1.0)
            except Exception: pass
        if self._bag_writer is not None:
            with self._bag_lock:
                try:
                    del self._bag_writer   # rosbag2_py closes on destruction
                except Exception:
                    pass
                self._bag_writer = None
            print("[bag] recording closed.")
        if self.node is not None:
            try: self.node.destroy_node()
            except Exception: pass
            try: rclpy.shutdown()
            except Exception: pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--namespace', default='/xr')
    p.add_argument('--endpoint', action='append', default=[])
    p.add_argument('--stats-period', type=float, default=2.0)
    p.add_argument('--use-producer-stamp', action='store_true')
    p.add_argument('--no-view', dest='view', action='store_false', default=True)
    p.add_argument('--no-ros', dest='ros', action='store_false', default=True)
    p.add_argument('--gpu-decode', action='store_true',
                   help='GPU H.264 decode (NVDEC) for the viewer / raw mode. '
                        'Needs torch + PyNvVideoCodec + CUDA. Default is CPU '
                        '(PyAV).')
    p.add_argument('--ros-format', choices=['passthrough', 'raw'],
                   default='passthrough',
                   help="'passthrough' (default): forward the source bytes "
                        "unchanged, format 'h264' — no transcode; the rectifier "
                        "does the single NVDEC decode downstream. 'raw': decode "
                        "once here and publish rgb8 sensor_msgs/Image on "
                        "{ns}/image_left|right (no /compressed) — feed a "
                        "rectifier running --raw-input.")
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--baseline-frame', choices=['eye', 'qcal'], default='eye',
                   help="Frame of the published /xr/baseline and saved npy. "
                        "'eye' (default): converted to the pipeline "
                        "convention (Y-dominant right-in-left, matching the "
                        "sim wire and HT_right_to_left.npy — drop-in for the "
                        "consumer). 'qcal': raw Android lens-pose delta "
                        "(X-dominant), as received.")
    p.add_argument('--save-calib', default=None, metavar='DIR',
                   help='Write the calibration this receiver builds from QCAL '
                        'to DIR as producer-format files, once: '
                        'camera_info_left.yaml, camera_info_right.yaml '
                        '(ROS camera_info YAML) + baseline.yaml.')
    p.add_argument('--record-bag', default=None, metavar='DIR',
                   help='Record the received WIRE stream to a rosbag2 (mcap) '
                        'at DIR: {ns}/image_left|right/compressed with the '
                        'source bytes and format (h264/jpeg, no transcode — '
                        'independent of --ros-format), plus CameraInfo x2 and '
                        '{ns}/baseline at the 1 Hz calibration cadence. The '
                        'bag replays into the rectifier H.264 path like the '
                        'live /xr topics. DIR must not already exist. '
                        'Requires ROS.')
    args = p.parse_args()

    if args.record_bag and not args.ros:
        p.error("--record-bag requires ROS (remove --no-ros).")

    if not args.ros and not args.view:
        p.error("Both --no-ros and --no-view set; nothing to do.")
    if args.view and av is None:
        print("[warn] PyAV not installed; H.264 frames won't display in cv2. "
              "JPEG still works. pip install av to fix.")

    viewer = CvViewer(scale=args.scale) if args.view else None
    receiver = QuestReceiver(
        namespace          = args.namespace,
        endpoints          = args.endpoint,
        stats_period_s     = args.stats_period,
        use_producer_stamp = args.use_producer_stamp,
        enable_ros         = args.ros,
        gpu_decode         = args.gpu_decode,
        viewer             = viewer,
        ros_format         = args.ros_format,
        record_bag         = args.record_bag,
        save_calib         = args.save_calib,
        baseline_frame     = args.baseline_frame,
    )

    try:
        if viewer is not None and receiver.node is not None:
            threading.Thread(target=rclpy.spin, args=(receiver.node,),
                             daemon=True).start()
            viewer.run()
        elif viewer is not None:
            viewer.run()
        else:
            rclpy.spin(receiver.node)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.stop_requested = True
        receiver.shutdown()


if __name__ == '__main__':
    main()