#!/usr/bin/env python3
"""
passthrough_consumer.py

Standalone IPCAI consumer for the passthrough_sim pipeline. It handles the
pipeline's ROS output directly:

    subscribes:  /left/image_rect, /right/image_rect   (rgb8, from the rectifier)
                 /left/camera_info_rect, /right/camera_info_rect
                 /xr/joint_states
                 /pose_init                            (FP seed, from the recorder)
    publishes:   /left/segmentation                    (mono8, seeds FoundationPose)
                 /handoff/estimated_pose   (optional, per-frame H_b2l PoseStamped)
                 + in-headset ghost overlay over a UNIX socket (optional)

Flow: load seg model -> fetch camera_info -> build the roboreg scene -> publish
segmentation while waiting for /pose_init -> from that seed, run the IPCAI
differentiable-rendering optimiser frame-by-frame.

This is a clean, slim driver. The proven ROS primitives (LiveSource, the
handoff seg-publisher, camera_info fetch, scene build) are copied in verbatim
from the original live/handoff pipeline; the IPCAI algorithm itself lives in
`stereo_ipcai_pipeline_bag` (srbag) and the renderer in `roboreg` --- those are
imported, not reimplemented, the same way the pipeline side depends on the
NVIDIA FoundationPose / ESS packages. Dropped vs. the original: the offline /
bag sources, video writing, frame dumping, on-screen display, and the full
timing-telemetry block.

Run (this is the T2 consumer in the passthrough_sim runbook):
    cd ~/roboreg   # so `roboreg`, `stereo_ipcai_pipeline_bag` are importable
    python3 ~/passthrough_sim/passthrough_consumer.py \
        --output-dir ~/runs/passthrough_$(date -u +%Y%m%dT%H%M%SZ) \
        --urdf-file ~/roboreg/test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf \
        --right-extrinsics-file <path>/HT_right_to_left.npy \
        --ipcai-lr 5e-3 --max-iterations 5 --pth 0.9 --send-overlay
"""

import argparse
import os
import tempfile
import time
from collections import deque
from threading import Lock, Thread, Event

import numpy as np
import yaml
import torch
import pytorch_kinematics as pk

from roboreg.core import (
    NVDiffRastRenderer, Robot, RobotScene, TorchKinematics,
    TorchMeshContainer, VirtualCamera,
)
from roboreg.io import (
    load_robot_data_from_ros_xacro, load_robot_data_from_urdf_file,
)

import stereo_ipcai_pipeline_bag as srbag
REGISTRATION_MODE = srbag.REGISTRATION_MODE


# =============================================================================
# Proven ROS primitives --- copied verbatim from stereo_pipeline_live_handoff.py
# (FrameSource / LiveSource / make_source / camera_info fetch / pose helpers /
#  HandoffSegPublisher / do_handoff_seg_and_wait / init_pipeline_resources).
# =============================================================================
def _decode_image_msg_to_numpy(msg):
    """Convert sensor_msgs/Image to (H, W, 3) uint8 RGB numpy."""
    return srbag._decode_image_msg_to_numpy(msg)


class FrameSource:
    """Yields {'left_img', 'right_img', 'joint_state', 'frame_idx'} dicts.

    Subclasses must override `_read_impl()` (NOT `read()`) and `close()`.
    `read()` wraps it to accumulate `read_decode_time_total_s` (cumulative time
    inside `_read_impl()`; frames / this = the raw read+decode rate) and
    `frames_delivered` (count of successful reads). Both feed the timing summary.
    """
    def __init__(self):
        self.read_decode_time_total_s = 0.0
        self.frames_delivered = 0

    def read(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            frame = self._read_impl(*args, **kwargs)
        finally:
            self.read_decode_time_total_s += time.perf_counter() - t0
        if frame is not None:
            self.frames_delivered += 1
        return frame

    def _read_impl(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        pass


class LiveSource(FrameSource):
    """Subscribe to live ROS2 stereo image + joint-state topics.

    Stereo images are paired via ApproximateTimeSynchronizer; joint state is
    subscribed separately (latest-wins) and attached to the most recent stereo
    pair when read. A background spin thread feeds a small queue.
    """
    def __init__(self, left_topic, right_topic, js_topic, device,
                 queue_size=2, slop_ms=50, compressed=False,
):
        super().__init__()
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import Image, CompressedImage, JointState
        import message_filters

        # Native-res optimiser path reads the producer's raw JPEG stream
        # (sensor_msgs/CompressedImage) directly; the rectified path reads
        # raw sensor_msgs/Image. srbag's decoder handles both.
        self._compressed = bool(compressed)
        self._img_msg_type = CompressedImage if self._compressed else Image
        # Compressed-path codec: 'jpeg' (nvJPEG, SMs) or 'h264' (NVDEC engine,
        # off the SMs). Per-eye NVDEC decoders are lazy (built on first frame).
        self._h264_dec_l = None
        self._h264_dec_r = None

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy

        self.device = device
        self.queue = deque(maxlen=queue_size)
        self.lock = Lock()
        self.new_frame = Event()
        self.stop_event = Event()
        self.frame_counter = 0
        self.latest_js = None

        self.node = Node('stereo_ipcai_live')
        # Dedicated executor avoids collision with concurrent rclpy.spin_once.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)

        # Dedicated CUDA stream for async H2D uploads from pinned memory,
        # so seg/opt kernels on other streams can overlap.
        self._upload_stream = (torch.cuda.Stream(device=device)
                               if str(device).startswith('cuda') else None)

        # Pinned-buffer pool (lazy alloc on first frame; round-robin pool
        # of 3 ensures wrap-around buffer is free of in-flight copies).
        self._pin_pool_size = 3
        self._pin_left = None
        self._pin_right = None
        self._pin_idx = 0

        sub_l = message_filters.Subscriber(self.node, self._img_msg_type, left_topic)
        sub_r = message_filters.Subscriber(self.node, self._img_msg_type, right_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_l, sub_r], queue_size=10, slop=slop_ms / 1000.0)
        self.sync.registerCallback(self._cb_stereo)
        self.js_sub = self.node.create_subscription(
            JointState, js_topic, self._cb_js, 10)

        self.spin_thread = Thread(target=self._spin, daemon=True)
        self.spin_thread.start()
        print(f"[Live] Subscribed: left='{left_topic}', right='{right_topic}', "
              f"js='{js_topic}', slop={slop_ms}ms")

    def _spin(self):
        while not self.stop_event.is_set() and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _cb_js(self, msg):
        if self._upload_stream is not None:
            with torch.cuda.stream(self._upload_stream):
                js = (torch.from_numpy(np.asarray(msg.position, dtype=np.float32))
                      .to(self.device, non_blocking=True).unsqueeze(0))
            self._upload_stream.synchronize()
        else:
            js = (torch.from_numpy(np.asarray(msg.position, dtype=np.float32))
                  .to(self.device).unsqueeze(0))
        with self.lock:
            self.latest_js = js

    def _cb_stereo(self, msg_l, msg_r):
        if self._compressed:
            # Decode EVERY access unit here, in arrival order (the single-
            # threaded executor guarantees ordering). H.264 streams from the
            # Quest/producer carry GOPs (P-frames referencing prior frames), so
            # a decode-on-grab design under latest_only would skip AUs and
            # break every reference chain -> full-frame mosaic. Decoding all
            # keeps NVDEC's reference state valid; latest_only then selects
            # among DECODED frames. Cost: NVDEC is a dedicated engine (off the
            # SMs) and the small convert runs on the legacy default stream,
            # which does NOT implicitly sync with srbag's non-blocking seg/opt
            # streams — the nvJPEG-era hazard that motivated deferring the
            # decode left with the jpeg path.
            frame = self._decode_h264(
                {'raw_l': bytes(msg_l.data), 'raw_r': bytes(msg_r.data)})
            if frame is None:
                return                      # NVDEC warmup (pre-IDR)
        else:
            left_img = _decode_image_msg_to_numpy(msg_l)
            right_img = _decode_image_msg_to_numpy(msg_r)

            if self._upload_stream is not None:
                # Lazy-allocate pinned buffers on the first frame.
                if self._pin_left is None:
                    shape = left_img.shape  # (H, W, C), uint8
                    self._pin_left = [torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                                      for _ in range(self._pin_pool_size)]
                    self._pin_right = [torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                                       for _ in range(self._pin_pool_size)]

                i = self._pin_idx
                self._pin_idx = (self._pin_idx + 1) % self._pin_pool_size
                self._pin_left[i].copy_(torch.from_numpy(left_img))
                self._pin_right[i].copy_(torch.from_numpy(right_img))

                # H2D on the dedicated upload stream (CPU decode above is host-
                # side, so no default-stream GPU op here); sync only this stream.
                with torch.cuda.stream(self._upload_stream):
                    left_t = (self._pin_left[i].to(self.device, non_blocking=True)
                              .float().div_(255.0).permute(2, 0, 1).contiguous())
                    right_t = (self._pin_right[i].to(self.device, non_blocking=True)
                               .float().div_(255.0).permute(2, 0, 1).contiguous())
                self._upload_stream.synchronize()
            else:
                left_t = (torch.from_numpy(left_img).to(self.device)
                          .float().div_(255.0).permute(2, 0, 1))
                right_t = (torch.from_numpy(right_img).to(self.device)
                           .float().div_(255.0).permute(2, 0, 1))
            frame = {'left_img': left_t, 'right_img': right_t}

        with self.lock:
            if self.latest_js is None:
                return  # wait for first joint-state before accepting frames
            frame.update({
                'joint_state': self.latest_js,
                'frame_idx': self.frame_counter,
                # Original capture-side wall-clock (µs) from the image header,
                # propagated to the overlay so the layer measures true
                # downstream-loop latency anchored to (near-)capture time.
                'stamp_us': (msg_l.header.stamp.sec * 1_000_000
                             + msg_l.header.stamp.nanosec // 1000),
                # Consumer-side receive time (system clock). grab - this = queue
                # depth; this - stamp_us = wire transport.
                'recv_us': time.time() * 1e6})
            self.queue.append(frame)
            self.frame_counter += 1
        self.new_frame.set()

    def _decode_h264(self, frame):
        """NVDEC-decode both eyes (persistent per-eye streams), returning RGB
        float in [0,1] like the JPEG path. NVDEC is a dedicated engine, so the
        decode runs off the SMs and doesn't contend with seg/opt — only the
        light NV12->RGB convert touches the SMs. Returns None until each eye's
        NVDEC locks on its first IDR. Streams may carry GOPs (Quest bags
        measured at ~30): the caller feeds every AU in order (see _cb_stereo),
        so reference chains stay valid regardless of GOP structure."""
        if self._h264_dec_l is None:
            from gpu_h264_codec import GpuH264Decoder
            dev = str(self.device)
            self._h264_dec_l = GpuH264Decoder(device=dev)
            self._h264_dec_r = GpuH264Decoder(device=dev)
        l_u8 = self._h264_dec_l.decode(frame.pop('raw_l'))   # (3,H,W) uint8 cuda rgb | None
        r_u8 = self._h264_dec_r.decode(frame.pop('raw_r'))
        if l_u8 is None or r_u8 is None:
            return None                                       # NVDEC warmup (pre-IDR)
        frame['left_img'] = l_u8.float().div_(255.0)
        frame['right_img'] = r_u8.float().div_(255.0)
        # Materialise the frame before the optimiser reads it, WITHOUT a
        # device-wide barrier. NVDEC runs on cudastream=0 and the RGBP clone +
        # float/div above are on the default stream, so syncing only the
        # default stream guarantees the image is fully written. Unlike the
        # nvJPEG path (own internal stream → needs a global sync), a device-wide
        # synchronize here would block on the in-flight opt(current)‖seg(next)
        # dual-stream work every frame and SERIALISE the pipeline. current_stream
        # waits on the decode only, leaving the opt/seg streams free to overlap.
        torch.cuda.current_stream(self.device).synchronize()
        return frame

    def _read_impl(self, timeout_sec=10.0, latest_only=False):
        """Pop a frame from the queue, blocking up to `timeout_sec`.

        latest_only=True: pop the NEWEST frame and discard older ones in the
        queue. Best for low-latency live tracking — always work on the freshest
        observation. latest_only=False (default): pop the oldest frame
        (FIFO) — best when you want to process every frame the camera publishes.
        """
        deadline = time.perf_counter() + timeout_sec
        frame = None
        while True:
            with self.lock:
                if self.queue:
                    if latest_only:
                        # Drain everything but the newest.
                        while len(self.queue) > 1:
                            self.queue.popleft()
                    frame = self.queue.popleft()
            if frame is not None:
                # Queue holds fully-decoded frames (both paths): the compressed
                # path decodes in _cb_stereo so every AU reaches NVDEC in order.
                return frame
            self.new_frame.clear()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            self.new_frame.wait(timeout=remaining)

    def close(self):
        self.stop_event.set()
        try:
            self.spin_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._executor.remove_node(self.node)
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass


def make_source(args, device, left_topic=None, right_topic=None,
                compressed=None):
    left_topic = left_topic if left_topic is not None else args.left_topic
    right_topic = right_topic if right_topic is not None else args.right_topic
    if compressed is None:
        compressed = (args.optimiser_image_transport == "compressed")
    return LiveSource(
        left_topic, right_topic, args.joint_state_topic,
        device, slop_ms=args.sync_slop_ms, compressed=compressed)


def _make_pose_stamped(H, node, frame_id="left_camera"):
    """Build a geometry_msgs/PoseStamped from a 4x4 SE(3) matrix H.
    Rotation extracted via a standard matrix→quaternion conversion."""
    from geometry_msgs.msg import PoseStamped
    import numpy as _np
    ps = PoseStamped()
    ps.header.stamp = node.get_clock().now().to_msg()
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(H[0, 3])
    ps.pose.position.y = float(H[1, 3])
    ps.pose.position.z = float(H[2, 3])
    R = H[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = _np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = _np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = _np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = _np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    ps.pose.orientation.w = float(qw)
    ps.pose.orientation.x = float(qx)
    ps.pose.orientation.y = float(qy)
    ps.pose.orientation.z = float(qz)
    return ps


# =============================================================================
# Camera-info fetching (from live topics or from a bag)
# =============================================================================
def fetch_camera_info_from_topics(args, timeout_sec=10.0,
                                  left_topic=None, right_topic=None):
    """Wait for one message on each CameraInfo topic; return (left, right).

    Defaults to the downsampled (rectified) topics on args; pass left_topic /
    right_topic to fetch a different pair (e.g. the producer's native-res
    camera_info for the optimiser scene).

    Uses a dedicated executor to avoid colliding with concurrent spinners.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import CameraInfo

    left_topic = left_topic or args.left_camera_info_topic
    right_topic = right_topic or args.right_camera_info_topic

    if not rclpy.ok():
        rclpy.init()
    node = Node('stereo_ipcai_caminfo_oneshot')
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    received = {'left': None, 'right': None}
    def _mk_cb(side):
        def _cb(msg):
            if received[side] is None:
                received[side] = msg
        return _cb
    sub_l = node.create_subscription(
        CameraInfo, left_topic, _mk_cb('left'), 10)
    sub_r = node.create_subscription(
        CameraInfo, right_topic, _mk_cb('right'), 10)

    print(f"[CamInfo] Waiting for one msg each on '{left_topic}', "
          f"'{right_topic}' (timeout {timeout_sec}s)...")
    deadline = time.perf_counter() + timeout_sec
    try:
        while received['left'] is None or received['right'] is None:
            if time.perf_counter() > deadline:
                missing = [s for s, m in received.items() if m is None]
                raise RuntimeError(f"[CamInfo] Timeout waiting for: {missing}")
            executor.spin_once(timeout_sec=0.1)
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_subscription(sub_l)
        node.destroy_subscription(sub_r)
        node.destroy_node()

    H, W = int(received['left'].height), int(received['left'].width)
    print(f"[CamInfo] Got {W}x{H}")
    return received['left'], received['right']

def _pose_msg_to_4x4(msg):
    """geometry_msgs/PoseStamped -> 4x4 numpy."""
    p = msg.pose.position
    q = msg.pose.orientation
    n = (q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w) ** 0.5
    if n < 1e-9:
        raise ValueError("Zero quaternion in PoseStamped")
    qx, qy, qz, qw = q.x/n, q.y/n, q.z/n, q.w/n
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T


# =============================================================================
# Handoff seg-publisher: subscribes to image_topic, runs srbag.extract_mask,
# publishes mono8 mask. Active until stop() is called. The masks feed FP's
# selector node so FP can fire its initial registration.
# =============================================================================
class HandoffSegPublisher:
    """rclpy node: subscribes to image_topic, runs the same seg model
    IPCAI's tracking phase will use, publishes mono8 masks on seg_topic."""

    def __init__(self, image_topic, seg_topic, device, pth=0.5,
                 debug_dir=""):
        import rclpy
        from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                               QoSDurabilityPolicy, QoSHistoryPolicy)
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node('ipcai_handoff_seg_publisher')
        self._device = device
        self._enabled = True
        self._stream = (torch.cuda.Stream(device=device)
                        if str(device).startswith('cuda') else None)
        self._pth = float(pth)
        self.frames_seen = 0
        self.frames_published = 0

        self._debug_dir = None
        if debug_dir:
            from pathlib import Path as _P
            self._debug_dir = _P(debug_dir)
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            for f in self._debug_dir.glob("frame_*.png"):
                try: f.unlink()
                except OSError: pass
            self._node.get_logger().info(
                f"[Handoff] seg-debug: writing image+mask PNGs to "
                f"{self._debug_dir}")

        # depth=1 (latest-only): always segment the freshest image_rect so the
        # seg's copied stamp stays aligned with the current depth/camera_info_rect.
        # depth>1 lets a backlog build when the bridge runs unthrottled, making the
        # seg lag past FP's 100ms sync window so the Selector never pairs all four.
        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)
        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5)
        self._pub = self._node.create_publisher(Image, seg_topic, pub_qos)
        self._sub = self._node.create_subscription(
            Image, image_topic, self._on_image, sub_qos)
        self._node.get_logger().info(
            f"[Handoff] seg-publisher: {image_topic} -> {seg_topic}")

    def node(self):
        return self._node

    def stop(self):
        self._enabled = False

    def _on_image(self, msg):
        self.frames_seen += 1
        if not self._enabled:
            return
        try:
            from sensor_msgs.msg import Image
            img = _decode_image_msg_to_numpy(msg)  # (H, W, 3) uint8 RGB
            H, W, _ = img.shape

            t = (torch.from_numpy(img).to(self._device)
                 .float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
                 .contiguous())
            # Seg CUDA graph was captured at batch=2 (stereo); duplicate
            # the single image into both slots and keep only mask[0].
            t_batch2 = torch.cat([t, t], dim=0).contiguous()
            mask = srbag.extract_mask(t_batch2, self._stream)
            if self._stream is not None:
                self._stream.synchronize()
            # extract_mask returns soft probs; binarise so FP gets {0, 255}.
            mask_np = ((mask[0, 0] > self._pth)
                       .to(torch.uint8).mul_(255)
                       .cpu().numpy())

            out = Image()
            out.header = msg.header
            out.height = int(H)
            out.width = int(W)
            out.encoding = 'mono8'
            out.is_bigendian = 0
            out.step = int(W)
            out.data = mask_np.tobytes()
            self._pub.publish(out)
            self.frames_published += 1

            if self._debug_dir is not None:
                idx = self.frames_published
                fg = int((mask_np > 0).sum())
                self._node.get_logger().info(
                    f"[seg-debug] frame {idx}: mask shape={mask_np.shape} "
                    f"dtype={mask_np.dtype} fg_pixels={fg}/"
                    f"{mask_np.shape[0]*mask_np.shape[1]}")
                try:
                    import cv2
                    bgr = img[:, :, ::-1]
                    img_path = str(self._debug_dir / f"frame_{idx:06d}_image.png")
                    mask_path = str(self._debug_dir / f"frame_{idx:06d}_mask.png")
                    ok_img = cv2.imwrite(img_path, bgr)
                    ok_mask = cv2.imwrite(mask_path, mask_np)
                    np.save(str(self._debug_dir / f"frame_{idx:06d}_mask.npy"),
                            mask_np)
                    if not ok_img or not ok_mask:
                        self._node.get_logger().warn(
                            f"[seg-debug] cv2.imwrite returned "
                            f"img={ok_img} mask={ok_mask}")
                except Exception as e:
                    self._node.get_logger().warn(
                        f"[Handoff] seg-debug write failed: {e}")
        except Exception as e:
            self._node.get_logger().error(
                f"[Handoff] seg-publish failed: {e}")


class ReInitManager:
    """Zero-idle-cost re-initialisation listener for the all-cold seed route.

    Watches /pose_init for a message with a stamp NEWER than the baseline
    (the original handoff's latched init) and /seed/arm (std_msgs/Bool,
    latched). On arm=true it resumes /left/segmentation publishing (its own
    HandoffSegPublisher instance — same model class the handoff used) so the
    warm resident ESS+FP stack can produce a fresh init; when the fresh
    /pose_init lands, seg publishing stops and the new 4x4 is handed to the
    tracking loop for a full DR reset. Between re-inits this costs one idle
    subscription pair — no compute, no VRAM.

    Workflow (warm seed, seed_warm:=true): press 'r'+Enter in the consumer
    terminal (or publish /seed/arm true). The resident ESS+FP stack computes
    behind the opened gate, the persistent recorder publishes a fresh
    /pose_init, the consumer resets its DR state and self-disarms; the stack
    idles again (VRAM-only residency).
    """

    def __init__(self, args, device):
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                               QoSDurabilityPolicy, QoSHistoryPolicy)
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Bool

        self._args, self._device = args, device
        self._lock = Lock()
        self._baseline_ns = None          # stamp of the accepted original init
        self._fresh = None                # (H 4x4, stamp_ns) awaiting pickup
        self._armed = False
        self._seg = None                  # live HandoffSegPublisher when armed
        self._stop = Event()

        self._node = Node('ipcai_reinit_manager')
        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self._node.create_subscription(
            PoseStamped, args.pose_init_topic, self._on_init, latched)
        self._node.create_subscription(Bool, '/seed/arm', self._on_arm, latched)
        self._frozen = False
        self._node.create_subscription(
            Bool, '/seed/frozen',
            lambda m: setattr(self, '_frozen', bool(m.data)), latched)
        self._arm_pub = self._node.create_publisher(Bool, '/seed/arm', latched)
        # Clear any STALE latched arm from a previous session/attempt: the
        # latch persists across restarts, and a stale 'true' would re-freeze
        # the bag source the moment it starts. A fresh session begins disarmed.
        _clear = Bool(); _clear.data = False
        self._arm_pub.publish(_clear)

        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while not self._stop.is_set():
            self._exec.spin_once(timeout_sec=0.2)
            # Deferred seg start: pumping seg before the recorder is wired
            # drops FP's first poses and can wedge its time-sync — the same
            # race the original handoff gates on. Start seg only once the
            # recorder's /pose_init publisher AND its matrix subscription
            # exist.
            if getattr(self, '_seg_pending', False) and self._armed:
                pubs = self._node.count_publishers(self._args.pose_init_topic)
                subs = self._node.count_subscribers(self._args.matrix_topic)
                if pubs > 0 and subs > 0:
                    self._seg_pending = False
                    print("[ReInit] seed stack ready — resuming "
                          "/left/segmentation")
                    self._seg = HandoffSegPublisher(
                        self._args.seg_image_topic,
                        self._args.seg_publish_topic,
                        self._device, pth=self._args.pth,
                        debug_dir=getattr(self._args, 'seg_debug_dir', ''))
                    self._exec.add_node(self._seg.node())

    @staticmethod
    def _stamp_ns(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def set_baseline(self):
        """Call right after the original handoff: whatever init is currently
        latched becomes the baseline; only strictly newer stamps re-init."""
        with self._lock:
            if self._fresh is not None:          # the latched original
                self._baseline_ns = self._fresh[1]
                self._fresh = None
            elif self._baseline_ns is None:
                self._baseline_ns = 0

    def _on_init(self, msg):
        H = np.eye(4, dtype=np.float64)
        q = msg.pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        H[:3, :3] = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])
        H[:3, 3] = [msg.pose.position.x, msg.pose.position.y,
                    msg.pose.position.z]
        ns = self._stamp_ns(msg)
        with self._lock:
            if self._baseline_ns is not None and ns <= self._baseline_ns:
                return                             # the latched original replayed
            self._fresh = (H, ns)

    def _on_arm(self, msg):
        want = bool(msg.data)
        if want and not self._armed:
            self._armed = True
            self._seg_pending = True
            print("[ReInit] ARMED: waiting for the seed stack (recorder "
                  "readiness) before resuming /left/segmentation...")
        elif not want and self._armed:
            self._armed = False
            self._seg_pending = False
            if self._seg is not None:
                try:
                    self._exec.remove_node(self._seg.node())
                    self._seg.stop()
                except Exception:
                    pass
                self._seg = None
            print("[ReInit] disarmed: seg publishing stopped.")

    def is_armed(self):
        return self._armed

    def abort(self):
        """Give up on the current re-init: disarm (stops seg; the source
        aborts its re-freeze on the same signal) and resume tracking."""
        from std_msgs.msg import Bool
        m = Bool(); m.data = False
        self._arm_pub.publish(m)

    def status(self):
        segf = self._seg.frames_published if self._seg is not None else -1
        pubs = self._node.count_publishers(self._args.pose_init_topic)
        return (f"armed={self._armed} frozen={self._frozen} "
                f"seg={'pending' if getattr(self, '_seg_pending', False) else segf} "
                f"recorder_pubs={pubs}")

    def hold(self):
        """Estimation must pause: armed (re-init intent) OR the source's
        stream is actually frozen — whichever signal arrives first."""
        return self._armed or self._frozen

    def poll(self):
        """Called once per tracking-loop iteration. Returns a fresh 4x4
        camera-in-base init exactly once when a re-init lands, else None."""
        with self._lock:
            if self._fresh is None or self._baseline_ns is None:
                return None
            H, ns = self._fresh
            self._fresh = None
            self._baseline_ns = ns
        if self._armed:
            from std_msgs.msg import Bool
            m = Bool(); m.data = False
            self._arm_pub.publish(m)               # self-disarm -> seg stops
        t = H[:3, 3]
        print(f"[ReInit] fresh /pose_init t=[{t[0]:+.3f}, {t[1]:+.3f}, "
              f"{t[2]:+.3f}] m -> resetting DR to original state")
        return H

    def close(self):
        self._stop.set()


def do_handoff_seg_and_wait(args, device):
    """Publish /left/segmentation while subscribing to /pose_init. When
    the recorder publishes /pose_init, return the received 4x4 (already
    in IPCAI custom frame). Stops seg-publishing before returning.

    The seg model must already be loaded by the caller (run_pipeline
    does this before scene build, so the load cost is paid before the
    bag streams)."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                           QoSDurabilityPolicy, QoSHistoryPolicy)
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped

    if not rclpy.ok():
        rclpy.init()

    pose_node = Node('ipcai_handoff_pose_init')
    pose_received = {'msg': None}

    def _pose_cb(msg):
        if pose_received['msg'] is None:
            pose_received['msg'] = msg

    pose_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1)
    pose_sub = pose_node.create_subscription(
        PoseStamped, args.pose_init_topic, _pose_cb, pose_qos)

    seg_pub = HandoffSegPublisher(
        args.seg_image_topic, args.seg_publish_topic, device,
        pth=args.pth, debug_dir=args.seg_debug_dir)

    executor = SingleThreadedExecutor()
    executor.add_node(seg_pub.node())
    executor.add_node(pose_node)

    # Wait until the recorder is fully wired before we start pumping seg
    # masks. Without this, on fast machines the handoff process gets to
    # this point before the recorder has finished bringing up its
    # /pose_init publisher (and its subscription to FP's pose output).
    # The seg pub then starts publishing, FP starts firing, but the
    # recorder is not yet subscribed to FP's output, so the first poses
    # are dropped. With message_filters' time sync FP may not produce
    # any further usable output. With --display-progress the slow
    # cv2.namedWindow call masked this race by burning 1–2 s of startup
    # time; without it, the race is exposed and /pose_init never
    # arrives.
    print(f"[Handoff] Waiting for recorder readiness (max "
          f"{args.handoff_ready_timeout:.1f}s): "
          f"pub-count({args.pose_init_topic})>0 AND "
          f"sub-count({args.matrix_topic})>0 ...")
    _ready_deadline = time.perf_counter() + args.handoff_ready_timeout
    _ready_last_log = time.perf_counter()
    while True:
        executor.spin_once(timeout_sec=0.1)
        pub_count = pose_node.count_publishers(args.pose_init_topic)
        sub_count = pose_node.count_subscribers(args.matrix_topic)
        if pub_count > 0 and sub_count > 0:
            print(f"[Handoff] Recorder is ready "
                  f"(pose_init pubs={pub_count}, matrix subs={sub_count}). "
                  f"Starting seg publisher.")
            break
        if time.perf_counter() > _ready_deadline:
            print(f"[Handoff] WARNING: recorder readiness timed out "
                  f"after {args.handoff_ready_timeout:.1f}s "
                  f"(pose_init pubs={pub_count}, matrix subs={sub_count}). "
                  f"Proceeding anyway — /pose_init may never arrive.")
            break
        if time.perf_counter() - _ready_last_log > 2.0:
            print(f"[Handoff]   still waiting: pose_init pubs={pub_count}, "
                  f"matrix subs={sub_count}")
            _ready_last_log = time.perf_counter()

    print(f"[Handoff] Publishing seg + waiting for /pose_init "
          f"(timeout {args.pose_init_timeout:.1f}s)...")
    deadline = time.perf_counter() + args.pose_init_timeout
    last_log = time.perf_counter()
    try:
        while pose_received['msg'] is None:
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"[Handoff] Timeout on '{args.pose_init_topic}' "
                    f"({args.pose_init_timeout:.1f}s)")
            executor.spin_once(timeout_sec=0.1)
            if time.perf_counter() - last_log > 2.0:
                print(f"[Handoff] seg seen={seg_pub.frames_seen} "
                      f"published={seg_pub.frames_published}")
                last_log = time.perf_counter()
    finally:
        seg_pub.stop()
        executor.remove_node(seg_pub.node())
        executor.remove_node(pose_node)
        try:
            seg_pub.node().destroy_subscription(seg_pub._sub)
            seg_pub.node().destroy_publisher(seg_pub._pub)
            seg_pub.node().destroy_node()
        except Exception:
            pass
        try:
            pose_node.destroy_subscription(pose_sub)
            pose_node.destroy_node()
        except Exception:
            pass
        executor.shutdown()

    print(f"[Handoff] /pose_init received. Final seg: "
          f"seen={seg_pub.frames_seen} published={seg_pub.frames_published}")

    msg = pose_received['msg']
    T = _pose_msg_to_4x4(msg)
    p = msg.pose.position
    print(f"[Handoff] Init pose (IPCAI convention): "
          f"t=[{p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f}] m")
    return T


def _caminfo_msg_to_temp_yaml(msg, side):
    """Dump a sensor_msgs/CameraInfo to a temp YAML in camera_info_manager format."""
    fd, path = tempfile.mkstemp(prefix=f'caminfo_{side}_', suffix='.yaml')
    os.close(fd)
    with open(path, 'w') as f:
        yaml.safe_dump({
            'image_width': int(msg.width),
            'image_height': int(msg.height),
            'width': int(msg.width),
            'height': int(msg.height),
            'k': [float(v) for v in msg.k],
            'camera_matrix': {'rows': 3, 'cols': 3, 'data': [float(v) for v in msg.k]},
            'distortion_model': str(msg.distortion_model) if msg.distortion_model else 'plumb_bob',
            'distortion_coefficients': {'rows': 1, 'cols': len(msg.d),
                                        'data': [float(v) for v in msg.d]},
            'rectification_matrix': {'rows': 3, 'cols': 3, 'data': [float(v) for v in msg.r]},
            'projection_matrix': {'rows': 3, 'cols': 4, 'data': [float(v) for v in msg.p]},
        }, f)
    return path


def init_pipeline_resources(args, device, caminfo_msgs):
    """Build cameras + scene + seg/CUDA-graph state from CameraInfo messages."""
    left_msg, right_msg = caminfo_msgs
    H, W = int(left_msg.height), int(left_msg.width)

    # VirtualCamera.from_camera_configs needs a YAML path; materialise temp.
    left_yaml = _caminfo_msg_to_temp_yaml(left_msg, 'left')
    right_yaml = _caminfo_msg_to_temp_yaml(right_msg, 'right')
    try:
        cameras = {
            "left": VirtualCamera.from_camera_configs(
                camera_info_file=left_yaml,
                device=device,
            ),
            "right": VirtualCamera.from_camera_configs(
                camera_info_file=right_yaml,
                extrinsics_file=args.right_extrinsics_file,
                device=device,
            ),
        }
    finally:
        for p in (left_yaml, right_yaml):
            try:
                os.remove(p)
            except OSError:
                pass

    if args.urdf_file:
        robot_data = load_robot_data_from_urdf_file(
            urdf_path=args.urdf_file,
            root_link_name=args.root_link_name,
            end_link_name=args.end_link_name,
            collision=args.collision_meshes,
        )
    else:
        robot_data = load_robot_data_from_ros_xacro(
            ros_package=args.ros_package,
            xacro_path=args.xacro_path,
            root_link_name=args.root_link_name,
            end_link_name=args.end_link_name,
            collision=args.collision_meshes,
        )

    mesh_container = TorchMeshContainer(
        meshes=robot_data.meshes,
        batch_size=1,
        device=device,
    )
    kinematics = TorchKinematics(
        urdf=robot_data.urdf,
        root_link_name=robot_data.root_link_name,
        end_link_name=robot_data.end_link_name,
        device=device,
    )
    robot = Robot(mesh_container=mesh_container, kinematics=kinematics)
    scene = RobotScene(
        cameras=cameras,
        robot=robot,
        renderer=NVDiffRastRenderer(device=device),
    )

    # Idempotent — skip if handoff has already initialised these.
    if srbag.coarse_model is None or srbag.seg_graph is None:
        srbag.init_seg_model_and_graph(args, device)
        srbag.init_processing_state(args, device)

    K_left = None
    k_data = list(left_msg.k) if left_msg.k is not None else None
    if k_data is not None and len(k_data) == 9:
        K_left = np.array(k_data, dtype=np.float64).reshape(3, 3)

    K_right = None
    kr_data = list(right_msg.k) if right_msg.k is not None else None
    if kr_data is not None and len(kr_data) == 9:
        K_right = np.array(kr_data, dtype=np.float64).reshape(3, 3)

    return {'img_diagonal': float(np.sqrt(H**2 + W**2)),
            'scene': scene, 'H': H, 'W': W, 'K_left': K_left, 'K_right': K_right}


# =============================================================================
# Args --- passthrough-tuned. The visible flags are the knobs that matter for a
# passthrough run; the fixed block below supplies the remaining attributes that
# srbag.process_one_frame_dual_stream reads, with viz/video/save all disabled.
# =============================================================================
def build_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Standalone IPCAI consumer for the passthrough_sim pipeline.")

    # Topics. The OPTIMISER reads the producer's native-res JPEG stream so the
    # differentiable render + mask run at full resolution; the seg downsample to
    # 576x960 happens inside srbag.extract_mask. The handoff seg (FP-facing) and
    # ESS/FP stay on the downsampled rectified topics (see --seg-image-topic).
    p.add_argument("--left-topic", default="/xr/image_left/compressed")
    p.add_argument("--right-topic", default="/xr/image_right/compressed")
    p.add_argument("--optimiser-image-transport", choices=["compressed", "raw"],
                   default="compressed",
                   help="'compressed' = native sensor_msgs/CompressedImage (JPEG, "
                        "GPU-decoded); 'raw' = sensor_msgs/Image. Use 'raw' with "
                        "--left-topic /left/image_rect for the old downsampled path.")
    p.add_argument("--left-camera-info-topic", default="/left/camera_info_rect")
    p.add_argument("--right-camera-info-topic", default="/right/camera_info_rect")
    # Optimiser runs at NATIVE resolution: its VirtualCamera/scene is built from
    # the producer's raw-res camera_info (CloudXR publishes it directly), while
    # the downsampled camera_info_rect above stays for the seg / ESS-FP path.
    p.add_argument("--opt-left-camera-info-topic", default="/xr/image_left/camera_info",
                   help="Native-res left camera_info for the optimiser scene "
                        "(VirtualCamera + K). Default: the producer's raw-res "
                        "camera_info topic.")
    p.add_argument("--opt-right-camera-info-topic", default="/xr/image_right/camera_info",
                   help="Native-res right camera_info for the optimiser scene.")
    # Optimiser resolution mode. 'native' (default): scene + render at the
    # producer's raw resolution, image from --left-topic/--right-topic (JPEG).
    # 'downsampled': the previous mode — scene from camera_info_rect and image
    # from the rectified --rect-left-topic/--rect-right-topic (raw Image), so
    # the whole optimiser runs at 960x576. Lower fidelity, less compute.
    p.add_argument("--optimiser-res", choices=["native", "downsampled"],
                   default="native",
                   help="'native': optimiser scene/render at the producer's raw "
                        "resolution (default). 'downsampled': previous mode — "
                        "scene from camera_info_rect, image from the rectified "
                        "image topics, optimiser runs at 960x576.")
    p.add_argument("--rect-left-topic", default="/left/image_rect",
                   help="Rectified left image topic (raw Image) used as the "
                        "optimiser input in --optimiser-res downsampled.")
    p.add_argument("--rect-right-topic", default="/right/image_rect",
                   help="Rectified right image topic for downsampled mode.")
    p.add_argument("--joint-state-topic", default="/xr/joint_states")
    p.add_argument("--sync-slop-ms", type=float, default=50.0)

    # Init: handoff (default) waits for /pose_init; or load a fixed .npy seed.
    p.add_argument("--extrinsics-file", default=None,
                   help="Use this 4x4 .npy (IPCAI frame) as the init pose "
                        "instead of waiting for /pose_init (disables handoff).")
    p.add_argument("--reinit-timeout", type=float, default=90.0,
                   help="Max seconds to wait for a fresh /pose_init "
                        "during a re-initialisation before aborting "
                        "(disarm + resume on the previous pose). "
                        "0 = wait forever.")
    p.add_argument("--pose-init-topic", default="/pose_init")
    p.add_argument("--pose-init-timeout", type=float, default=120.0)
    p.add_argument("--matrix-topic", default="/pose_estimation/pose_matrix_output")
    p.add_argument("--handoff-ready-timeout", type=float, default=20.0)
    p.add_argument("--seg-publish-topic", default="/left/segmentation")
    p.add_argument("--seg-image-topic", default="/left/image_rect")
    p.add_argument("--viz-layout", choices=["2x2", "3x2"], default="3x2",
                   help="Display/video composite: 3x2 = overlay + wireframe-"
                        "DIFFERENCE + segmentation rows (the stereo_ipcai_"
                        "pipeline reference view); 2x2 omits the difference "
                        "row.")
    p.add_argument("--gpu-viz", action="store_true",
                   help="Enable the GPU-rendered viz panels in the "
                        "--display-progress window (rendered silhouette + "
                        "seg-vs-render DIFFERENCE view). Extra GPU work per "
                        "displayed frame — off for benchmarks.")
    p.add_argument("--wireframe-thickness", type=int, default=2,
                   help="Line thickness (px) of the estimate wireframe in "
                        "the --display-progress window (srbag viz).")
    p.add_argument("--display-size", default="1280x720", metavar="WxH",
                   help="Window size for --display-progress (WINDOW_NORMAL: "
                        "content scales to fit, whatever layout srbag draws). "
                        "Same fit-to-screen behavior as the old viewer.")
    p.add_argument("--display-progress", action="store_true",
                   help="Re-enable srbag's live cv2 window (wireframe "
                        "overlay on the current frame). Costs a compositor "
                        "hop on the pipeline machine — keep OFF for "
                        "benchmarks; the printed Frame-N stats and the "
                        "headset ghost are the zero-cost instruments.")
    p.add_argument("--seg-debug-dir", default="")

    # Robot model
    p.add_argument("--urdf-file", default=None)
    p.add_argument("--ros-package", default="")
    p.add_argument("--xacro-path", default="")
    p.add_argument("--root-link-name", default="lbr_link_0")
    p.add_argument("--end-link-name", default="lbr_link_7")
    p.add_argument("--collision-meshes", action="store_true")
    p.add_argument("--right-extrinsics-file", required=True)

    # IPCAI optimiser
    p.add_argument("--ipcai-lr", type=float, default=5e-3)
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mode", choices=[m.value for m in REGISTRATION_MODE],
                   default=REGISTRATION_MODE.SEGMENTATION.value)
    p.add_argument("--tversky-alpha", type=float, default=0.7)
    p.add_argument("--tversky-beta", type=float, default=0.3)
    p.add_argument("--use-gaussian-blur", default="false")
    p.add_argument("--gaussian-kernel-size", type=int, default=15)
    p.add_argument("--gaussian-sigma", type=float, default=2.0)
    p.add_argument("--pth", type=float, default=0.5)
    p.add_argument("--num-components", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--preserve-aspect", action="store_true",
                   help="Letterbox the seg input to 576x960 (single scale, "
                        "zero-pad) instead of anamorphically stretching it. "
                        "Avoids squishing the robot for the seg model; the mask "
                        "is cropped to the valid region before upsampling back "
                        "to the input size.")
    p.add_argument("--no-pipeline", action="store_true",
                   help="Disable seg-of-N+1 / opt-of-N overlap.")

    # Seg model
    p.add_argument("--model-path", default=".cache/torch/hub/checkpoints/roboreg")
    p.add_argument("--model-name", default="model.pt")
    p.add_argument("--model-url",
                   default="https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro")
    p.add_argument("--max-jobs", type=int, default=2)

    # Overlay (in-headset ghost) + estimated-pose republish
    p.add_argument("--send-overlay", action="store_true")
    p.add_argument("--overlay-socket", default="/tmp/xr_overlay.sock")
    p.add_argument("--overlay-intrinsics-json",
                   default="/tmp/xr_intrinsics_snapshot.json")
    p.add_argument("--overlay-color", default="0,255,0")
    p.add_argument("--overlay-debug-every", type=int, default=15)
    p.add_argument("--publish-estimated-pose", action="store_true")
    # Defaults match the validated CloudXR bridge so the on-device WebXR
    # OverlayModule (which subscribes /xr/extrinsic_left over rosbridge) reads
    # the estimate exactly as it read the bridge's ground-truth extrinsic:
    # camera-in-base (H_c2b), framed in the robot base.
    p.add_argument("--estimated-pose-topic",
                   default="/xr/extrinsic_left_estimated")
    p.add_argument("--estimated-pose-frame", default="robot_base")

    # Run control + output
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--live-read-timeout", type=float, default=10.0)
    p.add_argument("--live-latest-only", action="store_true",
                   help="Always process the freshest frame (lowest latency).")
    p.add_argument("--final-extrinsics-file", default=None)
    p.add_argument("--output-dir", required=True)

    args = p.parse_args()

    # Handoff is the default; a fixed --extrinsics-file opts out.
    args.handoff = args.extrinsics_file is None

    # Attributes srbag reads that are intentionally fixed for this slim driver
    # (no on-screen display, no video, no frame dumping).
    _fixed = dict(
        optimizer="ipcai",
        save_frames=False, save_frames_dir=None, save_per_frame_npy=False,
        save_video=False, video_fps=20, video_codec="mp4v",
        video_quality=80, video_scale=1.0,
    )
    for k, v in _fixed.items():
        setattr(args, k, v)
    args.video_layout = args.viz_layout      # srbag reads video_layout
    return args


# =============================================================================
# Slim run loop --- the live -> handoff -> IPCAI essence of the original
# run_pipeline, minus video / frame-dump / display / telemetry.
# =============================================================================
def run(args, device):
    mode = REGISTRATION_MODE(args.mode)
    srbag._cache.reset(args.ipcai_lr, args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    per_dir = os.path.join(args.output_dir, "per_frame")
    os.makedirs(per_dir, exist_ok=True)

    print(f"[Consumer] IPCAI live | lr={args.ipcai_lr} "
          f"| iter={args.max_iterations} | mode={mode.value} | pth={args.pth}")

    # Load the seg model before the handoff so masks are ready immediately.
    if srbag.coarse_model is None or srbag.seg_graph is None:
        print("[Consumer] loading segmentation model...")
        srbag.init_seg_model_and_graph(args, device)
        srbag.init_processing_state(args, device)
        print("[Consumer] seg model ready.")

    # In handoff mode the producer is gated until the user presses A, so wait
    # generously for the first camera_info.
    caminfo_timeout = 300.0 if args.handoff else 10.0
    # Optimiser scene resolution follows --optimiser-res:
    #   native (default): scene from the producer's raw-res camera_info, image
    #     from the native JPEG topics — render + upsampled mask at full res.
    #   downsampled: scene from camera_info_rect, image from the rectified raw
    #     topics — the whole optimiser runs at 960x576 (the previous mode).
    # Either way the seg downsample to 576x960 happens inside srbag.extract_mask.
    if args.optimiser_res == "downsampled":
        scene_caminfo = fetch_camera_info_from_topics(
            args, timeout_sec=caminfo_timeout,
            left_topic=args.left_camera_info_topic,
            right_topic=args.right_camera_info_topic)
    else:
        scene_caminfo = fetch_camera_info_from_topics(
            args, timeout_sec=caminfo_timeout,
            left_topic=args.opt_left_camera_info_topic,
            right_topic=args.opt_right_camera_info_topic)
    print(f"[CamInfo] optimiser scene ({args.optimiser_res}): "
          f"{int(scene_caminfo[0].width)}x{int(scene_caminfo[0].height)}")
    resources = init_pipeline_resources(args, device, caminfo_msgs=scene_caminfo)
    img_diagonal, scene = resources["img_diagonal"], resources["scene"]
    K_left = resources.get("K_left")

    # Subscribe before the handoff wait so frames are buffering when the seed
    # pose lands. In downsampled mode the optimiser reads the rectified raw
    # Image topics; in native mode it reads --left-topic/--right-topic.
    if args.optimiser_res == "downsampled":
        source = make_source(args, device,
                             left_topic=args.rect_left_topic,
                             right_topic=args.rect_right_topic,
                             compressed=False)
    else:
        source = make_source(args, device)

    estimated_pose_pub = None
    if args.publish_estimated_pose:
        from geometry_msgs.msg import PoseStamped as _PoseStamped
        estimated_pose_pub = source.node.create_publisher(
            _PoseStamped, args.estimated_pose_topic, 10)
        print(f"[Consumer] publishing estimated pose on "
              f"{args.estimated_pose_topic}")

    overlay_sender = overlay_streamer = None
    if args.send_overlay:
        import overlay_sender as _ovl
        try:
            overlay_sender = _ovl.OverlaySender(args.overlay_socket)
            K_l, K_r = resources.get("K_left"), resources.get("K_right")
            ph, pw = resources["H"], resources["W"]
            if K_l is None or K_r is None:
                raise RuntimeError("pipeline K_left/K_right unavailable")
            fov_l = _ovl.k_to_fov(K_l[0, 0], K_l[1, 1], K_l[0, 2], K_l[1, 2], pw, ph)
            fov_r = _ovl.k_to_fov(K_r[0, 0], K_r[1, 1], K_r[0, 2], K_r[1, 2], pw, ph)
            overlay_sender.set_fov(fov_l, fov_r)
            overlay_streamer = _ovl.AsyncMonoStreamer(overlay_sender, ph, pw, device)
            print(f"[overlay] reuse-mask mono8 streamer ready ({pw}x{ph}) "
                  f"-> {args.overlay_socket}")
        except Exception as e:
            print(f"[overlay] disabled --- setup failed: {e}")
            overlay_sender = overlay_streamer = None

    # Initial pose: handoff seed (/pose_init) or a fixed file.
    if args.handoff:
        print("[Consumer] publishing segmentation, waiting for /pose_init...")
        H_init = do_handoff_seg_and_wait(args, device)
    else:
        H_init = np.load(args.extrinsics_file)
        if H_init.ndim > 2:
            H_init = H_init.reshape(4, 4)
        print(f"[Consumer] loaded init pose from {args.extrinsics_file}")

    # H_init is camera-in-base (IPCAI frame); invert to base-in-camera H_b2l.
    H_b2l = np.linalg.inv(H_init)
    H_b2l_t = torch.from_numpy(H_b2l).float().to(device)
    if H_b2l_t.dim() == 2:
        H_b2l_t = H_b2l_t.unsqueeze(0)
    prev_9d = pk.matrix44_to_se3_9d(H_b2l_t).squeeze(0)

    # Re-initialisation route (all-cold seed): idle listener; see ReInitManager.
    reinit = None
    if args.handoff:
        reinit = ReInitManager(args, device)
        reinit.set_baseline()

        # Keyboard trigger: 'r'+Enter arms a re-init (one key end-to-end
        # against the warm resident seed stack), 'a'+Enter aborts one in
        # progress. stdin thread — works with or without the display window.
        def _kbd():
            import sys
            from std_msgs.msg import Bool
            print("[ReInit] keyboard ready: 'r'+Enter = re-initialise, "
                  "'a'+Enter = abort")
            for line in sys.stdin:
                c = line.strip().lower()
                if c == 'r':
                    m = Bool(); m.data = True
                    reinit._arm_pub.publish(m)
                    print("[ReInit] 'r' -> armed (re-initialisation "
                          "requested)")
                elif c == 'a':
                    reinit.abort()
                    print("[ReInit] 'a' -> abort requested")
        Thread(target=_kbd, daemon=True).start()

    win = "passthrough_ipcai"      # srbag viz window (--display-progress)
    if getattr(args, "display_progress", False):
        # Pre-create resizable + sized: srbag's later imshow reuses the named
        # window, so its composite renders scaled to this size (fit-to-screen,
        # as before) instead of opening at raw composite resolution.
        import cv2
        try:
            dw, dh = (int(v) for v in args.display_size.lower().split("x"))
        except Exception:
            dw, dh = 1280, 720
            print(f"[Consumer] bad --display-size '{args.display_size}', "
                  f"using {dw}x{dh}")
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, dw, dh)
    video_writer = None
    seg_events, opt_events, dt_events, iter_rates = [], [], [], []

    def _read_next():
        return source.read(timeout_sec=args.live_read_timeout,
                           latest_only=args.live_latest_only)

    print("[Consumer] streaming live, forward")
    # Wall-fps origin: now (after model load / scene build / handoff wait), so the
    # reported fps reflects steady-state streaming, not one-time init.
    estimation_start_time = time.perf_counter()
    total_processing_time = 0.0          # host time inside per_frame_core only
    n = 0
    # Steady-state accounting: frame 1 is warmup (excluded). Baselines are
    # snapshotted right after it so subtracting gives steady-state-only totals.
    warmup_t_steady_start_s = None
    warmup_total_processing_time_s = 0.0
    warmup_seg_skipped = warmup_opt_skipped = 0
    warmup_dt_skipped = warmup_iter_skipped = 0
    warmup_frames_delivered = 0
    warmup_read_decode_time_s = 0.0
    steady_frame_intervals_s = []
    last_steady_frame_t_s = None
    # Producer-publish -> consumer-grab latency from the ROS image header stamp
    # (the bridge stamps at publish on the system clock; the consumer reads the
    # same clock), split into components:
    #   wire  = stamp -> subscriber-callback recv  (DDS transport + the rectifier
    #           stage, since the consumer reads /left/image_rect whose stamp is
    #           the preserved original capture stamp)
    #   queue = recv  -> optimiser grab            (consumer-side pipeline depth)
    #   total = wire + queue
    # The producer-side capture+encode is NOT included (the publish stamp is set
    # after encode); exposing it needs a capture stamp threaded through the wire.
    lat_total_ms, lat_wire_ms, lat_queue_ms = [], [], []

    current_frame = _read_next()
    if current_frame is not None:
        next_frame = _read_next()
        prefetched_seg = None
        prev_opt_end_event = None
        opt = None
        try:
            while True:
                _stamp_us = current_frame.get("stamp_us", 0)
                _recv_us  = current_frame.get("recv_us", 0)
                if _stamp_us:
                    _grab_us = time.time() * 1e6
                    lat_total_ms.append(max(0.0, _grab_us - _stamp_us) / 1000.0)
                    if _recv_us:
                        lat_wire_ms.append(max(0.0, _recv_us - _stamp_us) / 1000.0)
                        lat_queue_ms.append(max(0.0, _grab_us - _recv_us) / 1000.0)

                # Re-init: while ARMED, estimation PAUSES entirely — the
                # tracker must not fight FP for the GPU nor keep publishing
                # possibly-bad poses while the seed stack works. We idle here
                # (draining the source so frames stay fresh) until the fresh
                # /pose_init lands, then reset and resume.
                H_new = None
                if reinit is not None and reinit.hold():
                    print("[ReInit] estimation PAUSED while the seed "
                          "stack re-initialises...")
                    # Drop the dual-stream pipeline's in-flight state — it is
                    # tied to pre-pause frames and must not leak across.
                    prefetched_seg = None
                    prev_opt_end_event = None
                    _deadline = (time.perf_counter() + args.reinit_timeout
                                 if args.reinit_timeout > 0 else None)
                    _next_status = time.perf_counter() + 5.0
                    while reinit.hold():
                        H_new = reinit.poll()      # consumes exactly once
                        if H_new is not None:
                            break
                        now = time.perf_counter()
                        if now >= _next_status:
                            _next_status = now + 5.0
                            print(f"[ReInit] waiting... {reinit.status()}")
                        if _deadline is not None and now > _deadline:
                            print(f"[ReInit] TIMEOUT after "
                                  f"{args.reinit_timeout:.0f}s "
                                  f"({reinit.status()}) — aborting: "
                                  f"disarm + resume on the previous pose")
                            reinit.abort()
                            break
                        time.sleep(0.05)
                        _drain = _read_next()      # keep source drained/fresh
                        if _drain is not None:
                            current_frame = _drain
                    if H_new is None:              # released without an init
                        print("[ReInit] released without a fresh init — "
                              "resuming with the previous pose")
                    next_frame = _read_next()

                # A fresh /pose_init resets the DR to original state — new
                # seed pose, optimiser (and all its caches) rebuilt,
                # in-flight seg/events dropped.
                if reinit is not None and H_new is None:
                    H_new = reinit.poll()          # non-paused arrival path
                if H_new is not None:
                        H_b2l_new = np.linalg.inv(H_new)
                        _t = torch.from_numpy(H_b2l_new).float().to(device)
                        if _t.dim() == 2:
                            _t = _t.unsqueeze(0)
                        prev_9d = pk.matrix44_to_se3_9d(_t).squeeze(0)
                        opt = None
                        prefetched_seg = None
                        prev_opt_end_event = None

                t_frame_start = time.perf_counter()
                (prev_9d, opt, opt_end_event, _interval, _frame_error,
                 next_prefetched_seg) = srbag.process_one_frame_dual_stream(
                    current_frame, next_frame, prefetched_seg,
                    scene, args, prev_9d, opt, video_writer, per_dir, img_diagonal,
                    seg_events, opt_events, dt_events, iter_rates, win,
                    H_final_gt=None, target_9d=None, K_left=K_left,
                    H_base_target_world=None, camera_trajectory=None,
                    prev_opt_end_event=prev_opt_end_event)
                total_processing_time += time.perf_counter() - t_frame_start
                n += 1

                if estimated_pose_pub is not None:
                    # Optimiser state prev_9d == best_p == base-in-camera (H_b2l).
                    # The WebXR OverlayModule (and the validated CloudXR bridge
                    # /xr/extrinsic_left) want camera-in-base, i.e. H_c2b =
                    # inv(H_base) @ H_eye  == inv(H_b2l). This is exactly the
                    # pipeline's output pose (srbag's H_l2b = inv(best_p)), so
                    # publish that — same quantity, same convention as the bridge.
                    _Hb2l = pk.se3_9d_to_matrix44(
                        prev_9d.unsqueeze(0) if prev_9d.dim() == 1 else prev_9d
                    )[0].detach().cpu().numpy()
                    _Hc2b = np.linalg.inv(_Hb2l)
                    estimated_pose_pub.publish(
                        _make_pose_stamped(_Hc2b, source.node,
                                           frame_id=args.estimated_pose_frame))

                if overlay_streamer is not None:
                    try:
                        overlay_streamer.pump(
                            srbag._overlay_pL, srbag._overlay_pR, n,
                            opt_end_event,
                            ts_us=current_frame.get("stamp_us", 0))
                    except Exception as _e:
                        if n % 120 == 1:
                            print(f"[overlay] pump error: {_e}")

                prev_opt_end_event = opt_end_event

                # Snapshot steady-state baselines right after the first frame.
                if n == 1:
                    warmup_t_steady_start_s = time.perf_counter()
                    warmup_total_processing_time_s = total_processing_time
                    warmup_seg_skipped = len(seg_events)
                    warmup_opt_skipped = len(opt_events)
                    warmup_dt_skipped = len(dt_events)
                    warmup_iter_skipped = len(iter_rates)
                    warmup_frames_delivered = getattr(source, "frames_delivered", 0)
                    warmup_read_decode_time_s = getattr(
                        source, "read_decode_time_total_s", 0.0)

                # Inter-frame interval for the stall-trimmed wall fps.
                _t_now_steady = time.perf_counter()
                if last_steady_frame_t_s is not None:
                    steady_frame_intervals_s.append(
                        _t_now_steady - last_steady_frame_t_s)
                last_steady_frame_t_s = _t_now_steady

                if n % 10 == 0:
                    wall_so_far = max(
                        time.perf_counter() - estimation_start_time, 1e-9)
                    wall_fps_so_far = n / wall_so_far
                    compute_fps_so_far = n / max(total_processing_time, 1e-9)
                    if lat_total_ms:
                        _lat_str = f"  lat={lat_total_ms[-1]:.0f} ms"
                        if lat_wire_ms and lat_queue_ms:
                            _lat_str += (f" (wire={lat_wire_ms[-1]:.0f} "
                                         f"queue={lat_queue_ms[-1]:.0f})")
                    else:
                        _lat_str = ""
                    print(f"  Frame {n}: wall={wall_fps_so_far:.1f} fps  "
                          f"(compute={compute_fps_so_far:.1f} fps){_lat_str}")

                if next_frame is None:
                    break
                if args.max_frames and n >= args.max_frames:
                    print(f"[Consumer] reached --max-frames={args.max_frames}")
                    break
                current_frame = next_frame
                prefetched_seg = next_prefetched_seg
                next_frame = None
                if not (args.max_frames and n + 1 >= args.max_frames):
                    next_frame = _read_next()
        except KeyboardInterrupt:
            print(f"\n[Consumer] interrupted after {n} frames.")

    source.close()
    if overlay_streamer is not None:
        try: overlay_streamer.flush()
        except Exception: pass
    if overlay_sender is not None:
        try: overlay_sender.close()
        except Exception: pass
    if srbag.dual_stream_mgr is not None:
        srbag.dual_stream_mgr.synchronize_all()

    if n > 0:
        final_4x4 = pk.se3_9d_to_matrix44(
            prev_9d.unsqueeze(0) if prev_9d.dim() == 1 else prev_9d)[0]
        H_final = torch.linalg.inv(final_4x4).cpu().detach().numpy()
        out = os.path.join(per_dir, "final_estimated_pose.npy")
        np.save(out, H_final)
        print(f"[Output] final estimated pose -> {out}")

    # ── Timing summary (mirrors stereo_pipeline_export.py [Done] block) ──
    # All metrics are steady state: frame 1 and everything before it (init,
    # model load, scene build, first compute) is excluded via the warmup_*
    # baselines snapshotted right after the first iteration.
    steady_frames = max(n - 1, 0)
    steady_stall_excluded_s = 0.0
    steady_wall_raw_total_s = 0.0
    if steady_frames > 0 and warmup_t_steady_start_s is not None:
        wall_end_s = (last_steady_frame_t_s if last_steady_frame_t_s is not None
                      else time.perf_counter())
        steady_wall_raw = wall_end_s - warmup_t_steady_start_s
        steady_wall_raw_total_s = steady_wall_raw
        # Stall-trim: cap each inter-frame interval at 3x median (500 ms floor)
        # so a single pause (producer disconnect, GC, end-of-run idle) doesn't
        # drag the headline fps.
        if steady_frame_intervals_s:
            _med_dt = float(np.median(steady_frame_intervals_s))
            _cap_dt = max(_med_dt * 3.0, 0.5)
            _capped = [min(dt, _cap_dt) for dt in steady_frame_intervals_s]
            steady_stall_excluded_s = float(
                sum(steady_frame_intervals_s) - sum(_capped))
            steady_wall_raw = float(sum(_capped))
        steady_wall_time = max(steady_wall_raw, 1e-9)
        steady_wall_fps = steady_frames / steady_wall_time
        steady_compute_time = total_processing_time - warmup_total_processing_time_s
        steady_compute_fps = (steady_frames / steady_compute_time
                              if steady_compute_time > 0 else 0)
        steady_io_overhead_ms = (1000.0 * (steady_wall_time - steady_compute_time)
                                 / steady_frames)
    else:
        steady_wall_time = max(time.perf_counter() - estimation_start_time, 1e-9)
        steady_wall_fps = n / steady_wall_time
        steady_compute_time = total_processing_time
        steady_compute_fps = (n / total_processing_time
                              if total_processing_time > 0 else 0)
        steady_io_overhead_ms = (1000.0 * (steady_wall_time - total_processing_time)
                                 / max(n, 1))
        steady_frames = n

    seg_steady = seg_events[warmup_seg_skipped:]
    opt_steady = opt_events[warmup_opt_skipped:]
    dt_steady = dt_events[warmup_dt_skipped:]
    iter_steady = iter_rates[warmup_iter_skipped:]

    avg_seg, std_seg, seg_fps, _ = srbag._compute_event_stats(seg_steady)
    avg_opt, std_opt, opt_fps, _ = srbag._compute_event_stats(opt_steady)
    avg_dt, std_dt, _, dt_times = srbag._compute_event_stats(dt_steady)
    avg_iter = float(np.mean(iter_steady)) if iter_steady else 0.0
    std_iter = float(np.std(iter_steady)) if iter_steady else 0.0
    seq_time = avg_seg + avg_opt + (avg_dt if dt_times else 0)
    sequential_fps = 1.0 / seq_time if seq_time > 0 else 0
    overlap_speedup = (steady_wall_fps / sequential_fps
                       if sequential_fps > 0 else 1.0)

    print(f"\n[Done] Frames={n} ({steady_frames} steady-state, 1 warmup excluded)")
    print(f"  wall:    {steady_wall_time:.1f}s  -> {steady_wall_fps:.1f} fps   "
          f"(host clock, excludes warmup)")
    if steady_stall_excluded_s > 0.25 and steady_wall_raw_total_s > 0:
        _raw_fps = steady_frames / max(steady_wall_raw_total_s, 1e-9)
        print(f"           (raw {steady_wall_raw_total_s:.1f}s -> {_raw_fps:.1f} fps; "
              f"trimmed {steady_stall_excluded_s:.1f}s of stalls — "
              f"producer pause or end-of-run idle)")
    print(f"  compute: {steady_compute_time:.1f}s  -> {steady_compute_fps:.1f} fps   "
          f"(per_frame_core only, GPU, excludes warmup)")

    src_delivered = getattr(source, "frames_delivered", 0)
    src_read_decode_total = getattr(source, "read_decode_time_total_s", 0.0)
    src_steady_delivered = max(src_delivered - warmup_frames_delivered, 0)
    src_steady_read_decode = max(
        src_read_decode_total - warmup_read_decode_time_s, 0.0)
    if src_steady_read_decode > 0 and src_steady_delivered > 0:
        src_rd_fps = src_steady_delivered / src_steady_read_decode
        src_rd_ms = 1000.0 * src_steady_read_decode / src_steady_delivered
        print(f"  source raw read+decode: {src_rd_fps:.1f} fps "
              f"({src_rd_ms:.1f} ms/frame in read())")

    print(f"  I/O overhead: {steady_io_overhead_ms:.1f} ms/frame  "
          f"(read + decode + upload, on the host between iterations)")
    print(f"  seg: {avg_seg*1000:.1f}\u00b1{std_seg*1000:.1f} ms -> {seg_fps:.1f} fps")
    print(f"  opt: {avg_opt*1000:.1f}\u00b1{std_opt*1000:.1f} ms -> {opt_fps:.1f} fps")
    print(f"  sequential (seg+opt): {sequential_fps:.1f} fps  |  "
          f"observed wall: {steady_wall_fps:.1f} fps  |  "
          f"overlap speedup: {overlap_speedup:.2f}x")
    print(f"  IPCAI inner-iter rate: {avg_iter:.1f}\u00b1{std_iter:.1f} it/s")

    # Producer-publish -> consumer-grab latency (ROS header stamp; frame 1
    # excluded as warmup), broken into wire vs consumer-queue components.
    def _stats(a):
        a = np.asarray(a, dtype=float)
        return a.mean(), float(np.median(a)), np.percentile(a, 95), a.max(), a.size
    if len(lat_total_ms) > 1:
        m, med, p95, mx, nn = _stats(lat_total_ms[1:])
        print(f"  producer->consumer latency: "
              f"mean={m:.1f}  median={med:.1f}  p95={p95:.1f}  max={mx:.1f} ms "
              f"(n={nn}, from ROS image header stamp)")
        if len(lat_wire_ms) > 1 and len(lat_queue_ms) > 1:
            wm, wmed, wp95, _, _ = _stats(lat_wire_ms[1:])
            qm, qmed, qp95, _, _ = _stats(lat_queue_ms[1:])
            print(f"    split: wire(publish->recv)  mean={wm:.1f} "
                  f"median={wmed:.1f} p95={wp95:.1f} ms  "
                  f"(DDS transport + rectifier stage)")
            print(f"           queue(recv->grab)    mean={qm:.1f} "
                  f"median={qmed:.1f} p95={qp95:.1f} ms  "
                  f"(consumer pipeline depth)")
            # End-to-end picture: where a frame's time goes once it lands.
            print(f"    chain: wire {wm:.0f}  +  queue {qm:.0f}  +  "
                  f"seg {avg_seg*1000:.0f}  +  opt {avg_opt*1000:.0f} ms "
                  f"(seg||opt overlap, not summed into wall)")
    else:
        print("  producer->consumer latency: n/a (no header stamps seen)")

    print(f"[Done] frames={n}  output={args.output_dir}")


def main():
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(args.max_jobs)
    run(args, device)


if __name__ == "__main__":
    main()