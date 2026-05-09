#!/usr/bin/env python3
"""
Stereo Registration Pipeline — IPCAI (AdamW). Single entry point with three
input modes:

  --source offline   Read a rosbag2 with batched preload (200 frames at a
                     time as uint8 GPU buffers). Lowest per-frame I/O cost
                     during a batch — the seg+opt core gets to run flat-out
                     against in-memory data.

  --source bag       Stream a rosbag2 frame-by-frame, paced to the original
                     record timestamps. Synchronous `next_triplet → cv2
                     decode → GPU upload` on the main thread per call, then
                     sleep so the inter-frame interval matches the bag's
                     recorded rate. Useful for emulating live-camera input.

  --source live      Subscribe to running ROS2 image + joint-state topics.

The streaming loop body is ported from `stereo_ipcai_pipeline_bag.run_pipeline`.
The per-frame seg + opt core is `srbag.process_one_frame_dual_stream`. The
only thing this script contributes on top of the shared library is the source
class abstraction and live-ROS CameraInfo fetching.

Usage:
  python stereo_pipeline_live.py --source offline --bag-path /path/to/bag --output-dir ...
  python stereo_pipeline_live.py --source bag     --bag-path /path/to/bag --output-dir ...
  python stereo_pipeline_live.py --source live --output-dir ...
"""
import os
import argparse
import time
import tempfile
import yaml
import numpy as np
import cv2
import torch
import pytorch_kinematics as pk
from collections import deque
from threading import Lock, Thread, Event

from roboreg.core import (
    NVDiffRastRenderer,
    Robot,
    RobotScene,
    TorchKinematics,
    TorchMeshContainer,
    VirtualCamera,
)
from roboreg.io import (
    load_robot_data_from_ros_xacro,
    load_robot_data_from_urdf_file,
)

# Shared per-frame core, optimizer, FK cache, dual-stream manager, etc.
import stereo_ipcai_pipeline_bag as srbag
from stereo_ipcai_pipeline_bag import REGISTRATION_MODE


# =============================================================================
# ROS2 stream sources — the only logic unique to this script
# =============================================================================
def _decode_image_msg_to_numpy(msg):
    """Convert sensor_msgs/Image to (H, W, 3) uint8 RGB numpy."""
    return srbag._decode_image_msg_to_numpy(msg)


class FrameSource:
    """Yields {'left_img', 'right_img', 'joint_state', 'frame_idx'} dicts.

    Subclasses must override `_read_impl()` (NOT `read()`) and `close()`.
    `read()` is wrapped here to record two timing numbers automatically:

        t_first_read_s, t_last_read_s        — perf_counter timestamps of the
                                                first and most recent successful
                                                read. Their difference gives the
                                                source's playing-speed (the rate
                                                at which frames were actually
                                                delivered during the run).

        read_decode_time_total_s             — cumulative time spent inside
                                                `_read_impl()`. Frames divided
                                                by this gives the raw read+
                                                decode rate (how fast the
                                                source CAN deliver frames).
    """
    def __init__(self):
        self.t_first_read_s = None
        self.t_last_read_s = None
        self.read_decode_time_total_s = 0.0
        self.frames_delivered = 0

    def read(self, *args, **kwargs):
        t0 = time.perf_counter()
        # Snapshot any time the subclass reports as "excluded" (e.g. preload
        # bursts that the offline source amortises). Anything added during
        # this call is subtracted from `read_decode_time_total_s` so the
        # latter measures only actual per-frame read+decode cost.
        excluded_before = getattr(self, 'preload_time_total_s', 0.0)
        try:
            frame = self._read_impl(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            excluded_added = getattr(self, 'preload_time_total_s', 0.0) - excluded_before
            self.read_decode_time_total_s += max(elapsed - excluded_added, 0.0)
        if frame is not None:
            now = time.perf_counter()
            if self.t_first_read_s is None:
                self.t_first_read_s = now
            self.t_last_read_s = now
            self.frames_delivered += 1
        return frame

    def _read_impl(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        pass


def _read_one_triplet_sync(reader, device, frame_counter):
    """Pull one synchronised triplet from a BagStreamReader, decode + upload,
    return (frame_dict, left_msg) or None at EOF.
    """
    triplet = reader.next_triplet()
    if triplet is None:
        return None
    mL, mR, mJ = triplet
    left_img = _decode_image_msg_to_numpy(mL)
    right_img = _decode_image_msg_to_numpy(mR)
    js_arr = np.asarray(mJ.position, dtype=np.float32)
    left_t = torch.from_numpy(left_img).to(device).float().div_(255.0).permute(2, 0, 1)
    right_t = torch.from_numpy(right_img).to(device).float().div_(255.0).permute(2, 0, 1)
    js = torch.tensor(js_arr, dtype=torch.float32, device=device).unsqueeze(0)
    return ({'left_img': left_t, 'right_img': right_t,
             'joint_state': js, 'frame_idx': frame_counter}, mL)


class OfflineBagSource(FrameSource):
    """Bag input, batched preload version.

    Reads `batch_size` frames at a time from the bag: cv2-decodes them on the
    host, uploads them to GPU as **uint8** tensors (`(N, H, W, 3)` for each
    side), and stores them along with the joint-state stack in CPU/GPU memory.
    Then `read()` walks through this in-memory batch and returns one frame
    per call, doing only the per-frame float32 conversion + permute at consume
    time. When the batch is drained, the next 200 frames are preloaded.

    Memory cost (default batch_size=200, 1080p stereo):
        200 frames × 2 cameras × 1080×1920×3 × 1 byte ≈ 2.4 GB GPU memory.

    Per-frame cost during a batch: just `uint8 → float32 / 255 → permute`,
    a small fused kernel on already-resident GPU data. The bulk
    read+decode+upload happens once per batch boundary, not once per frame —
    so the GPU dual-stream prefetch overlap inside
    `process_one_frame_dual_stream` actually has time to overlap with the
    next-frame H2D upload (which it can't when uploads happen synchronously
    between iterations).
    """
    def __init__(self, bag_path, left_topic, right_topic, js_topic, device,
                 max_dt_ms=50, storage_id=None, batch_size=200):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.reader = srbag.BagStreamReader(
            bag_path, left_topic, right_topic, js_topic,
            max_dt_ms=max_dt_ms, storage_id=storage_id)
        self.frame_counter = 0

        # Per-batch state — refilled by _preload_batch.
        self._batch_left_u8 = None   # uint8 (N, H, W, 3) on GPU
        self._batch_right_u8 = None  # uint8 (N, H, W, 3) on GPU
        self._batch_js = None        # float32 (N, J) on GPU
        self._batch_idx = 0          # next-to-yield index inside current batch
        self._batch_n = 0            # number of frames in current batch
        self._exhausted = False

        # Total time spent inside `_preload_batch` (host clock). The main loop
        # subtracts this from wall_time when computing wall fps so that
        # processing throughput isn't dragged down by the bulk read+decode+
        # upload bursts that happen at every batch boundary.
        self.preload_time_total_s = 0.0

    def _preload_batch(self):
        """Fill _batch_* with the next batch_size frames from the reader.
        Sets self._exhausted=True if the reader is fully drained. Returns
        True if any frames were loaded into the batch. Adds elapsed wall
        time to self.preload_time_total_s so the main loop can exclude it
        from the throughput calculation."""
        if self._exhausted:
            return False

        t0 = time.perf_counter()
        try:
            left_imgs, right_imgs, js_arrs = [], [], []
            for _ in range(self.batch_size):
                triplet = self.reader.next_triplet()
                if triplet is None:
                    self._exhausted = True
                    break
                mL, mR, mJ = triplet
                left_imgs.append(_decode_image_msg_to_numpy(mL))
                right_imgs.append(_decode_image_msg_to_numpy(mR))
                js_arrs.append(np.asarray(mJ.position, dtype=np.float32))

            n = len(left_imgs)
            if n == 0:
                self._batch_n = 0
                self._batch_idx = 0
                return False

            # Stack on the host then upload once. uint8 keeps GPU memory tiny.
            left_np = np.stack(left_imgs, axis=0)   # (N, H, W, 3) uint8
            right_np = np.stack(right_imgs, axis=0)
            js_np = np.stack(js_arrs, axis=0)        # (N, J) float32

            self._batch_left_u8 = torch.from_numpy(left_np).to(self.device, non_blocking=True)
            self._batch_right_u8 = torch.from_numpy(right_np).to(self.device, non_blocking=True)
            self._batch_js = torch.from_numpy(js_np).to(self.device, non_blocking=True)

            self._batch_n = n
            self._batch_idx = 0
            return True
        finally:
            self.preload_time_total_s += time.perf_counter() - t0

    def _read_impl(self):
        # Refill the batch if we've drained the current one.
        if self._batch_idx >= self._batch_n:
            if not self._preload_batch():
                return None

        i = self._batch_idx
        # Per-frame conversion: uint8 (H, W, 3) -> float32 (3, H, W) / 255.
        # `permute(2,0,1)` then `.contiguous()` so downstream `extract_mask`'s
        # `torch.stack([L, R], dim=0)` produces a contiguous (2, 3, H, W).
        left_t = (self._batch_left_u8[i].float().div_(255.0).permute(2, 0, 1).contiguous())
        right_t = (self._batch_right_u8[i].float().div_(255.0).permute(2, 0, 1).contiguous())
        js = self._batch_js[i].unsqueeze(0)

        self._batch_idx += 1
        idx = self.frame_counter
        self.frame_counter += 1
        return {'left_img': left_t, 'right_img': right_t,
                'joint_state': js, 'frame_idx': idx}

    def close(self):
        # Free GPU memory held by the current batch.
        self._batch_left_u8 = None
        self._batch_right_u8 = None
        self._batch_js = None


class BagSource(FrameSource):
    """Bag input, paced to original record timestamps.

    Same synchronous read path as OfflineBagSource — `next_triplet → cv2
    decode → CPU→GPU upload` on the main thread — but inserts a sleep
    between successive frames so the inter-frame interval matches the wall
    time at which the messages were originally recorded. Replays at the
    camera's true publish rate, useful for emulating live-camera behavior
    against a saved bag.

    The pacing sleep happens in the overridden `read()`, AFTER `_read_impl`
    has returned. `read_decode_time_total_s` (in the base class) therefore
    counts only actual read+decode cost, not the sleep — so the
    "source raw read+decode" line in the timing summary still reflects the
    true I/O cost while wall fps reflects paced playback.

    Compared to `OfflineBagSource`:
        offline: 200-frame batched preload, no pacing, runs as fast as possible.
        bag:     stream-read frame-by-frame, paced to the bag's record rate.
    """
    def __init__(self, bag_path, left_topic, right_topic, js_topic, device,
                 max_dt_ms=50, storage_id=None):
        super().__init__()
        self.device = device
        self.reader = srbag.BagStreamReader(
            bag_path, left_topic, right_topic, js_topic,
            max_dt_ms=max_dt_ms, storage_id=storage_id)
        self.frame_counter = 0
        # Pacing state — set on first successful read.
        self._first_bag_ns = None
        self._first_wall_s = None
        self._last_bag_ns = None  # captured by _read_impl, consumed by read()

    def _read_impl(self):
        result = _read_one_triplet_sync(self.reader, self.device, self.frame_counter)
        if result is None:
            return None
        frame, mL = result
        # Capture the bag-clock timestamp for the overridden read() to use
        # for pacing. Hidden in an attribute so we don't leak it through the
        # frame dict (the rest of the pipeline doesn't need to know).
        self._last_bag_ns = (srbag._stamp_to_ns(mL.header.stamp)
                             if hasattr(mL, 'header') else None)
        self.frame_counter += 1
        return frame
        # Capture the bag-clock timestamp for the overridden read() to use
        # for pacing. Hidden in an attribute so we don't leak it through the
        # frame dict (the rest of the pipeline doesn't need to know).
        self._last_bag_ns = (srbag._stamp_to_ns(mL.header.stamp)
                             if hasattr(mL, 'header') else None)
        self.frame_counter += 1
        return frame

    def read(self):
        # Run the timed read+decode via the base class.
        frame = super().read()
        if frame is None:
            return None
        bag_ns = self._last_bag_ns
        if bag_ns is not None:
            now = time.perf_counter()
            if self._first_bag_ns is None:
                # Anchor the bag clock to wall time at the first frame.
                self._first_bag_ns = bag_ns
                self._first_wall_s = now
            else:
                target = self._first_wall_s + (bag_ns - self._first_bag_ns) / 1e9
                delay = target - now
                if delay > 0:
                    time.sleep(delay)
        # The base class already updated t_last_read_s before this sleep.
        # Update it again so it reflects the post-sleep wall time, making
        # `t_last_read_s - t_first_read_s` track paced playback.
        self.t_last_read_s = time.perf_counter()
        return frame

    def close(self):
        pass


class LiveSource(FrameSource):
    """Subscribe to live ROS2 stereo image + joint-state topics.

    Stereo images are paired via ApproximateTimeSynchronizer; joint state is
    subscribed separately (latest-wins) and attached to the most recent stereo
    pair when read. A background spin thread feeds a small queue.
    """
    def __init__(self, left_topic, right_topic, js_topic, device,
                 queue_size=2, slop_ms=50):
        super().__init__()
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import Image, JointState
        import message_filters

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
        # Dedicated executor so this source's spinning can't collide with
        # another `rclpy.spin_once`-based caller (e.g. a one-shot caminfo
        # fetcher running concurrently for bag-play mode).
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)

        # Dedicated CUDA upload stream for the spin-thread callbacks. With
        # pinned source memory (allocated lazily on first frame, see below),
        # H2D copies issued on this stream go through the GPU's hardware
        # copy engine — a piece of silicon separate from the SMs. That means
        # the seg+opt kernels (which use SMs and SM-attached memory
        # bandwidth) can run concurrently with the upload without contending
        # for the same hardware. Without pinned memory, `non_blocking=True`
        # silently falls back to a synchronous copy that stalls the upload
        # stream and burns memory bandwidth shared with the seg/opt streams.
        self._upload_stream = (torch.cuda.Stream(device=device)
                               if str(device).startswith('cuda') else None)

        # Pinned-memory buffer pool, lazily allocated on first frame because
        # we don't know (H, W, C) until then. Round-robin reused — pool size
        # of 3 guarantees that by the time we wrap back to a buffer, the
        # H2D copy that used it is long done (queue_size is 2, main thread
        # pops one at a time, callbacks fire at ~camera rate).
        self._pin_pool_size = 3
        self._pin_left = None   # list[Tensor] (CPU, pinned), shape (H,W,C)
        self._pin_right = None
        self._pin_idx = 0

        sub_l = message_filters.Subscriber(self.node, Image, left_topic)
        sub_r = message_filters.Subscriber(self.node, Image, right_topic)
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
        left_img = _decode_image_msg_to_numpy(msg_l)
        right_img = _decode_image_msg_to_numpy(msg_r)

        if self._upload_stream is not None:
            # Lazy-allocate the pinned-buffer pool on first frame, now that
            # we know the camera resolution.
            if self._pin_left is None:
                shape = left_img.shape  # (H, W, C), uint8
                self._pin_left = [torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                                  for _ in range(self._pin_pool_size)]
                self._pin_right = [torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                                   for _ in range(self._pin_pool_size)]

            # Stage the numpy frames into the next pinned buffer (CPU memcpy
            # ~6 MB per side at 1080p — fast, ~1ms). Round-robin index.
            i = self._pin_idx
            self._pin_idx = (self._pin_idx + 1) % self._pin_pool_size
            self._pin_left[i].copy_(torch.from_numpy(left_img))
            self._pin_right[i].copy_(torch.from_numpy(right_img))

            # Async H2D via the copy engine, then dtype convert + permute on
            # the same upload stream. With pinned source, `non_blocking=True`
            # is honoured by the runtime and the copy proceeds asynchronously
            # on dedicated copy-engine hardware. We still synchronize() at
            # the end so the consumer queue holds GPU-resident tensors.
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
        with self.lock:
            if self.latest_js is None:
                return  # wait for first joint-state message before accepting frames
            self.queue.append({
                'left_img': left_t, 'right_img': right_t,
                'joint_state': self.latest_js,
                'frame_idx': self.frame_counter})
            self.frame_counter += 1
        self.new_frame.set()

    def _read_impl(self, timeout_sec=10.0, latest_only=False):
        """Pop a frame from the queue, blocking up to `timeout_sec`.

        latest_only=True: pop the NEWEST frame and discard older ones in the
        queue. Best for low-latency live tracking — always work on the freshest
        observation. latest_only=False (default): pop the oldest frame
        (FIFO) — best when you want to process every frame the camera publishes.
        """
        deadline = time.perf_counter() + timeout_sec
        dropped = 0
        while True:
            with self.lock:
                if self.queue:
                    if latest_only and len(self.queue) > 1:
                        dropped = len(self.queue) - 1
                        # Drain everything but the newest.
                        while len(self.queue) > 1:
                            self.queue.popleft()
                    frame = self.queue.popleft()
                    if dropped > 0:
                        frame['dropped_before'] = dropped
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


def make_source(args, device):
    if args.source == 'offline':
        return OfflineBagSource(
            args.bag_path, args.left_topic, args.right_topic, args.joint_state_topic,
            device, max_dt_ms=args.sync_slop_ms,
            storage_id=getattr(args, 'bag_storage_id', None),
            batch_size=getattr(args, 'batch_size', 200))
    if args.source == 'bag':
        return BagSource(
            args.bag_path, args.left_topic, args.right_topic, args.joint_state_topic,
            device, max_dt_ms=args.sync_slop_ms,
            storage_id=getattr(args, 'bag_storage_id', None))
    return LiveSource(
        args.left_topic, args.right_topic, args.joint_state_topic,
        device, slop_ms=args.sync_slop_ms)


# =============================================================================
# Camera-info fetching (from live topics or from a bag)
# =============================================================================
def fetch_camera_info_from_topics(args, timeout_sec=10.0):
    """Subscribe once to left & right CameraInfo topics, return (left_msg, right_msg).

    Uses a dedicated `SingleThreadedExecutor` rather than the default one so
    it can't collide with another spinner (e.g. a `LiveSource` spin thread
    already running for bag-play mode — see `BagSource`). `rclpy.spin_once`
    on the default executor would raise "Executor is already spinning" in
    that case.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import CameraInfo

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
        CameraInfo, args.left_camera_info_topic, _mk_cb('left'), 10)
    sub_r = node.create_subscription(
        CameraInfo, args.right_camera_info_topic, _mk_cb('right'), 10)

    print(f"[CamInfo] Waiting for one msg each on '{args.left_camera_info_topic}', "
          f"'{args.right_camera_info_topic}' (timeout {timeout_sec}s)...")
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


def fetch_camera_info_from_bag(args):
    """Walk the bag once until first CameraInfo on each side, return (left_msg, right_msg)."""
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    sid = srbag._detect_bag_storage_id(args.bag_path,
                                       override=getattr(args, 'bag_storage_id', None))
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag_path, storage_id=sid),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for topic in (args.left_camera_info_topic, args.right_camera_info_topic):
        if topic not in type_map:
            raise RuntimeError(f"[CamInfo] Topic '{topic}' not in bag {args.bag_path}. "
                               f"Available: {list(type_map.keys())}")
    msg_cls = {t: get_message(type_map[t])
               for t in (args.left_camera_info_topic, args.right_camera_info_topic)}

    print(f"[CamInfo] Scanning bag for '{args.left_camera_info_topic}' and "
          f"'{args.right_camera_info_topic}'...")
    received = {args.left_camera_info_topic: None, args.right_camera_info_topic: None}
    while reader.has_next() and any(v is None for v in received.values()):
        topic, raw, _ = reader.read_next()
        if topic in received and received[topic] is None:
            received[topic] = deserialize_message(raw, msg_cls[topic])

    missing = [t for t, m in received.items() if m is None]
    if missing:
        raise RuntimeError(f"[CamInfo] Bag ended without CameraInfo on: {missing}")

    left_msg = received[args.left_camera_info_topic]
    right_msg = received[args.right_camera_info_topic]
    H, W = int(left_msg.height), int(left_msg.width)
    print(f"[CamInfo] Got {W}x{H} from bag")
    return left_msg, right_msg


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
    """Build cameras + scene from CameraInfo messages, then init the shared
    seg model / CUDA graph / dual-stream manager / cache via the bag-pipeline
    module. After this returns, `srbag.process_one_frame_dual_stream` is ready
    to be called.

    Camera + scene construction follows the rr-stereo `stereo_ipcai_pipeline.py`
    reference: VirtualCamera + Robot + RobotScene + NVDiffRastRenderer composed
    directly. No `stereo_benchmark` / `create_stereo_cameras` /
    `create_scene_from_args` helpers — the new install doesn't expose those.
    """
    left_msg, right_msg = caminfo_msgs
    H, W = int(left_msg.height), int(left_msg.width)

    # `VirtualCamera.from_camera_configs` takes a YAML path. CameraInfo arrives
    # as a sensor_msgs message in live mode and from the bag in offline/bag
    # mode — in both cases we materialise it to a temp YAML and feed that in,
    # then delete the file once the camera object has loaded it.
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

    # Robot description: prefer --urdf-file when given, else fall back to
    # --ros-package / --xacro-path. Same precedence as rr-stereo's reference.
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

    # Streaming runs one frame at a time → batch_size=1.
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

    # Set up the bag-pipeline module's seg model, CUDA graph and processing state.
    srbag.init_seg_model_and_graph(args, device)
    srbag.init_processing_state(args, device)

    # Build K from the CameraInfo message directly — the temp YAML has been
    # deleted by now, so anything downstream that wants the intrinsics has to
    # get them from here.
    K_left = None
    k_data = list(left_msg.k) if left_msg.k is not None else None
    if k_data is not None and len(k_data) == 9:
        K_left = np.array(k_data, dtype=np.float64).reshape(3, 3)

    return {'img_diagonal': float(np.sqrt(H**2 + W**2)),
            'scene': scene, 'H': H, 'W': W, 'K_left': K_left}


# =============================================================================
# CLI
# =============================================================================
def args_factory():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="IPCAI Stereo Registration Pipeline (ROS2 streaming)")
    # Source
    parser.add_argument("--source", choices=['offline', 'bag', 'live'], required=True,
                        help="'offline': read a rosbag2 with batched preload (200 frames at a "
                             "time, uint8 GPU buffer). Lowest per-frame I/O. "
                             "'bag': stream a rosbag2 frame-by-frame, paced to the original "
                             "record timestamps. Synchronous next_triplet + cv2 decode + GPU "
                             "upload per call, then sleep to match record rate. "
                             "'live': subscribe to running ROS2 topics.")
    parser.add_argument("--bag-path", default=None,
                        help="Path to rosbag2 directory (required for --source offline or bag).")
    parser.add_argument("--bag-storage-id", default=None, choices=['mcap', 'sqlite3'],
                        help="Override rosbag2 storage backend. Auto-detected if omitted.")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Offline-only: number of frames to preload per batch as "
                             "uint8 GPU tensors. Larger = fewer I/O stalls but more "
                             "GPU memory (~12 MB per frame at 1080p stereo). "
                             "Ignored for --source bag and --source live.")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="Disable seg-of-N+1 / opt-of-N overlap (forwarded to shared core).")
    parser.add_argument("--left-topic", default="/zed/zed_node/left/image_rect_color")
    parser.add_argument("--right-topic", default="/zed/zed_node/right/image_rect_color")
    parser.add_argument("--left-camera-info-topic",
                        default="/zed/zed_node/left/camera_info")
    parser.add_argument("--right-camera-info-topic",
                        default="/zed/zed_node/right/camera_info")
    parser.add_argument("--joint-state-topic", default="/lbr/joint_states")
    parser.add_argument("--sync-slop-ms", type=float, default=50.0,
                        help="Approximate-time-sync slop for triplet association (ms).")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after this many frames (0 = run until source ends / Ctrl-C).")
    parser.add_argument("--live-read-timeout", type=float, default=10.0)
    parser.add_argument("--live-latest-only", action="store_true",
                        help="Live mode: drop queued older frames and always process the newest. "
                             "Lowest latency for tracking; default is FIFO (process every frame).")
    # Paths
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extrinsics-file", required=True,
                        help="Initial camera-to-base transform (4x4 .npy).")
    parser.add_argument("--right-extrinsics-file", required=True)
    parser.add_argument("--final-extrinsics-file", default=None,
                        help="Optional final camera-to-base transform (.npy). When present, "
                             "per-frame translation/rotation error against this GT is reported "
                             "and saved to per_frame_pose_error.csv.")
    # IPCAI optimizer
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--ipcai-lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Loss
    parser.add_argument("--mode", choices=[m.value for m in REGISTRATION_MODE],
                        default=REGISTRATION_MODE.SEGMENTATION.value,
                        help="'segmentation' = Tversky, 'distance_function' = MSE.")
    parser.add_argument("--tversky-alpha", type=float, default=0.7)
    parser.add_argument("--tversky-beta", type=float, default=0.3)
    parser.add_argument("--use-gaussian-blur", default="false",
                        help="Set 'true' to apply blur to soft target (only valid in distance_function mode).")
    parser.add_argument("--gaussian-kernel-size", type=int, default=15)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    # Model / Robot
    parser.add_argument('--model-path', default='.cache/torch/hub/checkpoints/roboreg')
    parser.add_argument('--model-name', default='model.pt')
    parser.add_argument('--model-url',
                        default='https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro')
    parser.add_argument("--ros-package", default="")
    parser.add_argument("--xacro-path", default="")
    parser.add_argument("--urdf-file", default=None)
    parser.add_argument("--root-link-name", default="lbr_link_0")
    parser.add_argument("--end-link-name", default="lbr_link_7")
    parser.add_argument("--collision-meshes", action="store_true")
    # Processing
    parser.add_argument("--pth", type=float, default=0.5)
    parser.add_argument("--num-components", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-jobs", type=int, default=2)
    # Visualization
    parser.add_argument("--display-progress", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-frames-dir", default=None)
    parser.add_argument("--save-per-frame-npy", action="store_true",
                        help="Per frame, write camera_to_base_{left,right}_<i>.npy "
                             "to <output-dir>/per_frame{_gb}/. Off by default — "
                             "synchronous disk writes can stall the loop on slow "
                             "filesystems (e.g. NTFS-FUSE) and produce ~2N small "
                             "files. The final pose is always saved at run end "
                             "regardless of this flag.")
    parser.add_argument("--wireframe-thickness", type=int, default=2)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--video-quality", type=int, default=80)
    parser.add_argument("--video-scale", type=float, default=1.0)

    p = parser.parse_args()
    if p.source in ('offline', 'bag') and not p.bag_path:
        parser.error(f"--bag-path is required when --source {p.source}")
    p.optimizer = "ipcai"
    return p


# =============================================================================
# Streaming loop — same shape as the bag-pipeline's stream-mode branch
# =============================================================================
def run_pipeline(args, device):
    """Stream from a source (offline batched preload / bag streaming / live
    ROS), run seg + opt per frame via `srbag.process_one_frame_dual_stream`.
    Body is ported from the standalone bag-pipeline's `run_pipeline` — only
    the per-frame read is swapped from a local `_read_one_frame()` to a
    source class.
    """
    mode = REGISTRATION_MODE(args.mode)
    srbag._cache.reset(args.ipcai_lr, args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    transform_suffix = "_gb" if args.use_gaussian_blur == "true" else ""
    mode_short = "dist" if mode == REGISTRATION_MODE.DISTANCE_FUNCTION else "seg"
    per_dir = os.path.join(args.output_dir, f"per_frame{transform_suffix}")
    os.makedirs(per_dir, exist_ok=True)

    optimizer_type = srbag.OptimizerType.IPCAI

    loss_str = ("MSE" if mode == REGISTRATION_MODE.DISTANCE_FUNCTION
                else f"Tversky(α={args.tversky_alpha},β={args.tversky_beta})")
    if args.use_gaussian_blur == "true":
        loss_str += f"+GB(k={args.gaussian_kernel_size},σ={args.gaussian_sigma})"
    print(f"\n[Pipeline] IPCAI | source={args.source} | lr={args.ipcai_lr} "
          f"| iter={args.max_iterations} | {loss_str}")

    # Camera info: from ROS topics (live) or from the bag (offline / bag).
    if args.source == 'live':
        caminfo_msgs = fetch_camera_info_from_topics(args)
    else:
        caminfo_msgs = fetch_camera_info_from_bag(args)

    resources = init_pipeline_resources(args, device, caminfo_msgs=caminfo_msgs)
    img_diagonal, scene = resources['img_diagonal'], resources['scene']

    # Initial extrinsics
    H_b2l = np.linalg.inv(np.load(args.extrinsics_file))
    H_b2l_t = torch.from_numpy(H_b2l).float().to(device)
    if H_b2l_t.dim() == 2:
        H_b2l_t = H_b2l_t.unsqueeze(0)
    prev_9d = pk.matrix44_to_se3_9d(H_b2l_t).squeeze(0)

    win = f"stereo - IPCAI + {mode_short.upper()}"
    video_writer = None
    if args.display_progress:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.save_frames:
        for subdir in ["optimization", "difference_wireframe", "segmentation"]:
            os.makedirs(os.path.join(args.save_frames_dir or os.path.join(args.output_dir, "frames"),
                                     subdir), exist_ok=True)

    segmentation_events, optimization_events, dt_events = [], [], []
    iteration_rates = []
    all_frame_errors = []

    # Optional final-pose ground truth (per-frame trans/rot error reporting only)
    H_final_gt = None
    if args.final_extrinsics_file and os.path.exists(args.final_extrinsics_file):
        H_final_gt = np.load(args.final_extrinsics_file)
        if H_final_gt.ndim > 2:
            H_final_gt = H_final_gt.reshape(4, 4)
        print(f"[Final Pose GT] Loaded from: {args.final_extrinsics_file}")

    target_9d = None
    if H_final_gt is not None:
        target_9d = pk.matrix44_to_se3_9d(
            torch.from_numpy(np.linalg.inv(H_final_gt)).float().to(device).unsqueeze(0))

    # K matrix for axis drawing in visualisation — already extracted from
    # the CameraInfo message inside init_pipeline_resources, since the temp
    # YAML written there has been removed by the time we reach this point.
    K_left = resources.get('K_left')
    if K_left is not None and (args.display_progress or args.save_frames or args.save_video):
        print(f"[Viz] K_left loaded: fx={K_left[0,0]:.1f} fy={K_left[1,1]:.1f}")

    print(f"[Estimation] streaming source={args.source} frame-by-frame, forward")

    estimation_start_time = time.perf_counter()
    global_frame_count = 0
    total_processing_time = 0.0

    # Warmup baselines — set after the first iteration completes.
    # Everything before that is "startup": Python init, model load, CUDA graph
    # capture, scene construction, the bag reader's first reads, plus the
    # first-frame compute (which has no prefetched seg from a prior iteration
    # — seg N runs fresh, no overlap). The summary subtracts these baselines
    # so reported fps reflects steady-state behavior, not init + first-frame
    # warmup.
    warmup_t_steady_start_s = None
    warmup_total_processing_time_s = 0.0
    warmup_seg_events_skipped = 0
    warmup_opt_events_skipped = 0
    warmup_dt_events_skipped = 0
    warmup_iter_rates_skipped = 0
    warmup_frames_delivered = 0
    warmup_read_decode_time_s = 0.0
    warmup_preload_time_s = 0.0

    source = make_source(args, device)

    def _read_next():
        if args.source == 'live':
            return source.read(timeout_sec=args.live_read_timeout,
                               latest_only=args.live_latest_only)
        return source.read()

    # Bootstrap: read frame 0 + frame 1 so the very first call can prefetch
    # frame 1's seg on next_stream while opt of frame 0 runs on current_stream.
    current_frame = _read_next()
    if current_frame is not None:
        # Lazy video-writer init from the first frame's resolution
        if args.save_video and video_writer is None:
            h, w = current_frame['left_img'].shape[1:]
            output_h, output_w = int(h * 2 * args.video_scale), int(w * 2 * args.video_scale)
            fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
            video_path = os.path.join(args.output_dir, f"visualization_ipcai_{mode_short}.mp4")
            video_writer = cv2.VideoWriter(video_path, fourcc, args.video_fps,
                                           (output_w, output_h),
                                           params=[cv2.VIDEOWRITER_PROP_QUALITY, args.video_quality])
            if not video_writer.isOpened():
                video_writer = cv2.VideoWriter(video_path, fourcc, args.video_fps, (output_w, output_h))
            if not video_writer.isOpened():
                video_writer, args.save_video = None, False

        next_frame = _read_next()
        prefetched_seg = None
        prev_opt_end_event = None
        opt = None

        try:
            while True:
                t_frame_start = time.perf_counter()
                (prev_9d, opt, opt_end_event, _pipeline_interval, frame_error,
                 next_prefetched_seg) = srbag.process_one_frame_dual_stream(
                    current_frame, next_frame, prefetched_seg,
                    scene, args, prev_9d, opt, video_writer, per_dir, img_diagonal,
                    segmentation_events, optimization_events, dt_events,
                    iteration_rates, win,
                    H_final_gt=H_final_gt, target_9d=target_9d, K_left=K_left,
                    H_base_target_world=None, camera_trajectory=None,
                    prev_opt_end_event=prev_opt_end_event)
                total_processing_time += time.perf_counter() - t_frame_start
                global_frame_count += 1

                if frame_error is not None:
                    all_frame_errors.append(frame_error)
                prev_opt_end_event = opt_end_event

                # Snapshot baselines right after the first iteration finishes.
                if global_frame_count == 1:
                    warmup_t_steady_start_s = time.perf_counter()
                    warmup_total_processing_time_s = total_processing_time
                    warmup_seg_events_skipped = len(segmentation_events)
                    warmup_opt_events_skipped = len(optimization_events)
                    warmup_dt_events_skipped = len(dt_events)
                    warmup_iter_rates_skipped = len(iteration_rates)
                    warmup_frames_delivered = getattr(source, 'frames_delivered', 0)
                    warmup_read_decode_time_s = getattr(source, 'read_decode_time_total_s', 0.0)
                    warmup_preload_time_s = getattr(source, 'preload_time_total_s', 0.0)

                if global_frame_count % 10 == 0:
                    # Subtract preload time for offline batched mode so wall
                    # fps reflects processing throughput, not the I/O bursts
                    # at every batch boundary.
                    preload_t = getattr(source, 'preload_time_total_s', 0.0)
                    wall_so_far = max(time.perf_counter() - estimation_start_time
                                      - preload_t, 1e-9)
                    wall_fps_so_far = global_frame_count / wall_so_far
                    compute_fps_so_far = global_frame_count / max(total_processing_time, 1e-9)
                    print(f"  Frame {global_frame_count}: wall={wall_fps_so_far:.1f} fps  "
                          f"(compute={compute_fps_so_far:.1f} fps)")

                if next_frame is None:
                    break  # source exhausted
                if args.max_frames and global_frame_count >= args.max_frames:
                    print(f"[Stream] Reached --max-frames={args.max_frames}.")
                    break

                # Promote: prefetched seg launched on the OLD next_stream is now
                # ready on what (after swap_streams() inside the core) is the NEW
                # current_stream — pass it through unchanged.
                current_frame = next_frame
                prefetched_seg = next_prefetched_seg
                next_frame = None
                if not (args.max_frames and global_frame_count + 1 >= args.max_frames):
                    next_frame = _read_next()
        except KeyboardInterrupt:
            print(f"\n[Stream] Interrupted after {global_frame_count} frames.")

    source.close()
    if video_writer is not None:
        video_writer.release()
    if srbag.dual_stream_mgr is not None:
        srbag.dual_stream_mgr.synchronize_all()
    if args.display_progress:
        cv2.destroyAllWindows()

    # Save final estimated pose
    if global_frame_count > 0:
        final_estimated_4x4 = pk.se3_9d_to_matrix44(
            prev_9d.unsqueeze(0) if prev_9d.dim() == 1 else prev_9d)[0]
        H_final_estimated = torch.linalg.inv(final_estimated_4x4).cpu().detach().numpy()
        np.save(os.path.join(per_dir, "final_estimated_pose.npy"), H_final_estimated)

    # Per-frame error CSV
    if all_frame_errors:
        import csv
        csv_path = os.path.join(args.output_dir, "per_frame_pose_error.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'trans_err_mm', 'rot_err_deg'])
            writer.writeheader()
            writer.writerows(all_frame_errors)
        print(f"\n[Pose Error] Saved {len(all_frame_errors)} frames to: {csv_path}")
    if H_final_gt is not None:
        np.save(os.path.join(per_dir, "final_gt_pose.npy"), H_final_gt)

    # Timing summary
    # `wall` is measured by the host clock from before the bootstrap reads to
    # after the loop exits — this is what a stopwatch would see and what
    # matters for downstream consumers (controller, GUI). For offline batched
    # mode we exclude `source.preload_time_total_s` (the bulk read+decode+
    # upload bursts at every batch boundary) so wall fps reflects steady-state
    # processing throughput rather than I/O overhead.
    # `compute_fps` is `frames / total_processing_time`, where
    # `total_processing_time` only accumulates time inside
    # `process_one_frame_dual_stream`. It excludes per-iteration host work:
    # bag/ROS read, cv2 decode, CPU->GPU upload. So `compute_fps` answers
    # "how fast is the GPU pipeline?", `wall_fps` answers "how fast is the
    # whole thing including the per-frame host work?". The gap is small
    # per-frame I/O.
    # All metrics below are computed in steady state — i.e. with the first
    # frame and everything before it (Python init, model load, CUDA graph
    # capture, scene construction, first bag reads, first compute) excluded.
    # `warmup_*` baselines were snapshotted right after the first iteration
    # finished, so subtracting them gives steady-state-only totals.
    steady_frames = max(global_frame_count - 1, 0)
    if steady_frames > 0 and warmup_t_steady_start_s is not None:
        steady_wall_raw = time.perf_counter() - warmup_t_steady_start_s
        steady_preload_t = (getattr(source, 'preload_time_total_s', 0.0)
                            - warmup_preload_time_s)
        steady_wall_time = max(steady_wall_raw - steady_preload_t, 1e-9)
        steady_wall_fps = steady_frames / steady_wall_time
        steady_compute_time = total_processing_time - warmup_total_processing_time_s
        steady_compute_fps = (steady_frames / steady_compute_time
                              if steady_compute_time > 0 else 0)
        steady_io_overhead_ms = (1000.0 * (steady_wall_time - steady_compute_time)
                                 / steady_frames)
    else:
        # Fallback when only the first frame ran (or none): include startup.
        steady_wall_raw = time.perf_counter() - estimation_start_time
        steady_preload_t = getattr(source, 'preload_time_total_s', 0.0)
        steady_wall_time = max(steady_wall_raw - steady_preload_t, 1e-9)
        steady_wall_fps = global_frame_count / steady_wall_time
        steady_compute_time = total_processing_time
        steady_compute_fps = (global_frame_count / total_processing_time
                              if total_processing_time > 0 else 0)
        steady_io_overhead_ms = (1000.0 * (steady_wall_time - total_processing_time)
                                 / max(global_frame_count, 1))
        steady_frames = global_frame_count

    # Per-stage GPU events: drop the warmup-period entries.
    seg_steady = segmentation_events[warmup_seg_events_skipped:]
    opt_steady = optimization_events[warmup_opt_events_skipped:]
    dt_steady = dt_events[warmup_dt_events_skipped:]
    iter_steady = iteration_rates[warmup_iter_rates_skipped:]

    avg_seg, std_seg, seg_fps, _ = srbag._compute_event_stats(seg_steady)
    avg_opt, std_opt, opt_fps, _ = srbag._compute_event_stats(opt_steady)
    avg_dt, std_dt, _, dt_times = srbag._compute_event_stats(dt_steady)
    avg_iter = float(np.mean(iter_steady)) if iter_steady else 0.0
    std_iter = float(np.std(iter_steady)) if iter_steady else 0.0
    seq_time = avg_seg + avg_opt + (avg_dt if dt_times else 0)
    sequential_fps = 1.0 / seq_time if seq_time > 0 else 0
    overlap_speedup = steady_wall_fps / sequential_fps if sequential_fps > 0 else 1.0

    # Compute the steady-period preload value once so we can both gate the
    # preload print and adjust the wall-line label for sources that don't
    # preload (bag, live).
    steady_preload_show = (getattr(source, 'preload_time_total_s', 0.0)
                           - warmup_preload_time_s)
    has_preload = steady_preload_show > 0
    wall_excludes = "excludes warmup + preload" if has_preload else "excludes warmup"

    print(f"\n[Done] Frames={global_frame_count} ({steady_frames} steady-state, 1 warmup excluded)")
    print(f"  wall:    {steady_wall_time:.1f}s  -> {steady_wall_fps:.1f} fps   "
          f"(host clock, {wall_excludes})")
    print(f"  compute: {steady_compute_time:.1f}s  -> {steady_compute_fps:.1f} fps   "
          f"(per_frame_core only, GPU, excludes warmup)")
    if has_preload:
        print(f"  preload: {steady_preload_show:.1f}s "
              f"(batched read+decode+upload, {args.source} mode; excluded from wall)")

    # Source-side timing — only `raw read+decode` is reported. The previous
    # "source playing speed" line was just wall fps in disguise: the source
    # delivers exactly when the loop asks for one, so its delivery span ≡
    # wall span. `raw read+decode` is the genuinely separate measurement —
    # how much time per frame is actually spent inside `_read_impl()`.
    src_delivered = getattr(source, 'frames_delivered', 0)
    src_read_decode_total = getattr(source, 'read_decode_time_total_s', 0.0)
    src_steady_delivered = max(src_delivered - warmup_frames_delivered, 0)
    src_steady_read_decode = max(src_read_decode_total - warmup_read_decode_time_s, 0.0)
    if src_steady_read_decode > 0 and src_steady_delivered > 0:
        src_rd_fps = src_steady_delivered / src_steady_read_decode
        src_rd_ms_per_frame = 1000.0 * src_steady_read_decode / src_steady_delivered
        print(f"  source raw read+decode: {src_rd_fps:.1f} fps "
              f"({src_rd_ms_per_frame:.1f} ms/frame in read())")

    print(f"  I/O overhead: {steady_io_overhead_ms:.1f} ms/frame  "
          f"(read + decode + upload, on the host between iterations)")
    print(f"  seg: {avg_seg*1000:.1f}±{std_seg*1000:.1f} ms -> {seg_fps:.1f} fps")
    print(f"  opt: {avg_opt*1000:.1f}±{std_opt*1000:.1f} ms -> {opt_fps:.1f} fps")
    print(f"  sequential (seg+opt): {sequential_fps:.1f} fps  |  "
          f"observed wall: {steady_wall_fps:.1f} fps  |  "
          f"overlap speedup: {overlap_speedup:.2f}x")
    print(f"  IPCAI inner-iter rate: {avg_iter:.1f}±{std_iter:.1f} it/s")
    if args.source == 'live':
        total_received = getattr(source, 'frame_counter', global_frame_count)
        cam_fps = total_received / max(steady_wall_time, 1e-9)
        bound = ("camera-bound" if cam_fps < sequential_fps * 0.95
                 else "compute-bound" if cam_fps > steady_wall_fps * 1.1
                 else "matched")
        print(f"  [live] camera received: {total_received} frames -> {cam_fps:.1f} fps "
              f"({bound})")
    print(f"[Output] {args.output_dir}")


def main():
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(args.max_jobs)
    run_pipeline(args, device)


if __name__ == "__main__":
    main()