#!/usr/bin/env python3
"""
stereo_pipeline_live_handoff.py

Live-only IPCAI stereo registration pipeline. Subscribes to running
ROS2 image + joint-state topics and runs per-frame segmentation +
differentiable-rendering pose optimisation.

Two init modes for the camera↔base extrinsics:

  --extrinsics-file PATH
    Load a 4x4 .npy directly. The file is in IPCAI's custom frame
    (T_link0_to_cam, as produced by track.py or by fp_pose_recorder's
    saved FP_init.npy).

  --handoff
    Subscribe to /pose_init (geometry_msgs/PoseStamped, latched)
    published by fp_pose_recorder.py in --init_only mode. The recorder
    has already applied the convention conversion before publishing,
    so the received pose is in IPCAI custom frame — used directly,
    same as --extrinsics-file.

The two init options are mutually exclusive; one is required.

Per-frame seg + opt core lives in `stereo_ipcai_pipeline_bag` (imported
as srbag). This script owns only the live-ROS source class, camera-info
fetching, and handoff helpers.
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
        # Subtract any subclass-reported "excluded" time (preload bursts).
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

            # Async H2D on the copy engine; sync so the queue holds
            # GPU-resident tensors.
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
                return  # wait for first joint-state before accepting frames
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
    return LiveSource(
        args.left_topic, args.right_topic, args.joint_state_topic,
        device, slop_ms=args.sync_slop_ms)


# =============================================================================
# Camera-info fetching (from live topics or from a bag)
# =============================================================================
def fetch_camera_info_from_topics(args, timeout_sec=10.0):
    """Wait for one message on each CameraInfo topic; return (left, right).

    Uses a dedicated executor to avoid colliding with concurrent spinners.
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

        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)
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

    return {'img_diagonal': float(np.sqrt(H**2 + W**2)),
            'scene': scene, 'H': H, 'W': W, 'K_left': K_left}


# =============================================================================
# CLI
# =============================================================================
def args_factory():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="IPCAI Stereo Registration Pipeline — live ROS2.")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="Disable seg-of-N+1 / opt-of-N overlap.")
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
                        help="Stop after this many frames (0 = run until Ctrl-C).")
    parser.add_argument("--live-read-timeout", type=float, default=10.0)
    parser.add_argument("--live-latest-only", action="store_true",
                        help="Drop queued older frames and always process the newest. "
                             "Lowest latency for tracking; default is FIFO.")
    # Paths
    parser.add_argument("--output-dir", required=True)
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument("--extrinsics-file", default=None,
                            help="Initial 4x4 .npy in IPCAI custom frame "
                                 "(T_link0_to_cam, as produced by track.py "
                                 "or fp_pose_recorder's saved FP_init.npy).")
    init_group.add_argument("--handoff", action="store_true",
                            help="Subscribe to /pose_init for the initial "
                                 "extrinsics. The recorder publishes the "
                                 "pose already in IPCAI custom frame.")
    parser.add_argument("--pose-init-topic", default="/pose_init",
                        help="(handoff mode) topic for the latched PoseStamped.")
    parser.add_argument("--pose-init-timeout", type=float, default=120.0)
    parser.add_argument("--seg-publish-topic", default="/left/segmentation",
                        help="(handoff mode) topic for IPCAI's seg masks. "
                             "FP's selector subscribes here.")
    parser.add_argument("--seg-image-topic", default="/left/image_rect",
                        help="(handoff mode) image topic the seg-publisher "
                             "consumes to produce masks.")
    parser.add_argument("--seg-debug-dir", default="",
                        help="(handoff mode) if set, save each published "
                             "seg mask + corresponding input image to this "
                             "directory for inspection. Filenames are "
                             "frame_NNNNNN_image.png and frame_NNNNNN_mask.png.")
    parser.add_argument("--right-extrinsics-file", required=True)
    parser.add_argument("--final-extrinsics-file", default=None,
                        help="Optional final pose GT (.npy). Per-frame trans/rot "
                             "error against this GT is saved to per_frame_pose_error.csv.")
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
    p.optimizer = "ipcai"
    return p


# =============================================================================
# Streaming loop — same shape as the bag-pipeline's stream-mode branch
# =============================================================================
def run_pipeline(args, device):
    """Live-stream seg + opt via srbag.process_one_frame_dual_stream."""
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
    print(f"\n[Pipeline] IPCAI live | lr={args.ipcai_lr} "
          f"| iter={args.max_iterations} | {loss_str}")

    # Pre-load seg model in handoff mode so it's ready before bag streams.
    if args.handoff:
        print("[Handoff] Loading segmentation model (pre-bag)...")
        if srbag.coarse_model is None or srbag.seg_graph is None:
            srbag.init_seg_model_and_graph(args, device)
            srbag.init_processing_state(args, device)
        print("[Handoff] Seg model ready.")

    # Long caminfo timeout in handoff: the bag is gated until user touches it.
    caminfo_timeout = 300.0 if args.handoff else 10.0
    caminfo_msgs = fetch_camera_info_from_topics(args, timeout_sec=caminfo_timeout)

    resources = init_pipeline_resources(args, device, caminfo_msgs=caminfo_msgs)
    img_diagonal, scene = resources['img_diagonal'], resources['scene']

    # Subscribe to image+joint streams before the handoff wait so the
    # source is already buffering frames when /pose_init arrives.
    source = make_source(args, device)

    H_init_handoff = None
    if args.handoff:
        print("[Handoff] IPCAI fully initialised. Waiting for the bag "
              "to start (manual gate) and then for /pose_init...")
        H_init_handoff = do_handoff_seg_and_wait(args, device)

    # H_init is the camera-in-base pose (IPCAI custom frame). Invert to
    # the base-in-camera pose H_b2l used by the optimiser.
    if H_init_handoff is not None:
        H_init = H_init_handoff
    else:
        H_init = np.load(args.extrinsics_file)
        if H_init.ndim > 2:
            H_init = H_init.reshape(4, 4)
        print(f"[Init] Loaded extrinsics from {args.extrinsics_file}: "
              f"t={H_init[:3, 3]}")
    H_b2l = np.linalg.inv(H_init)
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

    K_left = resources.get('K_left')
    if K_left is not None and (args.display_progress or args.save_frames or args.save_video):
        print(f"[Viz] K_left loaded: fx={K_left[0,0]:.1f} fy={K_left[1,1]:.1f}")

    print("[Estimation] streaming live frame-by-frame, forward")

    estimation_start_time = time.perf_counter()
    global_frame_count = 0
    total_processing_time = 0.0

    warmup_t_steady_start_s = None
    warmup_total_processing_time_s = 0.0
    warmup_seg_events_skipped = 0
    warmup_opt_events_skipped = 0
    warmup_dt_events_skipped = 0
    warmup_iter_rates_skipped = 0
    warmup_frames_delivered = 0
    warmup_read_decode_time_s = 0.0
    warmup_preload_time_s = 0.0

    def _read_next():
        return source.read(timeout_sec=args.live_read_timeout,
                           latest_only=args.live_latest_only)

    # Bootstrap: prime two frames so the first iteration can overlap
    # next-frame seg with current-frame opt.
    current_frame = _read_next()
    if current_frame is not None:
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

    # Timing summary. `wall` = host stopwatch (excluding offline preload bursts).
    # `compute_fps` = frames / time spent in process_one_frame_dual_stream.
    # All metrics in steady state (first frame + warm-up excluded).
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
              f"(batched read+decode+upload; excluded from wall)")

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