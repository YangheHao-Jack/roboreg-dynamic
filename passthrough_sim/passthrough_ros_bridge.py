#!/usr/bin/env python3
"""passthrough_ros_bridge.py — slim ROS2 publisher sidecar for the
passthrough_sim producer.

Runs under the system Python (3.12 on Ubuntu 24) so it can import the ROS2
Jazzy rclpy bindings the Isaac Sim 3.11 env can't. test_cloudxr.py auto-spawns
it and feeds capture records over a UNIX socket; this script publishes them.

This is the slimmed passthrough version of xr_ros_bridge.py. The UNIX-socket
WIRE PROTOCOL is byte-for-byte identical to the original (so the producer's
RosBridge sender is unchanged), but it publishes ONLY the topics the
passthrough pipeline consumes:

    <ns>/image_left/compressed   <ns>/image_right/compressed   (CompressedImage)
    <ns>/image_left/camera_info  <ns>/image_right/camera_info  (CameraInfo)
    <ns>/joint_states                                          (JointState)
    <ns>/baseline                                              (PoseStamped)

Dropped vs. the original (nothing downstream subscribes to them): the
world-frame eye/base poses (pose_left/right/base) and the per-eye-in-base
extrinsics (extrinsic_left/right). The stereo baseline is still computed every
frame because it sets the right camera_info P[0,3] = -fx_R * |baseline|, which
ESS / stereo_depth_saver / FoundationPose need for non-zero depth.

The bridge treats image bytes as opaque; CompressedImage.format is stamped
from --image-codec ("h264"; the producer encodes NVENC H.264). Run is automatic via test_cloudxr.py
--ros-publish*; to launch by hand:
    python3.12 passthrough_ros_bridge.py --socket /tmp/xr_ros_bridge.sock \
        --namespace /xr --joint-names lbr_A1,lbr_A2,lbr_A3,lbr_A4,lbr_A5,lbr_A6,lbr_A7
"""

import argparse
import array
import math
import os
import socket
import struct
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CompressedImage, CameraInfo, JointState, Image
from geometry_msgs.msg import PoseStamped

# Wire protocol — MUST match test_cloudxr.py's RosBridge sender exactly.
MAGIC            = 0x42525258
VERSION          = 1
TYPE_CAMERA_INFO = 1
TYPE_CAPTURE     = 2
TYPE_GOODBYE     = 3
HEADER_FMT       = "<IIII"
HEADER_SIZE      = struct.calcsize(HEADER_FMT)
CAMERA_INFO_FMT  = "<II4fII4f"
CAMERA_INFO_SIZE = struct.calcsize(CAMERA_INFO_FMT)
CAPTURE_PRE_FMT  = "<Id7f16f16f16f"        # idx, ts, 7 joints, H_base, H_eye_L, H_eye_R
CAPTURE_PRE_SIZE = struct.calcsize(CAPTURE_PRE_FMT)


def recv_exact(sock, n):
    """Read exactly n bytes or return None on EOF."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        m = sock.recv_into(view[got:])
        if m == 0:
            return None
        got += m
    return bytes(buf)


def se3_inv(H):
    """Inverse of a 4x4 SE(3) (row-major flat 16-tuple). Pure Python."""
    R00, R01, R02, tx = H[0],  H[1],  H[2],  H[3]
    R10, R11, R12, ty = H[4],  H[5],  H[6],  H[7]
    R20, R21, R22, tz = H[8],  H[9],  H[10], H[11]
    itx = -(R00*tx + R10*ty + R20*tz)
    ity = -(R01*tx + R11*ty + R21*tz)
    itz = -(R02*tx + R12*ty + R22*tz)
    return (R00, R10, R20, itx,
            R01, R11, R21, ity,
            R02, R12, R22, itz,
            0.0, 0.0, 0.0, 1.0)


def se3_compose(A, B):
    """A @ B for two row-major-flat 4x4 SE(3) transforms."""
    out = [0.0] * 16
    for i in range(4):
        ai0, ai1, ai2, ai3 = A[i*4], A[i*4+1], A[i*4+2], A[i*4+3]
        for j in range(4):
            out[i*4+j] = (ai0 * B[j] + ai1 * B[4 + j]
                          + ai2 * B[8 + j] + ai3 * B[12 + j])
    return tuple(out)


def rot_to_quat_xyzw(R):
    """3x3 rotation (row-major 9-tuple) -> ROS quaternion (qx,qy,qz,qw)."""
    tr = R[0] + R[4] + R[8]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[7] - R[5]) / s
        qy = (R[2] - R[6]) / s
        qz = (R[3] - R[1]) / s
    elif R[0] > R[4] and R[0] > R[8]:
        s = math.sqrt(1.0 + R[0] - R[4] - R[8]) * 2
        qw = (R[7] - R[5]) / s
        qx = 0.25 * s
        qy = (R[1] + R[3]) / s
        qz = (R[2] + R[6]) / s
    elif R[4] > R[8]:
        s = math.sqrt(1.0 + R[4] - R[0] - R[8]) * 2
        qw = (R[2] - R[6]) / s
        qx = (R[1] + R[3]) / s
        qy = 0.25 * s
        qz = (R[5] + R[7]) / s
    else:
        s = math.sqrt(1.0 + R[8] - R[0] - R[4]) * 2
        qw = (R[3] - R[1]) / s
        qx = (R[2] + R[6]) / s
        qy = (R[5] + R[7]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


class BridgeNode(Node):
    def __init__(self, namespace: str, frame_world: str, joint_names: list,
                 image_codec: str = "h264"):
        super().__init__("passthrough_capture_bridge")
        self.frame_cam_l = "camera_left"
        self.frame_cam_r = "camera_right"
        self.joint_names = joint_names
        self.image_codec = image_codec
        # TEST: raw rgb8 Image instead of JPEG CompressedImage (set by the
        # producer's --raw-images, inherited via env). Dims come from the
        # cached CameraInfo, which the producer sends before any capture.
        self.raw_images = os.environ.get("XR_RAW_IMAGES", "").startswith("1")
        self.get_logger().info(
            f"image codec = {self.image_codec} "
            f"(CompressedImage.format will be set to this)")

        ns = namespace.rstrip("/") or "/xr"
        qos = QoSProfile(depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE)

        # Raw rgb8 frames are ~22 MB each. RELIABLE + deep history makes the
        # publisher block on a busy subscriber (stalls the producer) and buffer
        # stale frames. For the live latest-only pipeline we want freshest-frame,
        # drop-the-rest: BEST_EFFORT + depth 1. The matching rectifier
        # subscription must use the same (see passthrough_rectifier --raw-input).
        img_qos = QoSProfile(depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE) if self.raw_images else qos

        if self.raw_images:
            self.pub_img_l = self.create_publisher(
                Image, f"{ns}/image_left", img_qos)
            self.pub_img_r = self.create_publisher(
                Image, f"{ns}/image_right", img_qos)
            self.get_logger().info(
                f"RAW image mode: rgb8 Image on {ns}/image_left|image_right "
                f"(no compression, BEST_EFFORT depth=1; run rectifier "
                f"with --raw-input)")
        else:
            self.pub_img_l    = self.create_publisher(
                CompressedImage, f"{ns}/image_left/compressed", qos)
            self.pub_img_r    = self.create_publisher(
                CompressedImage, f"{ns}/image_right/compressed", qos)
        self.pub_info_l   = self.create_publisher(
            CameraInfo, f"{ns}/image_left/camera_info", qos)
        self.pub_info_r   = self.create_publisher(
            CameraInfo, f"{ns}/image_right/camera_info", qos)
        self.pub_joints   = self.create_publisher(
            JointState, f"{ns}/joint_states", qos)
        self.pub_baseline = self.create_publisher(
            PoseStamped, f"{ns}/baseline", qos)
        self.get_logger().info(f"publishers up under {ns}/ (passthrough set)")

        self._published = 0
        self._caminfo_msg_l = None
        self._caminfo_msg_r = None

    def publish_camera_info(self, payload: bytes):
        (w_L, h_L, fx_L, fy_L, cx_L, cy_L,
         w_R, h_R, fx_R, fy_R, cx_R, cy_R) = struct.unpack(CAMERA_INFO_FMT, payload)
        built = []
        for frame, w, h, fx, fy, cx, cy in (
            (self.frame_cam_l, w_L, h_L, fx_L, fy_L, cx_L, cy_L),
            (self.frame_cam_r, w_R, h_R, fx_R, fy_R, cx_R, cy_R),
        ):
            msg = CameraInfo()
            msg.header.frame_id = frame
            msg.height = int(h); msg.width = int(w)
            msg.distortion_model = "plumb_bob"
            msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            msg.k = [float(fx), 0.0, float(cx),
                     0.0, float(fy), float(cy),
                     0.0, 0.0, 1.0]
            msg.r = [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0]
            msg.p = [float(fx), 0.0, float(cx), 0.0,
                     0.0, float(fy), float(cy), 0.0,
                     0.0, 0.0, 1.0, 0.0]
            built.append(msg)
        first_build = self._caminfo_msg_l is None
        self._caminfo_msg_l, self._caminfo_msg_r = built
        self.get_logger().info(
            "CameraInfo built + cached — republished per frame" if first_build
            else "CameraInfo cache refreshed")

    def publish_capture(self, payload: bytes):
        head = struct.unpack_from(CAPTURE_PRE_FMT, payload, 0)
        idx       = head[0]
        joint_rad = head[2:9]
        # head[9:25] is H_base (unused here); eye poses drive the baseline.
        H_eye_L   = head[25:41]
        H_eye_R   = head[41:57]
        offset = CAPTURE_PRE_SIZE
        enc_l_size = struct.unpack_from("<I", payload, offset)[0]; offset += 4
        enc_l = payload[offset:offset + enc_l_size]; offset += enc_l_size
        enc_r_size = struct.unpack_from("<I", payload, offset)[0]; offset += 4
        enc_r = payload[offset:offset + enc_r_size]

        stamp = self.get_clock().now().to_msg()

        # Images
        if self.raw_images:
            # Dims from cached CameraInfo (sent before captures). Validate the
            # payload length so a render/caminfo size mismatch is loud, not
            # silent garbage.
            if self._caminfo_msg_l is None or self._caminfo_msg_r is None:
                return                                # dims not known yet
            for pub, frame, data, cinfo in (
                (self.pub_img_l, self.frame_cam_l, enc_l, self._caminfo_msg_l),
                (self.pub_img_r, self.frame_cam_r, enc_r, self._caminfo_msg_r),
            ):
                w = int(cinfo.width); h = int(cinfo.height)
                if len(data) != w * h * 3:
                    self.get_logger().warn(
                        f"raw: payload {len(data)}B != {w}x{h}x3 "
                        f"({w*h*3}B) — render res != camera_info res; skipping")
                    continue
                im = Image()
                im.header.stamp = stamp
                im.header.frame_id = frame
                im.height = h
                im.width = w
                im.encoding = "rgb8"
                im.is_bigendian = 0
                im.step = w * 3
                im.data = array.array("B", data)
                pub.publish(im)
        else:
            img_l = CompressedImage()
            img_l.header.stamp = stamp
            img_l.header.frame_id = self.frame_cam_l
            img_l.format = self.image_codec
            img_l.data = array.array("B", enc_l)
            self.pub_img_l.publish(img_l)

            img_r = CompressedImage()
            img_r.header.stamp = stamp
            img_r.header.frame_id = self.frame_cam_r
            img_r.format = self.image_codec
            img_r.data = array.array("B", enc_r)
            self.pub_img_r.publish(img_r)

        # Stereo baseline (right eye in left-eye frame). In this OpenXR /
        # Omniverse frame the eye offset is along Y (index 7), not X. |Y| is
        # the baseline magnitude; right camera_info P[0,3] = -fx_R * |baseline|
        # so ESS / depth_saver / FP read a positive baseline (T = -P[0,3]/fx).
        H_inv_eye_L = se3_inv(H_eye_L)
        H_r_in_l    = se3_compose(H_inv_eye_L, H_eye_R)
        B_m         = abs(float(H_r_in_l[7]))

        # CameraInfo (re-emit cached, current stamp; VOLATILE QoS so the
        # per-frame republish is what downstream / a bag actually receives).
        if self._caminfo_msg_l is not None:
            self._caminfo_msg_l.header.stamp = stamp
            self._caminfo_msg_r.header.stamp = stamp
            fx_R = float(self._caminfo_msg_r.k[0])
            self._caminfo_msg_r.p[3] = -fx_R * B_m
            self.pub_info_l.publish(self._caminfo_msg_l)
            self.pub_info_r.publish(self._caminfo_msg_r)
            if self._published == 0:
                self.get_logger().info(
                    f"First per-frame camera_info republished "
                    f"(baseline={B_m*1000:.1f} mm, "
                    f"P_right[0,3]={self._caminfo_msg_r.p[3]:.2f})")
        elif self._published == 0:
            self.get_logger().warning(
                "publish_capture: camera_info not yet received from the "
                "layer — downstream will stall until the CAMERA_INFO record "
                "arrives.")

        # JointState
        js = JointState()
        js.header.stamp = stamp
        js.name = self.joint_names
        js.position = [float(v) for v in joint_rad]
        self.pub_joints.publish(js)

        # Baseline as PoseStamped (right-in-left), for the rectifier.
        bl = PoseStamped()
        bl.header.stamp = stamp
        bl.header.frame_id = self.frame_cam_l
        bl.pose.position.x = float(H_r_in_l[3])
        bl.pose.position.y = float(H_r_in_l[7])
        bl.pose.position.z = float(H_r_in_l[11])
        R = (H_r_in_l[0], H_r_in_l[1], H_r_in_l[2],
             H_r_in_l[4], H_r_in_l[5], H_r_in_l[6],
             H_r_in_l[8], H_r_in_l[9], H_r_in_l[10])
        qx, qy, qz, qw = rot_to_quat_xyzw(R)
        bl.pose.orientation.x = qx
        bl.pose.orientation.y = qy
        bl.pose.orientation.z = qz
        bl.pose.orientation.w = qw
        self.pub_baseline.publish(bl)

        self._published += 1
        if self._published == 1 or self._published % 30 == 0:
            self.get_logger().info(
                f"captured #{idx:04d} -> topics ({self._published} total)")


def serve(sock_path: str, node: BridgeNode):
    """Bind, accept one client, publish records until EOF/goodbye."""
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)
    print(f"[passthrough_ros_bridge] listening on {sock_path}", flush=True)
    try:
        client, _ = server.accept()
        print("[passthrough_ros_bridge] client connected", flush=True)
        while True:
            header = recv_exact(client, HEADER_SIZE)
            if header is None:
                print("[passthrough_ros_bridge] client disconnected", flush=True)
                return
            magic, version, rtype, psize = struct.unpack(HEADER_FMT, header)
            if magic != MAGIC:
                print(f"[passthrough_ros_bridge] bad magic 0x{magic:08x}", flush=True)
                return
            if version != VERSION:
                print(f"[passthrough_ros_bridge] unsupported version {version}",
                      flush=True)
                return
            payload = recv_exact(client, psize) if psize else b""
            if payload is None and psize:
                return
            if rtype == TYPE_CAMERA_INFO:
                node.publish_camera_info(payload)
            elif rtype == TYPE_CAPTURE:
                node.publish_capture(payload)
            elif rtype == TYPE_GOODBYE:
                print("[passthrough_ros_bridge] goodbye received", flush=True)
                return
            else:
                print(f"[passthrough_ros_bridge] unknown record_type {rtype}",
                      flush=True)
    finally:
        try: client.close()
        except Exception: pass
        try: server.close()
        except Exception: pass
        try: os.unlink(sock_path)
        except FileNotFoundError: pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--socket", required=True,
                    help="UNIX socket path to bind/listen on")
    ap.add_argument("--namespace", default="/xr",
                    help="ROS2 topic namespace (default /xr)")
    ap.add_argument("--frame", default="world",
                    help="kept for CLI compatibility with the producer spawn; "
                         "unused (no world-frame poses are published here)")
    ap.add_argument("--joint-names", default="lbr_A1,lbr_A2,lbr_A3,lbr_A4,lbr_A5,lbr_A6,lbr_A7",
                    help="comma-separated joint names for JointState")
    ap.add_argument("--image-codec", default="h264", choices=["h264"],
                    help="Stamped into CompressedImage.format (opaque payload).")
    args = ap.parse_args()

    rclpy.init(args=None)
    joint_names = [n.strip() for n in args.joint_names.split(",") if n.strip()]
    node = BridgeNode(args.namespace, args.frame, joint_names,
                      image_codec=args.image_codec)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="rclpy-spin",
                                   daemon=True)
    spin_thread.start()

    exit_code = 0
    try:
        serve(args.socket, node)
    except KeyboardInterrupt:
        print("[passthrough_ros_bridge] interrupted", flush=True)
    except Exception as e:
        print(f"[passthrough_ros_bridge] error: {e}", flush=True)
        import traceback; traceback.print_exc()
        exit_code = 1
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()