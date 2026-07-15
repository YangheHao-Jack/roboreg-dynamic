#!/usr/bin/env python3
"""
bag_h264_source.py --- replay an H.264 ros2 bag as a live source for the
passthrough rectifier + IPCAI consumer, to benchmark inference speed.

This is the bag-fed analog of the CloudXR receiver: it owns one persistent
NVDEC stream per eye (GpuH264Decoder, the same class the receiver/rectifier
uses), decodes the bag's H.264 CompressedImage stream sequentially (H.264 is
inter-frame, so it MUST be decoded in order from the first IDR --- you cannot
decode per-message like JPEG), and republishes:

    /zed/.../left/image_compressed  --(NVDEC)-->  raw rgb8 Image  -> rectifier --raw-input
    /zed/.../right/image_compressed --(NVDEC)-->  raw rgb8 Image
    /zed/.../left|right/camera_info  ----pass-through---->  rectifier caminfo in
    (derived from right P matrix)    ----stereo baseline-->  /xr/baseline (PoseStamped)
    /lbr/joint_states                ----remap----------->  /xr/joint_states

--republish-compressed (deployment-faithful mode): forward the bag's H.264
CompressedImage UNCHANGED so the RECTIFIER's NVDEC does the decode --- the
exact production path. Because H.264 is stateful, this mode (a) gates playback
until the rectifier's subscriptions are matched (the bag's first AU is the only
one guaranteed to carry SPS/PPS+IDR with libx264 bags, and VOLATILE pubs keep
no history), (b) supports --freeze-until-pose-init by repeating the first
headers+IDR access unit (a legal, self-contained stream that both locks NVDEC
and holds a stable picture for the FP handoff), and (c) republishes the stereo
baseline with every caminfo (a late rectifier's VOLATILE sub never sees
TRANSIENT_LOCAL history).

Why raw (not re-JPEG): re-encoding would put nvJPEG back on the CUDA cores and
contend with the optimiser (the decode-once-GPU result). NVDEC decode here runs
on the dedicated video engine, so it does not steal optimiser compute --- the
consumer's measured seg/opt stays honest. The rectifier then runs exactly as in
production (--raw-input), so the consumer sees its real /left/image_rect input.

Downstream (unchanged):
    ros2 run ... passthrough_rectifier.py --raw-input \
        --left-image-topic /xr/image_left --right-image-topic /xr/image_right \
        --left-caminfo-topic /xr/image_left/camera_info \
        --right-caminfo-topic /xr/image_right/camera_info \
        --extrinsics-topic /xr/baseline
    python3 passthrough_consumer.py ... --optimiser-res downsampled \
        --extrinsics-file <seed.npy>   (joints arrive on /xr/joint_states)
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
from rclpy.serialization import deserialize_message

import rosbag2_py
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped

import torch  # noqa: F401  (ensures the primary CUDA context exists for NVDEC)
from gpu_h264_codec import GpuH264Decoder


# Image QoS must match the rectifier's raw-input subscription (the bridge's raw
# publishers): BEST_EFFORT, depth 1, VOLATILE --- latest frame wins, no backlog.
IMG_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST, depth=1)

# caminfo + baseline: RELIABLE + TRANSIENT_LOCAL so a late-joining rectifier
# still gets the one-shot templates it needs to build the rectify maps.
LATCH_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

# joints: RELIABLE depth 10, matching the consumer's JointState subscription.
JS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=10)

# --republish-compressed QoS must match the rectifier's COMPRESSED-image
# subscription, which is RELIABLE depth-10 (NOT the BEST_EFFORT depth-1 it uses
# for --raw-input, rectifier L153-157/213-218). A BEST_EFFORT publisher into a
# RELIABLE subscriber is an INCOMPATIBLE pair -> the rectifier binds nothing and
# sees zero frames. RELIABLE also suits H.264 inter-frame (a dropped frame would
# corrupt until the next IDR).
CMP_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST, depth=10)


def _open_bag(uri):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id="mcap"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                            output_serialization_format="cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, types


class BagH264Source(Node):
    def __init__(self, args):
        super().__init__("bag_h264_source")
        self.args = args
        self.device = args.device

        # One NVDEC stream per eye --- persistent, fed in order from the IDR.
        # Skipped entirely in --republish-compressed: the rectifier decodes, so
        # this source never touches NVDEC (the decode moves to the rectifier
        # process, matching the live --image-codec h264 downsampled path).
        self.dec_l = self.dec_r = None
        if not args.republish_compressed:
            self.dec_l = GpuH264Decoder(device=args.device, gpuid=args.gpuid)
            self.dec_r = GpuH264Decoder(device=args.device, gpuid=args.gpuid)

        self.pub_img_l = self.create_publisher(Image, args.out_left_image, IMG_QOS)
        self.pub_img_r = self.create_publisher(Image, args.out_right_image, IMG_QOS)
        # --republish-compressed: forward the bag's H.264 CompressedImage UNCHANGED
        # on <out>/compressed so the rectifier (--image-codec h264, NO --raw-input)
        # does the NVDEC decode. Uses CMP_QOS (RELIABLE depth-10) to match the
        # rectifier's compressed subscription -- BEST_EFFORT here would bind nothing.
        self.pub_cmp_l = self.pub_cmp_r = None
        if args.republish_compressed:
            self.pub_cmp_l = self.create_publisher(
                CompressedImage, args.out_left_image + "/compressed", CMP_QOS)
            self.pub_cmp_r = self.create_publisher(
                CompressedImage, args.out_right_image + "/compressed", CMP_QOS)
        self.pub_ci_l = self.create_publisher(CameraInfo, args.out_left_caminfo, LATCH_QOS)
        self.pub_ci_r = self.create_publisher(CameraInfo, args.out_right_caminfo, LATCH_QOS)
        self.pub_base = self.create_publisher(PoseStamped, args.out_baseline, LATCH_QOS)
        self.pub_js = self.create_publisher(JointState, args.out_joints, JS_QOS)

        self._baseline_done = False
        self._n_l = self._n_r = self._n_js = 0
        self._warm_l = self._warm_r = False

        # Re-freeze on demand (/seed/arm, latched Bool): during playback,
        # arming re-enters a freeze on the MOST RECENT IDR access unit per eye
        # (cached continuously below — the stream carries an IDR every ~0.5 s,
        # so the frozen picture is effectively "now"). FP gets the stable view
        # a re-initialisation needs; a fresh /pose_init releases, and playback
        # resumes with stamps/pacing shifted by the frozen duration.
        self._refreeze_req = False
        self._armed_state = False
        self._refreeze_au = {True: None, False: None}   # is_left -> headers+IDR
        self._refreeze_hdr = {True: b"", False: b""}    # cached SPS/PPS per eye
        try:
            from std_msgs.msg import Bool
            arm_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(Bool, '/seed/arm', self._on_arm, arm_qos)
            # Freeze-state broadcast: the consumer pauses estimation while
            # this is true, binding its pause to the ACTUAL stream state
            # instead of racing the arm signal.
            self._frozen_pub = self.create_publisher(Bool, '/seed/frozen',
                                                     arm_qos)
        except Exception as e:
            self.get_logger().warn(f"/seed/arm subscription unavailable: {e}")

        # Freeze gate: hold the first frame until /pose_init latches (FP needs a
        # stable view to seed; we must not stream past it before the handoff).
        self._pose_init_seen = False
        if args.freeze_until_pose_init:
            pi_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,  # catch the latched seed
                history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(PoseStamped, args.pose_init_topic,
                                     self._on_pose_init, pi_qos)

        self.get_logger().info(
            f"bag='{args.bag}' rate={args.rate} "
            f"-> img L='{args.out_left_image}' R='{args.out_right_image}' "
            f"(raw rgb8), joints->'{args.out_joints}', baseline->'{args.out_baseline}'")

    def _on_pose_init(self, _msg):
        if not self._pose_init_seen:
            self._pose_init_seen = True
            self.get_logger().info("/pose_init received --- releasing freeze")

    def _on_arm(self, msg):
        want = bool(msg.data)
        self._armed_state = want
        if want and not self._refreeze_req:
            self._refreeze_req = True
            self.get_logger().info(
                "/seed/arm --- will RE-FREEZE on the latest IDR for "
                "re-initialisation")
        elif not want and self._refreeze_req:
            self._refreeze_req = False
            self.get_logger().info(
                "/seed/arm cleared --- pending re-freeze cancelled")

    def _cache_refreeze_au(self, is_left, payload):
        """Track headers + the newest self-contained IDR AU per eye."""
        types = self._nal_types(payload)
        self._refreeze_hdr[is_left] += self._split_header_nals(payload)
        if 5 in types:
            self._refreeze_au[is_left] = (
                payload if {7, 8} & types
                else self._refreeze_hdr[is_left] + payload)

    def _refreeze_phase(self, last_stamp_ns):
        """Hold the cached latest-IDR AUs at freeze_hz until a FRESH
        /pose_init arrives (the existing latched original cannot re-fire this
        subscription — transient_local redelivers only to NEW subscriptions).
        Returns the frozen duration in ns so run() shifts stamps + pacing."""
        a = self.args
        if self._refreeze_au[True] is None or self._refreeze_au[False] is None:
            if not getattr(self, '_refreeze_wait_logged', False):
                self._refreeze_wait_logged = True
                self.get_logger().info(
                    "re-freeze requested before an IDR was cached --- "
                    "will freeze at the next IDR (<= one GOP away)")
            return 0                       # request stays pending; retried
        self._refreeze_req = False
        self._refreeze_wait_logged = False
        self._pose_init_seen = False          # re-arm the release gate
        from std_msgs.msg import Bool
        _b = Bool(); _b.data = True
        self._frozen_pub.publish(_b)
        self.get_logger().info(
            f"RE-FREEZE: holding latest IDR AU per eye at "
            f"{a.freeze_hz:.0f} Hz until a fresh '{a.pose_init_topic}' ...")
        hold_dt = 1.0 / max(a.freeze_hz, 1.0)
        rel_ns = 0
        self._armed_state = True
        while (rclpy.ok() and not self._pose_init_seen
               and self._armed_state):
            stamp = Time(nanoseconds=last_stamp_ns + rel_ns).to_msg()
            for is_left, pub in ((True, self.pub_cmp_l),
                                 (False, self.pub_cmp_r)):
                m = CompressedImage()
                m.header.stamp = stamp
                m.header.frame_id = (a.left_frame_id if is_left
                                     else a.right_frame_id)
                m.format = "h264"
                m.data = self._refreeze_au[is_left]
                pub.publish(m)
            rel_ns += int(hold_dt * 1e9)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(hold_dt)
        _b = Bool(); _b.data = False
        self._frozen_pub.publish(_b)
        why = ("fresh /pose_init" if self._pose_init_seen
               else "disarmed (aborted)")
        self.get_logger().info(
            f"RE-FREEZE released after {rel_ns/1e9:.1f}s ({why}) "
            f"--- resuming playback")
        return rel_ns

    # ---- decode + publish one eye --------------------------------------------
    def _decode_hwc(self, dec, payload):
        """H.264 payload -> (H,W,3) uint8 RGB host array, or None on warmup."""
        rgb_chw = dec.decode(bytes(payload))     # (3,H,W) uint8 cuda, or None
        if rgb_chw is None:
            return None
        return rgb_chw.permute(1, 2, 0).contiguous().to("cpu").numpy()

    def _publish_raw(self, pub, hwc, stamp_msg, cam):
        h, w = int(hwc.shape[0]), int(hwc.shape[1])
        msg = Image()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = (self.args.left_frame_id if cam == "L"
                               else self.args.right_frame_id)
        msg.height, msg.width = h, w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = hwc.tobytes()
        pub.publish(msg)

    def _emit_image(self, dec, pub, payload, stamp_msg, cam):
        hwc = self._decode_hwc(dec, payload)
        if hwc is None:
            return False                          # NVDEC parser warmup
        self._publish_raw(pub, hwc, stamp_msg, cam)
        return True

    def _maybe_publish_baseline(self, ci_right):
        """Right-eye-in-left-eye PoseStamped from the rectified right P matrix:
        P[3] = -fx * baseline  ->  baseline = -P[3] / P[0]. Pure x translation
        (ZED images are rectified). Flip the sign if rectification comes out
        with vertical disparity / mirrored."""
        fx = float(ci_right.p[0])
        tx = float(ci_right.p[3])
        if fx == 0.0:
            return
        baseline = -tx / fx
        ps = PoseStamped()
        ps.header.frame_id = self.args.left_frame_id
        ps.pose.position.x = baseline
        ps.pose.orientation.w = 1.0
        # Publish on EVERY right caminfo, not once: the latched TRANSIENT_LOCAL
        # history is invisible to the rectifier's VOLATILE subscription, so a
        # rectifier that starts after the one-shot (the launch-orchestrated
        # case: bake -> +4s rectifier, while this source streams from t=0)
        # would never get extrinsics -> never build grids -> publish NOTHING.
        # The rectifier destroys its baseline sub after the first receipt, so
        # repeating is free.
        self.pub_base.publish(ps)
        if not self._baseline_done:
            self._baseline_done = True
            self.get_logger().info(f"publishing stereo baseline = {baseline*1000:.1f} mm "
                                   f"on '{self.args.out_baseline}' (repeats with caminfo)")

    # ---- Annex-B helpers (compressed mode) -----------------------------------
    @staticmethod
    def _nal_types(payload: bytes):
        """NAL unit types in an Annex-B access unit. Empty set => no start
        codes found, i.e. the payload is NOT Annex-B (e.g. AVCC length-
        prefixed) and NVDEC's parser will never sync on it."""
        types = set()
        b = bytes(payload)
        i, n = 0, len(b)
        while i + 3 < n:
            if b[i] == 0 and b[i+1] == 0 and (
                    b[i+2] == 1 or (b[i+2] == 0 and i + 4 < n and b[i+3] == 1)):
                j = i + (3 if b[i+2] == 1 else 4)
                if j < n:
                    types.add(b[j] & 0x1F)
                i = j
            else:
                i += 1
        return types

    @classmethod
    def _split_header_nals(cls, payload: bytes):
        """Return the concatenated SPS(7)/PPS(8) NAL bytes from an Annex-B AU
        (with start codes), or b'' if none present."""
        b = bytes(payload)
        out = bytearray()
        i, n = 0, len(b)
        starts = []                       # (start_code_pos, nal_payload_pos)
        while i + 3 < n:
            if b[i] == 0 and b[i+1] == 0 and (
                    b[i+2] == 1 or (b[i+2] == 0 and i + 4 < n and b[i+3] == 1)):
                j = i + (3 if b[i+2] == 1 else 4)
                starts.append((i, j))
                i = j
            else:
                i += 1
        for k, (sc, p) in enumerate(starts):
            end = starts[k+1][0] if k + 1 < len(starts) else n
            if p < n and (b[p] & 0x1F) in (7, 8):
                out += b[sc:end]
        return bytes(out)

    def _wait_for_compressed_subscribers(self, synth_base_ns):
        """Block until the rectifier's compressed subscriptions are matched,
        WHILE publishing the bag's first joints + caminfo + baseline at ~10 Hz.
        The publishing is not optional: under the launch orchestration the
        rectifier only starts after bake exits, and bake exits only after it
        receives joint_states --- a silent gate would deadlock the whole
        launch (bake waits for joints, gate waits for rectifier, rectifier
        waits for bake). Returns the advanced synthetic-clock base.

        Why gate at all: the bag's first AU is the only one guaranteed to
        carry SPS/PPS+IDR with libx264 bags, and VOLATILE pubs keep no
        history --- starting playback before the rectifier subscribes means
        NVDEC never sees stream headers and never locks on."""
        a = self.args
        reader, _ = _open_bag(a.bag)
        ci_l = ci_r = first_js = None
        while reader.has_next() and rclpy.ok():
            topic, data, _t = reader.read_next()
            if topic == a.in_left_caminfo and ci_l is None:
                ci_l = deserialize_message(data, CameraInfo)
            elif topic == a.in_right_caminfo and ci_r is None:
                ci_r = deserialize_message(data, CameraInfo)
            elif topic == a.in_joints and first_js is None:
                first_js = deserialize_message(data, JointState)
            if ci_l and ci_r and first_js:
                break
        del reader

        t0 = time.perf_counter()
        last = 0.0
        rel_ns = 0
        hold_dt = 0.1
        while rclpy.ok():
            nl = self.pub_cmp_l.get_subscription_count()
            nr = self.pub_cmp_r.get_subscription_count()
            if nl >= 1 and nr >= 1:
                self.get_logger().info(
                    f"compressed subscribers matched (L={nl} R={nr}) after "
                    f"{time.perf_counter()-t0:.1f}s --- starting playback from "
                    f"the bag's first AU (carries SPS/PPS+IDR)")
                # Re-base to NOW: rel_ns under-counts wall time by the
                # per-tick publish work, and carrying that lag forward shows
                # up as a constant bogus offset in the consumer's 'wire'
                # latency column.
                return self.get_clock().now().nanoseconds + 33_000_000
            stamp = Time(nanoseconds=synth_base_ns + rel_ns).to_msg()
            if first_js is not None:
                first_js.header.stamp = stamp
                self.pub_js.publish(first_js)         # feeds bake
            if ci_l is not None:
                ci_l.header.stamp = stamp; ci_l.header.frame_id = a.left_frame_id
                self.pub_ci_l.publish(ci_l)
            if ci_r is not None:
                ci_r.header.stamp = stamp; ci_r.header.frame_id = a.right_frame_id
                self.pub_ci_r.publish(ci_r)
                self._maybe_publish_baseline(ci_r)
            now = time.perf_counter()
            if now - last > 2.0:
                self.get_logger().info(
                    f"waiting for the rectifier to subscribe on "
                    f"'{a.out_left_image}/compressed' "
                    f"(publishing joints/caminfo so bake can proceed) "
                    f"... (L={nl} R={nr}, {now-t0:.0f}s)")
                last = now
            rel_ns += int(hold_dt * 1e9)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(hold_dt)
        return self.get_clock().now().nanoseconds + 33_000_000

    def _freeze_phase_compressed(self, synth_base_ns):
        """Compressed-mode freeze: find the first IDR access unit per eye
        (prepending cached SPS/PPS if the IDR AU lacks them --- x264 puts the
        headers once at stream start), then republish that SAME self-contained
        AU at freeze_hz with advancing stamps + caminfo + first joints until
        /pose_init latches. Repeating one headers+IDR AU is a legal H.264
        stream: the rectifier's NVDEC locks on immediately and decodes a
        STABLE picture --- exactly what FoundationPose needs to seed, and it
        also solves the late-join lock-on problem outright. Playback then
        starts from the bag's beginning; replaying the IDR once more is a
        harmless duplicate picture."""
        a = self.args
        reader, _ = _open_bag(a.bag)
        au = {True: None, False: None}            # is_left -> frozen AU bytes
        hdr = {True: b"", False: b""}             # cached SPS/PPS per eye
        ci_l = ci_r = first_js = None
        diagnosed = False
        while reader.has_next() and rclpy.ok():
            topic, data, _t = reader.read_next()
            if topic in (a.in_left_image, a.in_right_image):
                is_left = (topic == a.in_left_image)
                if au[is_left] is not None:
                    continue
                payload = bytes(deserialize_message(data, CompressedImage).data)
                types = self._nal_types(payload)
                if not diagnosed:
                    diagnosed = True
                    self.get_logger().info(
                        f"first AU NAL types: {sorted(types)} "
                        f"(7=SPS 8=PPS 5=IDR 1=P/B)")
                    if not types:
                        self.get_logger().error(
                            "first AU has NO Annex-B start codes --- payload is "
                            "likely AVCC (length-prefixed). NVDEC's parser "
                            "cannot sync on this; the rectifier will never "
                            "decode it. Re-encode the bag to Annex-B.")
                hdr[is_left] += self._split_header_nals(payload)
                if 5 in types:                    # IDR found
                    full = (payload if {7, 8} & types
                            else hdr[is_left] + payload)
                    if not ({7, 8} & types) and not hdr[is_left]:
                        self.get_logger().warn(
                            "IDR AU without SPS/PPS and none cached earlier "
                            "in the bag --- decoder lock-on may still fail")
                    au[is_left] = full
            elif topic == a.in_left_caminfo and ci_l is None:
                ci_l = deserialize_message(data, CameraInfo)
            elif topic == a.in_right_caminfo and ci_r is None:
                ci_r = deserialize_message(data, CameraInfo)
            elif topic == a.in_joints and first_js is None:
                first_js = deserialize_message(data, JointState)
            if au[True] is not None and au[False] is not None \
                    and ci_l and ci_r and first_js:
                break
        del reader
        if au[True] is None or au[False] is None:
            self.get_logger().warn(
                "freeze(compressed): no IDR AU found in the bag --- skipping "
                "freeze; lock-on will rely on the subscriber gate alone")
            return synth_base_ns

        self.get_logger().info(
            f"freeze(compressed): holding headers+IDR AU per eye "
            f"(L={len(au[True])}B R={len(au[False])}B) at {a.freeze_hz:.0f} Hz "
            f"until '{a.pose_init_topic}' "
            f"(timeout {a.freeze_timeout:.0f}s, 0=forever)...")
        hold_dt = 1.0 / max(a.freeze_hz, 1.0)
        rel_ns = 0
        deadline = (time.perf_counter() + a.freeze_timeout
                    if a.freeze_timeout > 0 else None)
        while rclpy.ok() and not self._pose_init_seen:
            stamp = Time(nanoseconds=synth_base_ns + rel_ns).to_msg()
            for is_left, pub in ((True, self.pub_cmp_l), (False, self.pub_cmp_r)):
                m = CompressedImage()
                m.header.stamp = stamp
                m.header.frame_id = (a.left_frame_id if is_left
                                     else a.right_frame_id)
                m.format = "h264"
                m.data = au[is_left]
                pub.publish(m)
            if ci_l is not None:
                ci_l.header.stamp = stamp; ci_l.header.frame_id = a.left_frame_id
                self.pub_ci_l.publish(ci_l)
            if ci_r is not None:
                ci_r.header.stamp = stamp; ci_r.header.frame_id = a.right_frame_id
                self.pub_ci_r.publish(ci_r)
                self._maybe_publish_baseline(ci_r)
            if first_js is not None:
                first_js.header.stamp = stamp
                self.pub_js.publish(first_js)
            rel_ns += int(hold_dt * 1e9)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(hold_dt)
            if deadline is not None and time.perf_counter() > deadline:
                self.get_logger().warn("freeze(compressed): timed out waiting "
                                       "for /pose_init --- starting playback")
                break
        # Re-base to NOW: rel_ns under-counts wall time by the per-tick
        # publish work, and carrying that lag forward shows up as a
        # constant bogus offset in the consumer's 'wire' latency column.
        return self.get_clock().now().nanoseconds + 33_000_000

    # ---- freeze on first frame until /pose_init ------------------------------
    def _freeze_phase(self, synth_base_ns):
        """Decode the first frame per eye, then hold it (republishing with
        advancing stamps + held first joints/caminfo) until /pose_init latches.
        Returns the advanced synthetic-clock base so the play loop stays
        monotonic. FoundationPose needs a stable view to seed the handoff."""
        a = self.args
        reader, _ = _open_bag(a.bag)
        dec_l = GpuH264Decoder(device=a.device, gpuid=a.gpuid)
        dec_r = GpuH264Decoder(device=a.device, gpuid=a.gpuid)
        hwc_l = hwc_r = ci_l = ci_r = first_js = None

        # 1) warm up NVDEC: grab the first decoded frame per eye + first
        #    caminfo + first joints (NVDEC may need a few packets to emit).
        while reader.has_next() and rclpy.ok():
            topic, data, _t = reader.read_next()
            if topic == a.in_left_image and hwc_l is None:
                hwc_l = self._decode_hwc(dec_l, deserialize_message(data, CompressedImage).data)
            elif topic == a.in_right_image and hwc_r is None:
                hwc_r = self._decode_hwc(dec_r, deserialize_message(data, CompressedImage).data)
            elif topic == a.in_left_caminfo and ci_l is None:
                ci_l = deserialize_message(data, CameraInfo)
            elif topic == a.in_right_caminfo and ci_r is None:
                ci_r = deserialize_message(data, CameraInfo)
            elif topic == a.in_joints and first_js is None:
                first_js = deserialize_message(data, JointState)
            if hwc_l is not None and hwc_r is not None and ci_l and ci_r and first_js:
                break
        del reader, dec_l, dec_r

        if hwc_l is None or hwc_r is None:
            self.get_logger().warn("freeze: could not decode a first frame "
                                   "(no IDR / not Annex-B?); skipping freeze")
            return synth_base_ns

        self.get_logger().info(
            f"freeze: holding first frame on '{a.pose_init_topic}' "
            f"(timeout {a.freeze_timeout:.0f}s, 0=forever)...")
        hold_dt = 1.0 / max(a.freeze_hz, 1.0)
        rel_ns = 0
        deadline = time.perf_counter() + a.freeze_timeout if a.freeze_timeout > 0 else None

        while rclpy.ok() and not self._pose_init_seen:
            stamp = Time(nanoseconds=synth_base_ns + rel_ns).to_msg()
            self._publish_raw(self.pub_img_l, hwc_l, stamp, "L")
            self._publish_raw(self.pub_img_r, hwc_r, stamp, "R")
            if ci_l is not None:
                ci_l.header.stamp = stamp; ci_l.header.frame_id = a.left_frame_id
                self.pub_ci_l.publish(ci_l)
            if ci_r is not None:
                ci_r.header.stamp = stamp; ci_r.header.frame_id = a.right_frame_id
                self.pub_ci_r.publish(ci_r)
                self._maybe_publish_baseline(ci_r)
            if first_js is not None:
                first_js.header.stamp = stamp
                self.pub_js.publish(first_js)
            rel_ns += int(hold_dt * 1e9)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(hold_dt)
            if deadline is not None and time.perf_counter() > deadline:
                self.get_logger().warn("freeze: timed out waiting for "
                                       "/pose_init --- starting playback anyway")
                break

        # Re-base to NOW (same wire-latency artifact as the compressed
        # freeze: rel_ns under-counts wall time by the per-tick publish work).
        return self.get_clock().now().nanoseconds + 33_000_000

    # ---- main replay loop ----------------------------------------------------
    def run(self):
        a = self.args
        # Synthetic monotonic clock: bag-relative time mapped onto a base that
        # advances every loop, so header stamps never repeat or run backward
        # (the consumer's time-sync drops "old" frames otherwise). L/R relative
        # deltas are preserved, so stereo pairing matches the recording. NOTE:
        # the latency column is only meaningful at --rate ~1.0; at --rate 0 the
        # stamps advance at bag cadence while publishing is faster, so latency
        # reads oddly --- but seg/opt/compute fps (the speed metric) is unaffected.
        synth_base_ns = self.get_clock().now().nanoseconds
        if a.republish_compressed:
            # H.264 is stateful: the rectifier's NVDEC must see the bag's
            # stream headers (SPS/PPS) + an IDR before anything decodes, and
            # VOLATILE pubs keep no history for late joiners.
            #   - freeze mode: go STRAIGHT to the freeze phase (no gate). It
            #     republishes the headers+IDR AU + joints/caminfo/baseline at
            #     freeze_hz, so a rectifier joining at ANY time locks on the
            #     next repeat, and bake gets its joints meanwhile. Gating
            #     first would deadlock: bake waits for joints, the gate waits
            #     for the rectifier, the rectifier waits for bake.
            #   - non-freeze mode: gate playback on the rectifier's
            #     subscriptions (the gate itself publishes joints/caminfo so
            #     bake can proceed), then play from the first AU.
            if a.freeze_until_pose_init:
                synth_base_ns = self._freeze_phase_compressed(synth_base_ns)
            else:
                synth_base_ns = self._wait_for_compressed_subscribers(
                    synth_base_ns)
        elif a.freeze_until_pose_init:
            synth_base_ns = self._freeze_phase(synth_base_ns)
        LOOP_GAP_NS = 33_000_000          # ~1 frame between loop seams
        printed_fmt = False
        loop_idx = 0

        while rclpy.ok():
            reader, _types = _open_bag(a.bag)
            # Fresh NVDEC stream per pass so it re-locks cleanly on the restart
            # IDR; a persistent decoder could carry stale state across the seam.
            # (Skipped in --republish-compressed: no in-source decode.)
            if not a.republish_compressed:
                self.dec_l = GpuH264Decoder(device=a.device, gpuid=a.gpuid)
                self.dec_r = GpuH264Decoder(device=a.device, gpuid=a.gpuid)

            bag_t0_ns = None
            wall_t0 = time.perf_counter()
            last_rel_ns = 0

            while reader.has_next() and rclpy.ok():
                topic, data, t_ns = reader.read_next()
                if bag_t0_ns is None:
                    bag_t0_ns = t_ns
                rel_ns = t_ns - bag_t0_ns
                last_rel_ns = rel_ns
                stamp_msg = Time(nanoseconds=synth_base_ns + rel_ns).to_msg()

                if a.rate > 0:               # pace to bag timeline
                    target = rel_ns / 1e9 / a.rate
                    while (time.perf_counter() - wall_t0) < target and rclpy.ok():
                        time.sleep(0.0005)

                if topic == a.in_left_image or topic == a.in_right_image:
                    msg = deserialize_message(data, CompressedImage)
                    if not printed_fmt:
                        self.get_logger().info(
                            f"image format='{msg.format}' "
                            + ("(forwarding compressed --- rectifier decodes)"
                               if a.republish_compressed
                               else "(feeding payload to NVDEC as-is)"))
                        printed_fmt = True
                    is_left = (topic == a.in_left_image)
                    if a.republish_compressed:
                        # Forward the H.264 CompressedImage unchanged (re-stamped);
                        # the rectifier's NVDEC does the decode.
                        payload = bytes(msg.data)
                        self._cache_refreeze_au(is_left, payload)
                        msg.header.stamp = stamp_msg
                        msg.header.frame_id = (a.left_frame_id if is_left
                                               else a.right_frame_id)
                        (self.pub_cmp_l if is_left else self.pub_cmp_r).publish(msg)
                        if is_left:
                            self._n_l += 1
                        else:
                            self._n_r += 1
                        if self._refreeze_req:
                            frozen_ns = self._refreeze_phase(
                                synth_base_ns + rel_ns)
                            # Shift the synthetic clock AND the wall pacing by
                            # the frozen duration: stamps stay monotonic and
                            # the rate pacer doesn't burst to "catch up".
                            synth_base_ns += frozen_ns
                            wall_t0 += frozen_ns / 1e9
                    elif is_left:
                        if self._refreeze_req and not getattr(
                                self, '_raw_refreeze_warned', False):
                            self._raw_refreeze_warned = True
                            self.get_logger().warn(
                                "re-freeze requested but the source runs in "
                                "RAW mode (--republish-compressed off) --- "
                                "re-freeze is compressed-only; playback "
                                "continues UNFROZEN")
                        if self._emit_image(self.dec_l, self.pub_img_l, msg.data,
                                            stamp_msg, "L"):
                            self._n_l += 1
                            if not self._warm_l:
                                self._warm_l = True
                                self.get_logger().info("left NVDEC locked on")
                    else:
                        if self._emit_image(self.dec_r, self.pub_img_r, msg.data,
                                            stamp_msg, "R"):
                            self._n_r += 1
                            if not self._warm_r:
                                self._warm_r = True
                                self.get_logger().info("right NVDEC locked on")

                elif topic == a.in_left_caminfo:
                    ci = deserialize_message(data, CameraInfo)
                    ci.header.stamp = stamp_msg
                    ci.header.frame_id = a.left_frame_id
                    self.pub_ci_l.publish(ci)

                elif topic == a.in_right_caminfo:
                    ci = deserialize_message(data, CameraInfo)
                    ci.header.stamp = stamp_msg
                    ci.header.frame_id = a.right_frame_id
                    self.pub_ci_r.publish(ci)
                    self._maybe_publish_baseline(ci)

                elif topic == a.in_joints:
                    js = deserialize_message(data, JointState)
                    js.header.stamp = stamp_msg
                    # Consumer reads msg.position BY ORDER (names ignored). The
                    # lbr driver publishes A1..A7 in chain order, matching the
                    # URDF FK, so a straight pass-through is correct.
                    self.pub_js.publish(js)
                    self._n_js += 1

                rclpy.spin_once(self, timeout_sec=0.0)

            loop_idx += 1
            self.get_logger().info(
                f"loop {loop_idx} done: cumulative L={self._n_l} "
                f"R={self._n_r} joints={self._n_js}")
            if not (self._warm_l and self._warm_r):
                self.get_logger().warn(
                    "one eye never produced a frame --- if the bag does not "
                    "start on an IDR, or the payload is not Annex-B, NVDEC "
                    "stays in warmup. Check the format string logged above.")
            del reader
            if not a.loop:
                break
            synth_base_ns += last_rel_ns + LOOP_GAP_NS   # keep stamps monotonic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", default="/media/jack/新加卷/26_04_13_displacement_0_1_h264_v3")
    p.add_argument("--rate", type=float, default=1.0,
                   help="playback speed multiple of bag timeline; <=0 = max throughput")
    p.add_argument("--loop", action="store_true",
                   help="replay the bag indefinitely (monotonic stamps across "
                        "seams; NVDEC re-locks each pass). Ctrl-C to stop.")
    p.add_argument("--freeze-until-pose-init", action="store_true",
                   help="hold the first decoded frame (advancing stamps) until "
                        "/pose_init latches, so FoundationPose has a stable view "
                        "to seed the handoff before playback starts.")
    p.add_argument("--pose-init-topic", default="/pose_init",
                   help="topic watched to release the freeze (FP recorder's seed).")
    p.add_argument("--freeze-hz", type=float, default=15.0,
                   help="republish rate of the held first frame during freeze.")
    p.add_argument("--freeze-timeout", type=float, default=0.0,
                   help="give up the freeze after N s and play anyway (0=forever).")
    p.add_argument("--republish-compressed", action="store_true",
                   help="Forward the bag's H.264 CompressedImage UNCHANGED on "
                        "<out>/compressed instead of NVDEC-decoding to raw. Pair "
                        "with the rectifier's --image-codec h264 (NO --raw-input) "
                        "so the DECODE runs in the rectifier --- the live "
                        "downsampled path --- and shows on the rectifier PID in "
                        "nvidia-smi pmon. Seed the consumer with --extrinsics-file "
                        "(freeze is ignored in this mode).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpuid", type=int, default=0)

    # bag (input) topics
    p.add_argument("--in-left-image", default="/zed/zed_node/left/image_compressed")
    p.add_argument("--in-right-image", default="/zed/zed_node/right/image_compressed")
    p.add_argument("--in-left-caminfo", default="/zed/zed_node/left/camera_info")
    p.add_argument("--in-right-caminfo", default="/zed/zed_node/right/camera_info")
    p.add_argument("--in-joints", default="/lbr/joint_states")

    # published (output) topics --- match rectifier --raw-input + consumer
    p.add_argument("--out-left-image", default="/xr/image_left")
    p.add_argument("--out-right-image", default="/xr/image_right")
    p.add_argument("--out-left-caminfo", default="/xr/image_left/camera_info")
    p.add_argument("--out-right-caminfo", default="/xr/image_right/camera_info")
    p.add_argument("--out-baseline", default="/xr/baseline")
    p.add_argument("--out-joints", default="/xr/joint_states")

    p.add_argument("--left-frame-id", default="zed_left_camera_optical_frame")
    p.add_argument("--right-frame-id", default="zed_right_camera_optical_frame")
    args = p.parse_args()

    rclpy.init()
    node = BagH264Source(args)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()