#!/usr/bin/env python3
"""
bag_to_qpf2.py

Replays an H.264-compressed ROS2 bag as a QPF2 TCP stream, matching
what the Quest 3 Android app produces. Connects out as a TCP client
to the qpf2_receiver.

Wire format: 24-byte big-endian header
    4  magic "QPF2"
    1  cameraId  (50 = left, 51 = right)
    1  flags     bit0=keyframe, bit1=has_csd_prepended
    2  reserved
    8  timestamp_us
    2  width
    2  height
    4  payloadLen
Payload: H.264 Annex-B (with SPS+PPS prepended on first keyframe per cam).

Pacing
------
Frames are sent at the bag's recorded inter-message intervals,
multiplied by 1/rate. rate=1.0 = real-time.

SPS/PPS handling
----------------
On startup, scan the bag for the first SPS/PPS NAL per camera. The
first keyframe sent on each new connection has SPS+PPS prepended and
FLAG_HAS_CSD set. Receiver needs this to initialise its decoder.

Optional gate file
------------------
If --start_signal_file PATH is given, wait until that file exists
before connecting to the receiver. Pairs with the same flag on the
receiver to start both ends together.
"""

import argparse
import logging
import os
import socket
import struct
import sys
import time
from pathlib import Path

try:
    from rosbag2_py import (SequentialReader, StorageOptions,
                             ConverterOptions, StorageFilter)
except ImportError:
    sys.exit("rosbag2_py not found. Source ROS2 first: "
             "source /opt/ros/jazzy/setup.bash")

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


# ── Wire-format constants ────────────────────────────────────────────
MAGIC = b"QPF2"
FLAG_KEYFRAME = 0x01
FLAG_HAS_CSD = 0x02

# ── Camera id <-> topic mapping (hardcoded) ──────────────────────────
LEFT_ID = 50
RIGHT_ID = 51
LEFT_TOPIC = "/zed/zed_node/left/image_compressed"
RIGHT_TOPIC = "/zed/zed_node/right/image_compressed"
TOPIC_TO_CAM = {LEFT_TOPIC: LEFT_ID, RIGHT_TOPIC: RIGHT_ID}

# ── Frame size to advertise in QPF2 header ───────────────────────────
DEFAULT_W = 1920
DEFAULT_H = 1080

# ── H.264 NAL types ──────────────────────────────────────────────────
NAL_SPS = 7
NAL_PPS = 8
NAL_IDR = 5


# ── NAL parsing ──────────────────────────────────────────────────────
def iter_nals(payload: bytes):
    """Yield (start_offset, nal_byte_offset, nal_end_offset) for each
    Annex-B NAL unit in `payload`. start_offset includes the start code."""
    n = len(payload)
    starts = []
    i = 0
    while i + 2 < n:
        if payload[i] == 0 and payload[i + 1] == 0:
            if payload[i + 2] == 1:
                starts.append((i, 3)); i += 3; continue
            if i + 3 < n and payload[i + 2] == 0 and payload[i + 3] == 1:
                starts.append((i, 4)); i += 4; continue
        i += 1
    for k, (start, prefix_len) in enumerate(starts):
        nal_byte = start + prefix_len
        nal_end = starts[k + 1][0] if k + 1 < len(starts) else n
        yield (start, nal_byte, nal_end)


def classify(payload: bytes):
    """Return (is_keyframe, sps_bytes_or_None, pps_bytes_or_None).
    SPS/PPS slices include their start codes so they can be prepended
    directly to the next keyframe."""
    is_kf, sps, pps = False, None, None
    for start, nal_byte, nal_end in iter_nals(payload):
        if nal_byte >= len(payload):
            continue
        nal_type = payload[nal_byte] & 0x1F
        if nal_type == NAL_IDR:
            is_kf = True
        elif nal_type == NAL_SPS:
            sps = payload[start:nal_end]
        elif nal_type == NAL_PPS:
            pps = payload[start:nal_end]
    return is_kf, sps, pps


# ── Bag reader ───────────────────────────────────────────────────────
def open_bag(path: str) -> SequentialReader:
    """Open a ROS2 bag, auto-detecting mcap vs sqlite3."""
    storage_id = "mcap"
    if os.path.isdir(path):
        files = os.listdir(path)
        if any(f.endswith(".db3") for f in files):
            storage_id = "sqlite3"
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"))
    try:
        reader.set_filter(StorageFilter(topics=[LEFT_TOPIC, RIGHT_TOPIC]))
    except Exception:
        pass
    return reader


# ── Sender ───────────────────────────────────────────────────────────
class QPF2Sender:

    def __init__(self, args, log):
        self.args = args
        self.log = log
        self._sps = {}      # cam_id -> SPS Annex-B bytes
        self._pps = {}      # cam_id -> PPS Annex-B bytes
        self._csd_sent = set()

    def prescan(self):
        """First pass: extract SPS+PPS for each camera."""
        self.log.info(f"Pre-scanning bag for SPS/PPS: {self.args.bag}")
        reader = open_bag(self.args.bag)
        scanned = 0
        while reader.has_next():
            topic, raw, _ = reader.read_next()
            cam_id = TOPIC_TO_CAM.get(topic)
            if cam_id is None:
                continue
            payload = bytes(deserialize_message(raw, CompressedImage).data)
            _, sps, pps = classify(payload)
            if sps is not None and cam_id not in self._sps:
                self._sps[cam_id] = sps
                self.log.info(f"cam{cam_id}: cached SPS ({len(sps)} B)")
            if pps is not None and cam_id not in self._pps:
                self._pps[cam_id] = pps
                self.log.info(f"cam{cam_id}: cached PPS ({len(pps)} B)")
            scanned += 1
            if len(self._sps) >= 2 and len(self._pps) >= 2:
                break
            if scanned > 200:
                break
        if not self._sps:
            self.log.warning("No SPS in pre-scan; relying on inline CSD.")

    def wait_for_gate(self):
        """Block until the gate file exists (if --start_signal_file given)."""
        if not self.args.start_signal_file:
            return
        gate = Path(self.args.start_signal_file)
        if gate.exists():
            self.log.info(f"Gate file {gate} already exists; starting.")
            return
        self.log.info(f"Waiting for gate file: {gate}")
        self.log.info(f"   Run `touch {gate}` from any terminal to start.")
        while not gate.exists():
            time.sleep(0.2)
        self.log.info(f"Gate file detected; connecting.")

    def connect(self) -> socket.socket:
        """Connect to the receiver, retrying until --connect_timeout elapses."""
        deadline = time.monotonic() + self.args.connect_timeout
        attempt = 0
        while True:
            attempt += 1
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(2.0)
                client.connect((self.args.host, self.args.port))
                client.settimeout(None)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
                self.log.info(
                    f"Connected to {self.args.host}:{self.args.port} "
                    f"(attempt {attempt})")
                self._csd_sent.clear()
                return client
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Could not connect after {attempt} attempts: {e}")
                self.log.info(f"Receiver not ready ({e}); retrying...")
                time.sleep(1.0)

    def stream_once(self, client: socket.socket):
        """Stream the bag once, paced to wall clock at args.rate."""
        reader = open_bag(self.args.bag)
        rate = max(0.01, float(self.args.rate))
        first_bag_ns = None
        wall_start = None
        sent = 0
        last_log = time.monotonic()

        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            cam_id = TOPIC_TO_CAM.get(topic)
            if cam_id is None:
                continue

            msg = deserialize_message(raw, CompressedImage)
            payload = bytes(msg.data)
            is_kf, _, _ = classify(payload)

            stamp_ns = (msg.header.stamp.sec * 1_000_000_000
                        + msg.header.stamp.nanosec) or t_ns
            timestamp_us = stamp_ns // 1000

            # Bag pacing
            if first_bag_ns is None:
                first_bag_ns = t_ns
                wall_start = time.monotonic()
            else:
                target = (t_ns - first_bag_ns) / 1e9 / rate
                lag = target - (time.monotonic() - wall_start)
                if lag > 0:
                    time.sleep(lag)

            # Flags + payload
            flags = FLAG_KEYFRAME if is_kf else 0
            send_payload = payload
            if is_kf and cam_id not in self._csd_sent:
                sps = self._sps.get(cam_id)
                pps = self._pps.get(cam_id)
                if sps and pps:
                    send_payload = sps + pps + payload
                    flags |= FLAG_HAS_CSD
                self._csd_sent.add(cam_id)

            header = struct.pack(
                ">4sBB2sQHHI",
                MAGIC, cam_id & 0xFF, flags & 0xFF,
                b"\x00\x00",
                timestamp_us & 0xFFFFFFFFFFFFFFFF,
                DEFAULT_W & 0xFFFF, DEFAULT_H & 0xFFFF,
                len(send_payload) & 0xFFFFFFFF)

            client.sendall(header)
            client.sendall(send_payload)
            sent += 1

            now = time.monotonic()
            if now - last_log >= 5.0:
                self.log.info(f"Sent {sent} frames")
                last_log = now

        self.log.info(f"Bag exhausted; sent {sent} frames total.")

    def run(self):
        """Top-level: prescan, optionally wait on gate, connect, stream
        (with optional --loop)."""
        self.prescan()
        self.wait_for_gate()
        while True:
            try:
                client = self.connect()
            except RuntimeError as e:
                self.log.error(str(e))
                return
            try:
                self.stream_once(client)
            except (BrokenPipeError, ConnectionResetError) as e:
                self.log.warning(f"Receiver disconnected: {e}")
            except Exception:
                self.log.exception("Stream error")
            finally:
                try:
                    client.close()
                except Exception:
                    pass
            if self.args.loop != "true":
                return
            self.log.info("Looping bag.")
            self._csd_sent.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7040)
    p.add_argument("--connect_timeout", type=float, default=15.0)
    p.add_argument("--rate", type=float, default=1.0,
                   help="Playback rate multiplier (1.0 = real-time)")
    p.add_argument("--loop", default="false", choices=["true", "false"],
                   help="If 'true', restart the bag after it finishes.")
    p.add_argument("--start_signal_file", default="",
                   help="If set, wait until this file exists before "
                        "connecting. Pairs with qpf2_receiver's flag.")
    args = p.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO)
    log = logging.getLogger("bag_to_qpf2")

    QPF2Sender(args, log).run()


if __name__ == "__main__":
    main()