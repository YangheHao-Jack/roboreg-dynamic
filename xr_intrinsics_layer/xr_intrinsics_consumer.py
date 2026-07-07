#!/usr/bin/env python3
"""
xr_intrinsics_consumer.py

Reads per-frame XrView data published by xr_intrinsics_layer.so
on /tmp/xr_intrinsics.sock, converts XrFovf (angleLeft/Right/Up/Down, radians)
into OpenCV-style intrinsics (fx, fy, cx, cy), and prints per-eye K matrices
once per second. Also emits them as JSON lines on stdout for downstream use.

Start BEFORE Isaac Sim if possible. If started later, it'll just connect and
begin reading the next frame.

Conversion reference:
    Given:
        w, h   = render resolution for this eye (pixels)
        aL,aR  = angleLeft, angleRight  (radians; signed, canonical OpenXR sign)
        aU,aD  = angleUp,   angleDown   (radians; signed, canonical OpenXR sign)
    In OpenXR the angles are signed such that:
        tan(angleRight) - tan(angleLeft)  is the horizontal NDC span (positive)
        tan(angleUp)    - tan(angleDown)  is the vertical NDC span   (positive)

    From the projection matrix derivation:
        fx = w  /  (tan(aR) - tan(aL))
        fy = h  /  (tan(aU) - tan(aD))
        cx = -fx * tan(aL)                    # so that pixel 0 = angleLeft
        cy =  fy * tan(aU)                    # image y grows downward in OpenCV;
                                              # angleUp maps to y=0 top
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import time

SOCKET_PATH = "/tmp/xr_intrinsics.sock"


def fov_to_K(w: int, h: int, aL: float, aR: float, aU: float, aD: float):
    """Convert XrFovf angles → (fx, fy, cx, cy) in pixels for OpenCV convention."""
    tanL = math.tan(aL)
    tanR = math.tan(aR)
    tanU = math.tan(aU)
    tanD = math.tan(aD)

    fx = w / (tanR - tanL)
    fy = h / (tanU - tanD)
    cx = -fx * tanL
    cy = fy * tanU
    return fx, fy, cx, cy


def open_socket_retry(path: str, poll_s: float = 0.5) -> socket.socket:
    """Block until the layer is up and listening."""
    print(f"[consumer] waiting for {path} (layer must be loaded by Isaac Sim)...",
          file=sys.stderr, flush=True)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    while True:
        try:
            s.connect(path)
            print(f"[consumer] connected to {path}", file=sys.stderr, flush=True)
            return s
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(poll_s)


def main():
    sock = open_socket_retry(SOCKET_PATH)
    # Wrap in a file-like for line reading
    f = sock.makefile("r", buffering=1)

    last_print = 0.0
    last_K = {}

    try:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            views = msg.get("views", [])
            for v in views:
                eye = v["eye"]
                w   = int(v["w"]);  h = int(v["h"])
                aL  = float(v["angleLeft"]);  aR = float(v["angleRight"])
                aU  = float(v["angleUp"]);    aD = float(v["angleDown"])
                if w <= 0 or h <= 0:
                    continue
                fx, fy, cx, cy = fov_to_K(w, h, aL, aR, aU, aD)
                last_K[eye] = {
                    "w": w, "h": h,
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "angleLeft":  aL, "angleRight": aR,
                    "angleUp":    aU, "angleDown":  aD,
                    "pose": {
                        "p": [v["px"], v["py"], v["pz"]],
                        "q": [v["qx"], v["qy"], v["qz"], v["qw"]],
                    },
                }

            now = time.monotonic()
            if now - last_print > 1.0 and last_K:
                last_print = now
                print("─" * 74)
                for eye_id in sorted(last_K):
                    k = last_K[eye_id]
                    label = {0: "L", 1: "R"}.get(eye_id, f"V{eye_id}")
                    print(
                        f"[eye {label}] {k['w']}×{k['h']}  "
                        f"fx={k['fx']:8.2f} fy={k['fy']:8.2f}  "
                        f"cx={k['cx']:8.2f} cy={k['cy']:8.2f}  "
                        f"FoV_h={math.degrees(k['angleRight']-k['angleLeft']):.2f}° "
                        f"FoV_v={math.degrees(k['angleUp']-k['angleDown']):.2f}°"
                    )
                # Also emit a compact JSON for programmatic consumers
                print(json.dumps({"K": last_K}), flush=True)
    except KeyboardInterrupt:
        print("\n[consumer] interrupted", file=sys.stderr)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
