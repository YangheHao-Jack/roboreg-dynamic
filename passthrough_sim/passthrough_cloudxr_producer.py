#!/usr/bin/env python3
"""
passthrough_cloudxr_producer.py — H.264-on-ROS-topics producer for the
passthrough_sim pipeline.

Stripped sibling of test_cloudxr.py, scoped to one job: render the KUKA LBR
Med 7 from URDF in Isaac Sim, accept Quest controller input (TROLLEY / JOINT
mode), capture stereo eyes via the xr_frame_layer OpenXR API layer, NVENC-
encode them, and PUBLISH to ROS topics through the python3.12 sidecar
(passthrough_ros_bridge.py). It reproduces the real Quest passthrough boundary:
H.264 CompressedImage on /xr/* topics, consumed by passthrough_rectifier.

Removed vs. test_cloudxr.py (none of it runs in the passthrough path anyway):
the NVENC .mp4 save-video stack, the per-frame .jpg / .npy disk artifacts and
their async IO pool, the rosbag recorder, and the rqt auto-launcher. What
remains is publish-only: press-A starts continuous publishing, press-B stops
(or --ros-publish-always to publish every sim tick).

The Isaac Sim setup, OpenXR session, controller handling, intrinsics capture,
and the capture wire protocol are unchanged from test_cloudxr.py.

For architecture, controller bindings, wire protocol, and pixel-format
rationale see ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import queue
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()

# Robot
parser.add_argument("--ros-package", default="lbr_description",
                    help="ROS package providing the xacro/URDF.")
parser.add_argument("--xacro-path", default="urdf/med7/med7.xacro",
                    help="xacro path relative to --ros-package share dir.")
parser.add_argument("--urdf-file", default=None,
                    help="Optional direct URDF path (skips xacro processing).")
parser.add_argument("--base-pose", default="-0.5,1.5",
                    help="Initial robot base X,Y in metres. The OpenXR "
                         "headset spawns at world (0, 0, eye_height) "
                         "looking in +Y direction (Y-up→Z-up: OpenXR -Z "
                         "forward becomes world +Y), so by default we "
                         "place the robot at (-0.5, 1.5) — 1.5 m forward "
                         "and 0.5 m to the user's right — for a sensible "
                         "off-axis viewing geometry. Z is determined by "
                         "mode: 0 (no OR) or trolley_height (OR).")
parser.add_argument("--trolley-height", type=float, default=0.85,
                    help="Trolley height in OR mode (m). Robot base sits on top.")
parser.add_argument("--initial-joints", default="0,0,0,-90,0,90,0",
                    help="Initial joint angles A1..A7 in degrees. "
                         "Ignored if --initial-joints-file is given.")
parser.add_argument("--initial-joints-file", default=None,
                    help="CSV file with one header row (lbr_A1..lbr_A7) "
                         "and one data row of 7 joint angles in RADIANS. "
                         "Overrides --initial-joints when present.")
parser.add_argument("--joint-mode", action="store_true",
                    help="Start in joint-control mode (default: trolley mode).")
parser.add_argument("--trolley-speed", type=float, default=0.5,
                    help="Trolley translation rate (m per sim tick × thumbstick).")
parser.add_argument("--joint-speed", type=float, default=2.0,
                    help="Joint rate (deg per sim tick × thumbstick).")

# Scene
parser.add_argument("--or-environment", default=None,
                    help="Path to OR USD. If omitted, just ground + dome light.")

# Intrinsics
parser.add_argument("--intrinsics-warmup-sec", type=float, default=2.0,
                    help="Seconds after AR activation to wait before snapshotting intrinsics.")
parser.add_argument("--intrinsics-snapshot-path", default="/tmp/xr_intrinsics_snapshot.json",
                    help="Where to write the frozen K snapshot.")

# Data capture
parser.add_argument("--capture-mode", default="off",
                    choices=["off", "layer"],
                    help="Stereo data capture mode. 'layer' requires the "
                         "xr_frame_layer .so on XR_API_LAYER_PATH and "
                         "XR_FRAME_STRIP_FOVEATION=1 in the environment. "
                         "See XR_CAPTURE_NOTES.md.")
parser.add_argument("--output-dir", default=None,
                    help="Output directory for captured stereo data. "
                         "If omitted, auto-generated as "
                         "~/xr_captures/<timestamp>/. Directory structure "
                         "matches sim_isaac_node.py.")
parser.add_argument("--capture-trigger-hold", type=float, default=0.3,
                    help="Left trigger must be held >0.9 for this many "
                         "seconds (TROLLEY mode only) to capture one frame.")
parser.add_argument("--frames-socket-path", default="/tmp/xr_frames.sock",
                    help="Unix socket the API layer publishes XR eye "
                         "frames on. Used only by --capture-mode=layer.")
parser.add_argument("--image-codec", default="h264", choices=["h264"],
                    help="Producer->rectifier image codec. H.264 only: NVENC "
                         "(dedicated engine; RGB->YUV CSC on the encode "
                         "engine, nothing on the SMs). The JPEG/nvJPEG path "
                         "was removed — nvJPEG ran on the SMs. The rectifier "
                         "NVDEC-decodes and rectifies in the NV12 domain.")
parser.add_argument("--h264-bitrate", type=int, default=12_000_000,
                    help="NVENC target bitrate (CBR) when --image-codec h264.")
parser.add_argument("--h264-ippp", action="store_true",
                    help="Use IPPP (gop=30) NVENC instead of the default "
                         "all-intra. Smaller wire, but inter-frame P-frames "
                         "can smear moving edges (only affects --image-codec "
                         "h264).")
parser.add_argument("--raw-images", action="store_true",
                    help="TEST: publish raw rgb8 sensor_msgs/Image instead of "
                         "H.264 CompressedImage on the producer->rectifier hop "
                         "(no compression). Host capture path only; "
                         "incompatible with XR_FRAME_EXPORT_FD, which keeps "
                         "pixels GPU-resident. Run the rectifier with "
                         "--raw-input to match.")
parser.add_argument("--ros-publish", action="store_true",
                    help="Publish live capture data to ROS2 topics "
                         "during continuous (press-A → press-B) runs. "
                         "Auto-spawns the Python 3.12 sidecar script "
                         "(--ros-bridge-script) that talks to ROS2 "
                         "via rclpy; this script communicates with "
                         "it over a UNIX socket. See ARCHITECTURE.md.")
parser.add_argument("--ros-publish-always", action="store_true",
                    help="Live ROS publishing service mode — publishes "
                         "to /xr/* every sim tick. No recording: "
                         "press-A is disabled, no rosbag, no .npy "
                         "files. Use plain --ros-publish (without "
                         "-always) if you want press-A to record. "
                         "Implies --ros-publish. Sim tick is bound "
                         "by the ~25 ms capture rate (~30-40 Hz).")
parser.add_argument("--ros-publish-no-record", action="store_true",
                    help="Like --ros-publish-always (live /xr/* publishing, "
                         "no rosbag, no .npy), BUT gated on press-A: nothing "
                         "is published until you press A, and press-B stops. "
                         "Use when the consumer/overlay should start on the A "
                         "cue without recording a dataset. Implies "
                         "--ros-publish. If combined with --ros-publish-always "
                         "(which disables press-A), always-mode wins.")
parser.add_argument("--ros-namespace", default="/xr",
                    help="ROS2 topic namespace prefix (default /xr).")
parser.add_argument("--ros-frame-id", default="world",
                    help="frame_id for world-frame PoseStamped messages "
                         "(left/right eye + base poses). Default 'world'.")
parser.add_argument("--ros-bridge-script",
                    default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "passthrough_ros_bridge.py"),
                    help="Path to the ROS2 sidecar script (Python 3.12). "
                         "Defaults to passthrough_ros_bridge.py next to this "
                         "file.")
parser.add_argument("--ros-bridge-setup", default="/opt/ros/jazzy/setup.bash",
                    help="ROS2 setup.bash that the sidecar will source "
                         "before importing rclpy (default "
                         "/opt/ros/jazzy/setup.bash).")
parser.add_argument("--ros-bridge-python", default="python3.12",
                    help="Python interpreter the sidecar should run "
                         "under. Default 'python3.12' — must match "
                         "the Python version your ROS2 distro was "
                         "built against (3.12 for Jazzy on Ubuntu 24). "
                         "Important: this script's own Python (3.11 "
                         "from isaac_env) cannot load rclpy, hence "
                         "the sidecar architecture.")
parser.add_argument("--ros-bridge-socket", default="/tmp/xr_ros_bridge.sock",
                    help="UNIX socket path test_cloudxr.py ↔ sidecar "
                         "(default /tmp/xr_ros_bridge.sock).")
parser.add_argument("--profile-capture", action="store_true",
                    help="Emit a per-frame phase-timing line "
                         "(sendGET / recv / encode / dispatch+state / "
                         "total) from LayerFrameCapture.capture(). Use "
                         "this to see which phase dominates the ~30 ms "
                         "per-capture latency before choosing the next "
                         "acceleration direction.")
parser.add_argument("--foveation-mode", default="default",
                    choices=["default", "none", "inset", "warped"],
                    help="Override Omniverse XR foveation carb setting. "
                         "Use 'none' alongside XR_FRAME_STRIP_FOVEATION=1 "
                         "for IPCAI data capture. See XR_CAPTURE_NOTES.md.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# --ros-publish-always implies --ros-publish (else there's no bridge
# to publish to). Auto-enable rather than erroring out.
if args.ros_publish_always and not args.ros_publish:
    print("[ros] --ros-publish-always implies --ros-publish; enabling both")
    args.ros_publish = True
if args.ros_publish_no_record and not args.ros_publish:
    print("[ros] --ros-publish-no-record implies --ros-publish; enabling both")
    args.ros_publish = True
if args.ros_publish_no_record and args.ros_publish_always:
    print("[ros] --ros-publish-always disables press-A; "
          "--ros-publish-no-record ignored")
    args.ros_publish_no_record = False

# --raw-images: publish uncompressed rgb8 Image on the producer->rectifier hop.
# Needs the host capture path (pixels in host memory); FD-export keeps them on
# the GPU, so the two are mutually exclusive. Signalled to the capture class and
# the bridge sidecar via XR_RAW_IMAGES (the sidecar inherits it; see keep_keys).
if args.raw_images:
    if os.environ.get("XR_FRAME_EXPORT_FD", "").startswith("1"):
        print("[raw] --raw-images needs the host path, but XR_FRAME_EXPORT_FD "
              "is set (pixels stay on the GPU). Unset it and rerun. Exiting.")
        sys.exit(2)
    os.environ["XR_RAW_IMAGES"] = "1"
    print("[raw] producer will publish raw rgb8 Image (compression disabled)")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ─── Post-launch imports ─────────────────────────────────────────────────────
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.devices import DeviceBase
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.xr_cfg import XrCfg
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

import omni.usd
import omni.kit.app
import omni.kit.commands
from pxr import UsdGeom, UsdLux, UsdPhysics, Usd, Gf, Sdf


XR_CAMERA_PATH = "/_xr/stage/xrCamera"

INTRINSICS_SOCKET = "/tmp/xr_intrinsics.sock"

JOINT_NAMES = [f"lbr_A{i}" for i in range(1, 8)]


# ═════════════════════════════════════════════════════════════════════════════
# URDF preparation (xacro → urdf if needed)
# ═════════════════════════════════════════════════════════════════════════════

def prepare_urdf(ros_package: str, xacro_path: str, urdf_file: str | None) -> str | None:
    """Return a usable .urdf file path, processing xacro if needed.

    Mirrors sim_isaac_node.prepare_urdf so behaviour matches the offline
    synthetic-data pipeline:
      1. xacro → URDF (unless --urdf-file given)
      2. Strip Gazebo plugins, ros2_control plugins, transmissions, and
         the dangling `world` link / lbr_world_joint that Isaac Sim 5.1's
         importer chokes on.
      3. Rewrite `package://<ros_package>/...` to absolute paths so the
         importer can resolve the mesh files without ROS_PACKAGE_PATH.
    """
    import re

    if urdf_file and os.path.exists(urdf_file):
        with open(urdf_file) as f:
            u = f.read()
    else:
        if not ros_package or not xacro_path:
            print("[URDF] need --ros-package and --xacro-path, or --urdf-file")
            return None
        try:
            r = subprocess.run(
                ["ros2", "pkg", "prefix", ros_package, "--share"],
                capture_output=True, text=True, check=True,
            )
            pkg = r.stdout.strip()
        except Exception:
            print(f"[URDF] Cannot find ROS package '{ros_package}'. "
                  f"Source your ROS workspace first.")
            return None
        xp = os.path.join(pkg, xacro_path)
        if not os.path.exists(xp):
            print(f"[URDF] Not found: {xp}")
            return None
        try:
            u = subprocess.run(
                ["xacro", xp], capture_output=True, text=True, check=True,
            ).stdout
        except Exception as e:
            print(f"[URDF] xacro failed: {e}")
            return None

    # Strip Gazebo / ros2_control / transmission / orphan world link
    for pat in [
        r"<gazebo>\s*<plugin[^>]*gz_ros2_control[^>]*>.*?</plugin>\s*</gazebo>",
        r"<plugin[^>]*gz_ros2_control[^>]*>.*?</plugin>",
        r"<gazebo[^>]*>.*?</gazebo>",
        r"<transmission[^>]*>.*?</transmission>",
        r'<joint\s+name="lbr_world_joint"[^>]*>.*?</joint>',
    ]:
        u = re.sub(pat, "", u, flags=re.DOTALL)
    u = re.sub(r'<link\s+name="world"\s*/>', "", u)
    u = re.sub(r'<link\s+name="world"\s*>\s*</link>', "", u)

    # Rewrite package://<ros_package>/... to absolute paths.
    if ros_package:
        try:
            pkg_share = subprocess.run(
                ["ros2", "pkg", "prefix", ros_package, "--share"],
                capture_output=True, text=True,
            ).stdout.strip()
            if pkg_share:
                u = u.replace(f"package://{ros_package}/", f"{pkg_share}/")
        except Exception:
            pass

    out = os.path.join(tempfile.gettempdir(), f"{ros_package}_isaac.urdf")
    with open(out, "w") as f:
        f.write(u)
    print(f"[URDF] → {out}")
    return out


def load_initial_joints_csv(path: str) -> np.ndarray:
    """Read a single-row CSV of 7 joint angles in radians, return degrees.

    Expected file format (matches the recorded joint_state_left_*.csv):
        lbr_A1,lbr_A2,lbr_A3,lbr_A4,lbr_A5,lbr_A6,lbr_A7
        <a1>,<a2>,<a3>,<a4>,<a5>,<a6>,<a7>

    Values in the data row are in RADIANS; the returned array is in DEGREES
    (to match the rest of the script's drive-target convention).
    """
    import csv
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{path}: expected header + at least one data row, "
                         f"got {len(rows)} rows")
    # Skip header, take first data row
    data_row = rows[1]
    if len(data_row) < 7:
        raise ValueError(f"{path}: data row has {len(data_row)} columns, need 7")
    rad = np.array([float(x) for x in data_row[:7]], dtype=np.float64)
    return np.degrees(rad)


# ═════════════════════════════════════════════════════════════════════════════
# Maths helpers
# ═════════════════════════════════════════════════════════════════════════════

def _rot_z_to_quat_wxyz(theta_rad: float):
    c = math.cos(theta_rad * 0.5)
    s = math.sin(theta_rad * 0.5)
    return (c, 0.0, 0.0, s)


def _rot_to_quat_wxyz(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 2 * math.sqrt(t + 1.0)
        return (0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def _set_quat(op, q):
    try:
        op.Set(Gf.Quatd(q[0], q[1], q[2], q[3]))
    except Exception:
        op.Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))


def _set_prim_pose(prim, pos, quat):
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*pos))
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            _set_quat(op, quat)


def _ensure_pose_ops(stage, path, pos, quat):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    if xf.GetOrderedXformOps():
        _set_prim_pose(stage.GetPrimAtPath(path), pos, quat)
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        _set_quat(xf.AddOrientOp(), quat)


# ═════════════════════════════════════════════════════════════════════════════
# Scene
# ═════════════════════════════════════════════════════════════════════════════

def load_or_environment(or_usd_path: str):
    if not os.path.exists(or_usd_path):
        raise FileNotFoundError(f"OR USD not found: {or_usd_path}")
    stage = omni.usd.get_context().get_stage()
    env_prim = stage.DefinePrim("/World/Environment", "Xform")
    env_prim.GetReferences().AddReference(or_usd_path)
    for _ in range(5):
        simulation_app.update()
    bbox_cache = UsdGeom.BBoxCache(0, ["default"])
    bbox = bbox_cache.ComputeWorldBound(env_prim)
    rng = bbox.GetRange()
    max_dim = max(rng.GetSize()[0], rng.GetSize()[1], rng.GetSize()[2])
    if max_dim > 50:
        scale = 0.01
        center = (rng.GetMin() + rng.GetMax()) * 0.5
        xf = UsdGeom.Xformable(env_prim)
        xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        xf.AddTranslateOp().Set(Gf.Vec3d(-center[0], -center[1], -rng.GetMin()[2]))
        print(f"[scene] OR auto-scaled ×{scale} (max_dim was {max_dim:.0f})")
    print(f"[scene] Loaded OR environment: {or_usd_path}")
    for _ in range(20):
        simulation_app.update()


def build_default_scene():
    """Default scene: ground plane + dome + distant light.

    Matches sim_isaac_node.py: DomeLight @ 1000 (soft ambient fill) plus
    a DistantLight @ 3000 (directional sun-like key). Together they give
    enough shadow contrast for the robot meshes to read clearly.
    """
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    stage = omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").GetIntensityAttr().Set(1000)
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").GetIntensityAttr().Set(3000)


def import_robot(urdf_path: str, base_pos: np.ndarray, base_yaw: float,
                 apply_pbr: bool = False) -> str:
    """Import the URDF, place the base. Returns USD path of robot root.

    Args:
        apply_pbr: if True, bind a UsdPreviewSurface PBR material to every
                   robot mesh. Needed when the scene uses a PBR-lit
                   environment (e.g. OR USD) — without it the URDF's
                   default white surfaces render flat/unlit because they
                   don't react to scene lighting.
    """
    stage = omni.usd.get_context().get_stage()

    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    for _ in range(10):
        simulation_app.update()

    from isaacsim.asset.importer.urdf import _urdf
    import_config = _urdf.ImportConfig()
    import_config.fix_base = True
    import_config.self_collision = False
    import_config.create_physics_scene = False

    result, robot_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
    )
    if not robot_path:
        robot_path = "/med7"
    print(f"[Isaac] URDF → {robot_path}")

    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        print(f"[ERROR] Robot prim not valid at {robot_path}")
        os._exit(1)

    # ── PBR material for proper interaction with scene lighting ──
    # Matches sim_isaac_node.py:307-325. Without this, the robot meshes
    # render as flat white because they don't have material bindings that
    # respond to the OR's PBR light setup.
    if apply_pbr:
        from pxr import UsdShade, Sdf
        mat_path = f"{robot_path}/RobotMaterial"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f
                           ).Set(Gf.Vec3f(0.9, 0.9, 0.9))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
        mat.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface")
        applied = 0
        for prim in stage.Traverse():
            if (prim.GetPath().pathString.startswith(robot_path)
                    and prim.IsA(UsdGeom.Mesh)):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
                applied += 1
        print(f"[Isaac] Applied PBR material to {applied} robot meshes")

    q = _rot_z_to_quat_wxyz(base_yaw)
    _ensure_pose_ops(stage, robot_path, base_pos.tolist(), q)
    print(f"[Isaac] Robot at [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}], "
          f"yaw={math.degrees(base_yaw):.1f}°")

    return robot_path


def spawn_trolley(robot_path: str, base_height: float):
    """Build a simple USD trolley as a child of the robot root.

    Replicates sim_isaac_node.py's trolley block: a top plate, four legs,
    a shelf, four wheels. The trolley is parented under the robot prim so
    it moves with the base. Only spawned when the base is far enough off
    the floor for legs to make geometric sense.

    Args:
        robot_path: USD path of the robot root prim.
        base_height: Z-coordinate of the robot base in world frame (m).
    """
    if base_height <= 0.1:
        print(f"[Isaac] Skipping trolley — base height {base_height:.3f}m too low")
        return

    stage = omni.usd.get_context().get_stage()
    trolley_h = base_height
    trolley_path = f"{robot_path}/Trolley"
    stage.DefinePrim(trolley_path, "Xform")

    # Top plate
    tt = UsdGeom.Cube.Define(stage, f"{trolley_path}/top")
    tt.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(tt.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
    xf.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.05))
    tt.GetDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.35, 0.4)])

    # Four legs
    for i, (dx, dy) in enumerate([(-0.2, -0.2), (0.2, -0.2),
                                  (-0.2,  0.2), (0.2,  0.2)]):
        leg = UsdGeom.Cylinder.Define(stage, f"{trolley_path}/leg_{i}")
        leg.GetRadiusAttr().Set(0.02)
        leg.GetHeightAttr().Set(trolley_h - 0.05)
        xf = UsdGeom.Xformable(leg.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(dx, dy, -(trolley_h - 0.05) / 2 - 0.025))
        leg.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])

    # Lower shelf
    shelf = UsdGeom.Cube.Define(stage, f"{trolley_path}/shelf")
    shelf.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(shelf.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -trolley_h + 0.15))
    xf.AddScaleOp().Set(Gf.Vec3f(0.45, 0.45, 0.02))
    shelf.GetDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.35, 0.4)])

    # Four wheels
    for i, (dx, dy) in enumerate([(-0.22, -0.22), (0.22, -0.22),
                                  (-0.22,  0.22), (0.22,  0.22)]):
        wh = UsdGeom.Cylinder.Define(stage, f"{trolley_path}/wheel_{i}")
        wh.GetRadiusAttr().Set(0.03)
        wh.GetHeightAttr().Set(0.02)
        xf = UsdGeom.Xformable(wh.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(dx, dy, -trolley_h + 0.03))
        wh.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])

    print(f"[Isaac] Robot+Trolley combined, height={trolley_h:.2f}m")


def apply_joint_targets(robot_path: str, joint_deg: np.ndarray):
    """Set USD drive target positions for each joint (degrees)."""
    stage = omni.usd.get_context().get_stage()
    for i, jn in enumerate(JOINT_NAMES[:min(len(joint_deg), 7)]):
        for prim in stage.Traverse():
            if (prim.GetPath().pathString.startswith(robot_path)
                and prim.GetName() == jn):
                dr = UsdPhysics.DriveAPI.Get(prim, "angular")
                if dr and dr.GetTargetPositionAttr():
                    dr.GetTargetPositionAttr().Set(float(joint_deg[i]))


def set_base_pose(robot_path: str, pos: np.ndarray, yaw: float):
    stage = omni.usd.get_context().get_stage()
    q = _rot_z_to_quat_wxyz(yaw)
    _set_prim_pose(stage.GetPrimAtPath(robot_path), pos.tolist(), q)


# ═════════════════════════════════════════════════════════════════════════════
# Controller input
# ═════════════════════════════════════════════════════════════════════════════

class _MotionCtrlReq(RetargeterBase):
    def retarget(self, data):
        return torch.zeros(1)
    def get_requirements(self):
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class ControllerState:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    l_trig: float = 0.0
    r_trig: float = 0.0
    l_grip: float = 0.0
    r_grip: float = 0.0
    a: bool = False
    b: bool = False
    a_edge: bool = False     # rising edge of A (press, not held)
    b_edge: bool = False     # rising edge of B (press, not held)
    menu: bool = False
    menu_edge: bool = False


class ControllerReader:
    DEADZONE = 0.15
    MENU_HOLD_SEC = 0.5  # how long MENU must be held before mode-toggle fires

    def __init__(self):
        """Instantiate the OpenXR device wrapper. If construction fails
        (e.g. AR not yet active, missing extension), enabled goes False
        and read() returns a zero state every tick. The script keeps
        running so the user can still see the scene."""
        self.enabled = False
        self._device = None
        self._toggle_press_started_at: float | None = None
        self._toggle_fired_this_press = False
        self._prev_a = False
        self._prev_b = False
        try:
            self._device = OpenXRDevice(
                cfg=OpenXRDeviceCfg(xr_cfg=XrCfg()),
                retargeters=[_MotionCtrlReq(RetargeterCfg(sim_device="cpu"))],
            )
            self.enabled = True
            # Report which inputs are mapped on this Isaac Lab version
            IIdx = DeviceBase.MotionControllerInputIndex
            available = [e.name for e in IIdx]
            print(f"[input] OpenXR controller inputs available: {available}")
            if "MENU" not in available:
                print("[input] No MENU button exposed — using "
                      "BOTH GRIPS held simultaneously for 0.5s as mode toggle.")
        except Exception as e:
            print(f"[input] OpenXRDevice unavailable ({e}); controllers disabled.")

    def _deadzone(self, v):
        return 0.0 if abs(v) < self.DEADZONE else float(v)

    def read(self, joint_mode: bool = False) -> ControllerState:
        if not self.enabled or self._device is None:
            return ControllerState()
        try:
            raw = self._device._get_raw_data()
        except Exception:
            return ControllerState()

        IIdx = DeviceBase.MotionControllerInputIndex
        idx_tx = IIdx.THUMBSTICK_X.value
        idx_ty = IIdx.THUMBSTICK_Y.value
        # Map button names defensively. Different Isaac Lab versions use
        # different names: older builds expose BUTTON_A/BUTTON_B, newer ones
        # expose BUTTON_0/BUTTON_1 (same physical buttons, different naming).
        def _idx(*candidates):
            for name in candidates:
                if hasattr(IIdx, name):
                    return getattr(IIdx, name).value
            return None

        idx_trig = _idx("TRIGGER")
        idx_grip = _idx("SQUEEZE", "GRIP")
        idx_btnA = _idx("BUTTON_A", "BUTTON_0")
        idx_btnB = _idx("BUTTON_B", "BUTTON_1")
        idx_menu = _idx("MENU")

        irow = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
        st = ControllerState()

        left = raw.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT)
        right = raw.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT)

        if left is not None and len(left) > irow:
            row = left[irow]
            st.lx = self._deadzone(row[idx_tx])
            st.ly = self._deadzone(row[idx_ty])
            if idx_trig is not None:
                try: st.l_trig = float(row[idx_trig])
                except Exception: pass
            if idx_grip is not None:
                try: st.l_grip = float(row[idx_grip])
                except Exception: pass
        if right is not None and len(right) > irow:
            row = right[irow]
            st.rx = self._deadzone(row[idx_tx])
            st.ry = self._deadzone(row[idx_ty])
            if idx_trig is not None:
                try: st.r_trig = float(row[idx_trig])
                except Exception: pass
            if idx_grip is not None:
                try: st.r_grip = float(row[idx_grip])
                except Exception: pass
            if idx_btnA is not None:
                try: st.a = bool(row[idx_btnA] > 0.5)
                except Exception: pass
            if idx_btnB is not None:
                try: st.b = bool(row[idx_btnB] > 0.5)
                except Exception: pass
            if idx_menu is not None:
                try: st.menu = bool(row[idx_menu] > 0.5)
                except Exception: pass

        # ── A/B rising-edge detection (right-hand controller) ──
        # Fires once per press; reset on release. Consumed by the main
        # loop only in TROLLEY mode (JOINT mode uses raw st.a/st.b for
        # joint A7).
        if st.a and not self._prev_a:
            st.a_edge = True
        if st.b and not self._prev_b:
            st.b_edge = True
        self._prev_a = st.a
        self._prev_b = st.b

        # ── Mode toggle: hold BOTH GRIPS (squeeze) for MENU_HOLD_SEC. ──
        # Grips are unused for any other action — won't conflict with
        # triggers (which drive joint A5 in JOINT mode) or anything else.
        toggle_intent = st.menu or (st.l_grip > 0.7 and st.r_grip > 0.7)
        now = time.monotonic()
        if toggle_intent:
            if self._toggle_press_started_at is None:
                self._toggle_press_started_at = now
            held_for = now - self._toggle_press_started_at
            if (not self._toggle_fired_this_press
                    and held_for >= self.MENU_HOLD_SEC):
                st.menu_edge = True
                self._toggle_fired_this_press = True
        else:
            self._toggle_press_started_at = None
            self._toggle_fired_this_press = False
        return st


# ═════════════════════════════════════════════════════════════════════════════
# USD pose reader (head + anchor)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Pose:
    pos: tuple
    quat_wxyz: tuple


def read_usd_pose(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    M = np.array(m, dtype=np.float64).T
    return Pose(
        pos=(float(M[0, 3]), float(M[1, 3]), float(M[2, 3])),
        quat_wxyz=tuple(float(v) for v in _rot_to_quat_wxyz(M[:3, :3])),
    )


def _quat_wxyz_to_R(q):
    """Convert wxyz quaternion to a 3×3 rotation matrix."""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=np.float64)


def _pose_to_H(pose):
    """Pose(pos, quat_wxyz) → 4×4 homogeneous transform."""
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = _quat_wxyz_to_R(pose.quat_wxyz)
    H[:3, 3] = np.array(pose.pos, dtype=np.float64)
    return H


def _xy_yaw_to_H(xyz, yaw):
    """Robot base pose (XY translation + Z yaw, Z translation included) → 4×4."""
    c, s = math.cos(yaw), math.sin(yaw)
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    H[:3, 3] = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return H


# ── Coordinate-frame conversions ────────────────────────────────────────────
# OpenXR Y-up ref space → USD Z-up world, and OpenXR camera-local
# (-Z fwd, +Y up, +X right) → Isaac Sim rig (+X fwd, +Z up, +Y left).
# Output convention matches sim_isaac_node.py's H_cam_world.
# See XR_CAPTURE_NOTES.md for the derivation.

R_REF_TO_WORLD = np.array([    # OpenXR ref Y-up → USD world Z-up
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0],
], dtype=np.float64)

R_CAM_TO_RIG = np.array([      # OpenXR cam-local → Isaac Sim rig-local
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
], dtype=np.float64)


def _ref_pose_to_world_rig(H_ref: np.ndarray) -> np.ndarray:
    """Convert OpenXR ref-space pose (Y-up, cam-local) → USD world (Z-up,
    rig-local). Output's +X column points along the gaze direction,
    matching sim_isaac_node.py's camera_world_*.npy. See XR_CAPTURE_NOTES.md.
    """
    # Step A: bring ref pose into world coords
    H_world_cam = np.eye(4, dtype=np.float64)
    H_world_cam[:3, :3] = R_REF_TO_WORLD @ H_ref[:3, :3]
    H_world_cam[:3, 3]  = R_REF_TO_WORLD @ H_ref[:3, 3]
    # Step B: change camera-local basis to rig-local basis
    H_world_rig = H_world_cam.copy()
    H_world_rig[:3, :3] = H_world_cam[:3, :3] @ R_CAM_TO_RIG
    return H_world_rig


def _usd_xrcam_to_world_rig(H_xrcam_world: np.ndarray) -> np.ndarray:
    """Convert a USD xrCamera prim's world transform (Omniverse XR places
    the prim with OpenXR's camera-local axis convention applied to a Z-up
    world) into Isaac Sim rig convention.

    The position is already in USD world (Z-up); we only need to re-express
    the camera basis from OpenXR cam-local to Isaac Sim rig-local.
    """
    H_world_rig = H_xrcam_world.copy()
    H_world_rig[:3, :3] = H_xrcam_world[:3, :3] @ R_CAM_TO_RIG
    return H_world_rig


def is_xr_active():
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(XR_CAMERA_PATH)
    return bool(prim and prim.IsValid())



# ═════════════════════════════════════════════════════════════════════════════
# Intrinsics capture — one-shot, self-terminating (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def _fov_to_K(w, h, aL, aR, aU, aD):
    fx = w / (math.tan(aR) - math.tan(aL))
    fy = h / (math.tan(aU) - math.tan(aD))
    cx = -fx * math.tan(aL)
    cy =  fy * math.tan(aU)
    return fx, fy, cx, cy


class IntrinsicsOneShot:
    def __init__(self, warmup_sec: float, save_path: str):
        self.warmup_sec = warmup_sec
        self.save_path = save_path
        self._snapshot: dict | None = None
        self._done = threading.Event()
        self._status = "init"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_done(self) -> bool:
        return self._done.is_set()

    def snapshot(self) -> dict | None:
        return self._snapshot

    def status(self) -> str:
        return self._status

    def _fail(self, status: str):
        """Latch a terminal failure status and signal done. Callers
        in `_run` should `return` immediately after."""
        self._status = status
        self._done.set()

    def _run(self):
        self._status = "connecting"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                sock.connect(INTRINSICS_SOCKET)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.5)
        else:
            self._fail("no socket")
            return

        self._status = "warmup"
        f = sock.makefile("r", buffering=1)
        first_valid_at = None
        last_msg = None

        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                views = msg.get("views") or []
                if not views:
                    continue
                if first_valid_at is None:
                    first_valid_at = time.monotonic()
                if time.monotonic() - first_valid_at >= self.warmup_sec:
                    last_msg = msg
                    break
        except Exception as e:
            self._status = f"read error: {e}"

        try:
            sock.close()
        except Exception:
            pass

        if last_msg is None:
            self._fail("no data")
            return

        snap: dict[int, dict] = {}
        for v in last_msg.get("views", []):
            w = int(v["w"]); h = int(v["h"])
            if w <= 0 or h <= 0:
                continue
            aL = float(v["angleLeft"]);  aR = float(v["angleRight"])
            aU = float(v["angleUp"]);    aD = float(v["angleDown"])
            fx, fy, cx, cy = _fov_to_K(w, h, aL, aR, aU, aD)
            snap[int(v["eye"])] = {
                "w": w, "h": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "angleLeft": aL, "angleRight": aR,
                "angleUp":   aU, "angleDown":  aD,
            }

        if not snap:
            self._fail("no eyes")
            return

        try:
            with open(self.save_path, "w") as f:
                json.dump({"K": snap}, f, indent=2)
        except Exception as e:
            self._fail(f"save error: {e}")
            return

        self._snapshot = snap
        self._status = "done"
        self._done.set()


# ═════════════════════════════════════════════════════════════════════════════
# Capture I/O helpers (shared by layer-mode disk writes)
# ═════════════════════════════════════════════════════════════════════════════

# Encoder state — populated by _init_encoder() at startup. Image encoding
# runs on the sim tick (CUDA serialised); encoded bytes are dispatched to
# the IO pool for disk write.
_GPU_DEVICE:   str = "cuda:0"
_TORCH        = None            # lazy-imported torch module
_H264_BITRATE = 12_000_000
_H264_GOP     = 1               # 1 = all-intra; 30 = IPPP
_H264_ENC     = {}              # eye_idx -> GpuH264Encoder (lazy, per dims)
_GpuH264Encoder = None          # lazy-imported class


def _init_encoder(device: str,
                  h264_bitrate: int = 12_000_000, h264_all_intra: bool = True):
    """Configure the per-frame H.264 encoder and warm it up.

    NVENC (GpuH264Encoder), a dedicated engine — per-eye encoders are built
    lazily on the first frame (they need the frame dimensions). The
    JPEG/nvJPEG encoder was removed: nvJPEG ran on the SMs, which is exactly
    the contention this pipeline is built to avoid. Raises on any failure; if
    CUDA / the encoder is unavailable, fix that rather than silently
    degrading."""
    global _GPU_DEVICE, _TORCH, _H264_BITRATE, _H264_GOP, _GpuH264Encoder
    _GPU_DEVICE   = device
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() == False — GPU encode requires CUDA")
    _TORCH = torch
    from gpu_h264_codec import GpuH264Encoder
    _GpuH264Encoder = GpuH264Encoder
    _H264_BITRATE   = int(h264_bitrate)
    _H264_GOP       = 1 if h264_all_intra else 30
    print(f"  Encoder:    NVENC h264 on {device} "
          f"({'all-intra' if h264_all_intra else 'IPPP gop=30'}, "
          f"{_H264_BITRATE / 1e6:.0f} Mbps) — NVDEC decode at the rectifier")


def _rgba_to_gpu_rgb_chw(img_rgba):
    """Upload an RGBA uint8 (H, W, 4) numpy view to GPU and convert to
    contiguous CHW RGB uint8 — the shared GPU representation for both
    encoders. .copy() is required because np.frombuffer views are
    read-only; the .to() is non-blocking, but downstream torch kernels
    on the same default stream implicitly wait for it."""
    rgba_t = _TORCH.from_numpy(img_rgba.copy()).to(_GPU_DEVICE,
                                                   non_blocking=True)
    return rgba_t[..., :3].permute(2, 0, 1).contiguous()


def _h264_encoder_for(eye: int, h: int, w: int):
    """Lazy per-eye NVENC encoder — one persistent stream per eye, built at the
    frame's resolution on first use."""
    enc = _H264_ENC.get(eye)
    if enc is None:
        enc = _GpuH264Encoder(width=w, height=h, gop=_H264_GOP,
                              bitrate=_H264_BITRATE, device=_GPU_DEVICE)
        _H264_ENC[eye] = enc
    return enc


def _encode_h264_from_gpu_rgba(rgba_hwc, eye: int) -> bytes:
    """(H,W,4) uint8 CUDA RGBA (FD-export) -> H.264 bytes. NVENC consumes the
    RGBA surface directly and does the RGB->YUV CSC on the encode engine — no
    SM colour convert. May return b'' if NVENC is briefly buffering."""
    h, w = int(rgba_hwc.shape[0]), int(rgba_hwc.shape[1])
    return _h264_encoder_for(eye, h, w).encode(rgba_hwc)


def _encode_h264_gpu(img_rgba, eye: int) -> bytes:
    """(H,W,4) uint8 host RGBA (host capture path) -> GPU -> H.264 via NVENC."""
    t = _TORCH.from_numpy(img_rgba).to(_GPU_DEVICE, non_blocking=True)
    return _encode_h264_from_gpu_rgba(t, eye)


def _camera_info_dict(K: dict, frame_id: str) -> dict:
    """Build a ROS 2 CameraInfo YAML dict from a single-eye intrinsics dict.

    Format matches `ros2 topic echo /camera/camera_info > file.yaml` output
    (flat keys: width/height/k/d/r/p/distortion_model/frame_id/binning_*/roi).
    Downstream consumers — roboreg's parse_camera_info, image_pipeline,
    image_proc — all accept this layout directly, so there's no
    normalization step needed on the read side.
    """
    fx, fy = float(K['fx']), float(K['fy'])
    cx, cy = float(K['cx']), float(K['cy'])
    w, h   = int(K['w']),    int(K['h'])
    return {
        'binning_x': 0,
        'binning_y': 0,
        # plumb_bob with 5 zeros: our OpenXR intrinsics come from ideal
        # pinhole FoV angles, so there is no distortion to model.
        'd': [0.0, 0.0, 0.0, 0.0, 0.0],
        'distortion_model': 'plumb_bob',
        'frame_id': frame_id,
        'height': h,
        'k': [fx, 0.0, cx,
              0.0, fy, cy,
              0.0, 0.0, 1.0],
        'p': [fx, 0.0, cx, 0.0,
              0.0, fy, cy, 0.0,
              0.0, 0.0, 1.0, 0.0],
        'r': [1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0],
        'roi': {
            'do_rectify': False,
            'height': 0,
            'width':  0,
            'x_offset': 0,
            'y_offset': 0,
        },
        'width': w,
    }


def _dump_yaml(path: str, info: dict):
    try:
        import yaml as _yaml
        with open(path, 'w') as f:
            # sort_keys=True → alphabetical key order, matches the
            # `ros2 topic echo > file.yaml` reference layout.
            _yaml.dump(info, f, default_flow_style=False, sort_keys=True)
    except Exception:
        with open(path, 'w') as f:
            json.dump(info, f, indent=2, sort_keys=True)


def save_camera_info_yaml_stereo(out_dir: str, snap: dict):
    """Write both eyes' camera_info files plus a single combined one.

    Produces (under `out_dir`):
        camera_info.yaml         → LEFT eye (back-compat with single-eye consumers)
        camera_info_left.yaml    → LEFT eye (explicit)
        camera_info_right.yaml   → RIGHT eye

    Each file is a flat ROS 2 CameraInfo YAML — same schema as
    `ros2 topic echo /xr/image_left/camera_info > file.yaml`.
    """
    if 0 in snap:
        _dump_yaml(os.path.join(out_dir, 'camera_info.yaml'),
                   _camera_info_dict(snap[0], 'xr_left'))
        _dump_yaml(os.path.join(out_dir, 'camera_info_left.yaml'),
                   _camera_info_dict(snap[0], 'xr_left'))
    if 1 in snap:
        _dump_yaml(os.path.join(out_dir, 'camera_info_right.yaml'),
                   _camera_info_dict(snap[1], 'xr_right'))


def setup_output_dir(arg_path: str | None) -> str:
    """Create per-session output directory and its subdirectories."""
    if arg_path:
        out = os.path.expanduser(arg_path)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.expanduser(f"~/xr_captures/{ts}")
    for sub in ["images/left", "images/right",
                "joint_states/left",
                "extrinsics", "base_world", "camera_world"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    print(f"[capture] Output directory: {out}")
    return out


class CaptureTrigger:
    """Edge-triggered hold detector for the left trigger.

    Fires once per held-press: trigger value must rise above 0.9 and stay
    there for `hold_sec`, then a single capture event is emitted. Releasing
    below 0.5 re-arms for the next press.
    """

    def __init__(self, hold_sec: float):
        self.hold_sec = float(hold_sec)
        self._press_started_at: float | None = None
        self._fired_this_press = False

    def update(self, l_trig_value: float) -> bool:
        """Call every tick. Returns True exactly once per qualifying press."""
        now = time.monotonic()
        pressed = l_trig_value > 0.9
        if pressed:
            if self._press_started_at is None:
                self._press_started_at = now
            held_for = now - self._press_started_at
            if (not self._fired_this_press
                    and held_for >= self.hold_sec):
                self._fired_this_press = True
                return True
        else:
            # Re-arm only after release
            if l_trig_value < 0.5:
                self._press_started_at = None
                self._fired_this_press = False
        return False




class RosBridge:
    """Sidecar-based ROS2 publisher.

    rclpy's compiled extension on ROS2 Jazzy is built for Python 3.12,
    but Isaac Sim runs Python 3.11. To bridge that gap we run a
    sidecar subprocess (xr_ros_bridge.py) under the system Python 3.12
    with ROS2 sourced, and feed it capture data over a UNIX socket.
    The sidecar does all rclpy work; this class just spawns it,
    connects, and ships binary records.

    Wire protocol: see xr_ros_bridge.py docstring for the byte layout.
    """

    MAGIC            = 0x42525258         # 'XRRB' little-endian
    VERSION          = 1
    TYPE_CAMERA_INFO = 1
    TYPE_CAPTURE     = 2
    TYPE_GOODBYE     = 3
    HEADER_FMT       = "<IIII"
    CAMERA_INFO_FMT  = "<II4fII4f"
    QUEUE_SIZE       = 8

    def __init__(self, script_path: str, setup_path: str,
                 socket_path: str, namespace: str, frame_id_world: str,
                 joint_names: list, intrinsics_snapshot_path: str,
                 python_bin: str = "python3.12", image_codec: str = "h264"):
        self.script_path        = script_path
        self.setup_path         = setup_path
        self.socket_path        = socket_path
        self.namespace          = namespace
        self.frame_id_world     = frame_id_world
        self.joint_names        = list(joint_names)
        self.intrinsics_path    = intrinsics_snapshot_path
        self.python_bin         = python_bin
        self.image_codec        = image_codec

        self._proc: "subprocess.Popen | None"   = None
        self._sock: "socket.socket | None"      = None
        self._queue                              = queue.Queue(maxsize=self.QUEUE_SIZE)
        self._thread: "threading.Thread | None" = None
        self._shutdown                           = threading.Event()
        self._info_sent                          = False
        self._drops                              = 0
        self._published                          = 0

    def start(self):
        """Spawn the sidecar, wait for its socket, connect, send
        camera_info if available, and start the publisher thread."""
        import shlex
        if not os.path.exists(self.script_path):
            raise RuntimeError(f"sidecar script not found: {self.script_path}")
        if not os.path.exists(self.setup_path):
            raise RuntimeError(f"ROS2 setup not found: {self.setup_path}")
        # Old socket might be left over from a crashed previous run.
        try: os.unlink(self.socket_path)
        except FileNotFoundError: pass

        # bash -c 'source ros && exec <py> sidecar ...' — sourcing
        # sets AMENT_PREFIX_PATH / PYTHONPATH for rclpy; exec replaces
        # bash so signals reach Python directly. We use explicit
        # python_bin (default python3.12) because PATH may have a
        # mismatched venv Python first.
        cmd = (f"source {shlex.quote(self.setup_path)} && "
               f"exec {shlex.quote(self.python_bin)} -u "
               f"{shlex.quote(self.script_path)} "
               f"--socket {shlex.quote(self.socket_path)} "
               f"--namespace {shlex.quote(self.namespace)} "
               f"--frame {shlex.quote(self.frame_id_world)} "
               f"--joint-names {shlex.quote(','.join(self.joint_names))} "
               f"--image-codec {shlex.quote(self.image_codec)}")
        # Clean env: keep only essentials, and strip venv-ish entries
        # from PATH so the sidecar's `python3.12` lookup hits the
        # system binary, not isaac_env's bin/.
        keep_keys = ("PATH", "HOME", "USER", "LANG", "LC_ALL",
                     "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION",
                     "DISPLAY", "XAUTHORITY", "XR_RAW_IMAGES",
                     "FASTRTPS_DEFAULT_PROFILES_FILE",
                     "FASTDDS_DEFAULT_PROFILES_FILE")
        sub_env = {k: os.environ[k] for k in keep_keys if k in os.environ}
        if "PATH" in sub_env:
            sub_env["PATH"] = ":".join(
                p for p in sub_env["PATH"].split(":")
                if "/isaac_env" not in p and "/venv" not in p
                and "/.virtualenvs/" not in p
            ) or "/usr/bin:/bin"
        else:
            sub_env["PATH"] = "/usr/bin:/bin"
        self._proc = subprocess.Popen(["bash", "-c", cmd], env=sub_env,
                                      stdout=sys.stdout, stderr=sys.stderr)
        print(f"[ros-bridge] sidecar spawned (pid {self._proc.pid}), "
              f"waiting for {self.socket_path}…")

        # Wait for the sidecar to bind its socket (up to 15 s; rclpy's
        # first-import does some work on a fresh shell).
        deadline = time.monotonic() + 15.0
        while not os.path.exists(self.socket_path):
            if self._proc.poll() is not None:
                raise RuntimeError(f"sidecar exited early (code "
                                   f"{self._proc.returncode}); see its "
                                   f"output above for the cause")
            if time.monotonic() > deadline:
                self._proc.terminate()
                raise RuntimeError(f"sidecar didn't bind {self.socket_path} "
                                   f"within 15 s; sourcing ROS2 may have "
                                   f"failed silently")
            time.sleep(0.1)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)
        # Try sending camera_info immediately; if the intrinsics snapshot
        # isn't ready yet, _run will retry on each capture.
        self._maybe_send_camera_info()

        self._thread = threading.Thread(target=self._run,
                                        name="ros-bridge-tx", daemon=True)
        self._thread.start()

    def publish(self, snap: dict, enc_l: bytes, enc_r: bytes,
                H_eye_L_world, H_eye_R_world) -> bool:
        """Enqueue one capture for the sidecar. Non-blocking; drops
        on full queue (slow consumer)."""
        try:
            self._queue.put_nowait({
                "idx":       int(snap["idx"]),
                "joint_rad": np.asarray(snap["joint_rad"],   dtype=np.float32).copy(),
                "H_base":    np.asarray(snap["H_base_world"], dtype=np.float32).flatten().copy(),
                "H_eye_L":   np.asarray(H_eye_L_world,        dtype=np.float32).flatten().copy(),
                "H_eye_R":   np.asarray(H_eye_R_world,        dtype=np.float32).flatten().copy(),
                "enc_l":     bytes(enc_l),
                "enc_r":     bytes(enc_r),
            })
            return True
        except queue.Full:
            self._drops += 1
            if self._drops == 1 or self._drops % 30 == 0:
                print(f"[ros-bridge] queue full, dropped {self._drops} "
                      f"frames (sidecar slow?)")
            return False

    def stop(self, timeout: float = 3.0):
        """Send goodbye, close socket, wait for sidecar to exit."""
        if self._thread is None:
            return
        self._shutdown.set()
        try:
            if self._sock is not None:
                header = struct.pack(self.HEADER_FMT, self.MAGIC, self.VERSION,
                                     self.TYPE_GOODBYE, 0)
                self._sock.sendall(header)
        except Exception:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None
        if self._sock is not None:
            try: self._sock.close()
            except Exception: pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try: self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired: self._proc.kill()
            self._proc = None
        try: os.unlink(self.socket_path)
        except FileNotFoundError: pass
        print(f"[ros-bridge] stopped — published {self._published}, "
              f"dropped {self._drops}")

    # ── Internal ─────────────────────────────────────────────────

    def _maybe_send_camera_info(self):
        """Read /tmp/xr_intrinsics_snapshot.json once and ship its
        contents as a CameraInfo record. Cheap retry — no-op if the
        file isn't ready yet (the intrinsics one-shot may still be
        in warmup).

        On-disk shape (written by `IntrinsicsCapture._run`):

            {"K": {"0": {...left...}, "1": {...right...}}}

        The integer eye indices from OpenXR (0=left, 1=right) become
        string keys after `json.dump`. Earlier versions of this method
        read `d.get("left")` / `d.get("right")` — neither key exists,
        so both came back None and the CAMERA_INFO opcode never got
        sent. The bridge's `_caminfo_msg_l` stayed None forever and
        the rosbag camera_info topic recorded zero messages.
        """
        if self._info_sent or self._sock is None:
            return
        try:
            with open(self.intrinsics_path) as f:
                d = json.load(f)
        except Exception:
            return
        K = d.get("K") or {}
        L = K.get("0") or K.get(0)
        R = K.get("1") or K.get(1)
        if not L or not R:
            return
        payload = struct.pack(self.CAMERA_INFO_FMT,
            int(L["w"]),  int(L["h"]),
            float(L["fx"]), float(L["fy"]), float(L["cx"]), float(L["cy"]),
            int(R["w"]),  int(R["h"]),
            float(R["fx"]), float(R["fy"]), float(R["cx"]), float(R["cy"]))
        header = struct.pack(self.HEADER_FMT, self.MAGIC, self.VERSION,
                             self.TYPE_CAMERA_INFO, len(payload))
        try:
            self._sock.sendall(header + payload)
            self._info_sent = True
            print("[ros-bridge] CAMERA_INFO opcode sent to sidecar "
                  f"(L: {L['w']}×{L['h']} fx={L['fx']:.1f}, "
                  f"R: {R['w']}×{R['h']} fx={R['fx']:.1f})")
        except Exception as e:
            print(f"[ros-bridge] camera_info send failed: {e}")

    def _run(self):
        """TX loop: dequeue capture items and ship as binary records."""
        while not self._shutdown.is_set():
            if not self._info_sent:
                self._maybe_send_camera_info()
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            payload = b"".join([
                struct.pack("<I", item["idx"]),
                struct.pack("<d", time.time()),
                item["joint_rad"].tobytes(),
                item["H_base"].tobytes(),
                item["H_eye_L"].tobytes(),
                item["H_eye_R"].tobytes(),
                struct.pack("<I", len(item["enc_l"])),
                item["enc_l"],
                struct.pack("<I", len(item["enc_r"])),
                item["enc_r"],
            ])
            header = struct.pack(self.HEADER_FMT, self.MAGIC, self.VERSION,
                                 self.TYPE_CAPTURE, len(payload))
            try:
                self._sock.sendall(header + payload)
                self._published += 1
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[ros-bridge] sidecar disconnected: {e}; stopping")
                self._shutdown.set()
                break
            except Exception as e:
                print(f"[ros-bridge] send error: {e}")
                self._shutdown.set()
                break



class LayerFrameCapture:
    """Synchronous XR frame extraction via XR_APILAYER_FRAME_GRAB.

    The sim tick calls capture(out_dir, snap), which performs the full
    GET → recv → encode → dispatch-writes pipeline inline. No worker
    thread, no queue. Trade-off: sim tick blocks ~25 ms per capture;
    benefit: every per-frame artifact is consistent with the moment
    the GET was sent (no T_sim/T_xr drift).

    Wire protocol, snap-dict schema, and rationale: see ARCHITECTURE.md.
    """

    HEADER_MAGIC = b"XRFR"   # 0x52465258 little-endian
    FMT_R8G8B8A8_SRGB = 1

    def __init__(self, socket_path: str, profile: bool = False,
                 ros_bridge: "RosBridge | None" = None):
        self.socket_path = socket_path
        self._sock: "socket.socket | None" = None
        self._ready = False
        self._captured = 0
        self._failed = 0
        # If True, capture() prints per-frame phase-timing breakdown
        # (sendGET / recv / encode / publish / total).
        self._profile = profile
        # Simple per-eye recv buffers — single bytearray each, reused
        # for every capture. 32 MB is plenty for Quest 3 native
        # 2048×1792×4 = 14.7 MB per eye.
        _BUF_SIZE = 32 * 1024 * 1024
        self._bufs = [bytearray(_BUF_SIZE), bytearray(_BUF_SIZE)]
        # Optional ROS2 publisher; capture() enqueues per-frame data.
        self._ros_bridge = ros_bridge
        # ── FD-export validation (XR_FRAME_EXPORT_FD=1): CUDA-import the layer's
        # exportable buffers and byte-compare against the host pixels. Additive;
        # the host pixel path above stays the ground-truth oracle.
        import os as _os
        self._export     = _os.environ.get("XR_FRAME_EXPORT_FD", "").startswith("1")
        # TEST: publish raw rgb8 instead of H.264 (host path; mutually
        # exclusive with export, enforced at startup).
        self._raw_images = _os.environ.get("XR_RAW_IMAGES", "").startswith("1")
        self._cmp_active = self._export
        self._ext        = {}       # slot_idx -> (cudaExternalMemory, devPtr)
        self._sem        = None     # imported timeline semaphore
        self._gpu_buf    = None     # torch cuda scratch for the D2D consume
        self._cmp_frames = 0
        self._cmp_ok     = 0
        # Stereo baseline (HT_right_to_left.npy) is a fixed rig
        # parameter — saved once on the first capture, not per frame.
        self._baseline_saved = False

    def start(self):
        """Connect to the layer's frame socket. Called once after Isaac
        Sim's XR session has been started (i.e. after the user clicks
        Start AR), so we poll briefly for the socket to appear."""
        deadline = time.monotonic() + 10.0
        while not os.path.exists(self.socket_path):
            if time.monotonic() > deadline:
                print(f"[layer-capture] socket {self.socket_path} did not "
                      f"appear within 10 s. Is XR_APILAYER_FRAME_GRAB "
                      f"enabled and has Start AR been clicked?")
                return False
            time.sleep(0.2)
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(self.socket_path)
            self._sock.settimeout(30.0)
            # Ask for a 32 MB recv buffer. Linux silently caps this at
            # net.core.rmem_max, which defaults to ~208 KB — and a small
            # buffer is the most likely cause of slow socket recv (each
            # 14.7 MB stereo eye gets dribbled in ~200 KB chunks, with
            # a context switch per chunk → 60+ ms per eye).
            asked = 32 * 1024 * 1024
            try:
                self._sock.setsockopt(socket.SOL_SOCKET,
                                      socket.SO_RCVBUF, asked)
            except OSError:
                pass
            # Linux returns 2× the actual buffer from getsockopt (kernel
            # bookkeeping overhead). Divide by 2 for the usable size.
            got_raw = self._sock.getsockopt(socket.SOL_SOCKET,
                                            socket.SO_RCVBUF)
            got = got_raw // 2
            note = ""
            if got < asked // 2:    # got less than half what we asked
                note = ("   ⚠  capped by net.core.rmem_max — try "
                        "`sudo sysctl -w net.core.rmem_max=67108864` "
                        "and rerun")
            print(f"[layer-capture] SO_RCVBUF: asked "
                  f"{asked // (1024*1024)} MB, got {got // 1024} KB"
                  f"{note}")
            self._ready = True
            print(f"[layer-capture] connected to {self.socket_path} (synchronous)")
            return True
        except Exception as e:
            print(f"[layer-capture] failed to connect: {e}")
            self._sock = None
            return False

    def stop(self):
        """Close the socket and the ROS bridge if any. No threads to
        join — capture runs on the sim tick."""
        if self._ros_bridge is not None:
            self._ros_bridge.stop()
            self._ros_bridge = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._ready = False
        print(f"[layer-capture] stopped "
              f"({self._captured} captured, {self._failed} failed)")

    def capture(self, out_dir: str, snap: dict) -> bool:
        """One capture cycle on the sim tick: GET -> recv per-eye headers +
        poses + pixels -> NVENC H.264 encode -> publish to the ROS bridge. No disk
        artifacts (passthrough is publish-only). Returns False on any protocol
        or I/O error. Set self._profile to log per-phase ms."""
        if not self._ready:
            print(f"[layer-capture] not ready — call start() first")
            return False
        import struct
        idx = snap["idx"]
        t = time.perf_counter if self._profile else (lambda: 0.0)
        t0 = t()
        try:
            self._sock.sendall(b"GET\n")
            t_get = t()

            eyes: "dict[int, np.ndarray]" = {}
            eye_poses: "dict[int, np.ndarray]" = {}
            for _ in range(2):
                # Header: 7 uint32 — magic, version, eye_id, w, h, fmt, nbytes
                hdr = self._recv_exact(7 * 4)
                magic, version, eye_id, w, h, fmt, nbytes = struct.unpack(
                    "<7I", hdr)
                if magic != int.from_bytes(self.HEADER_MAGIC, "little"):
                    print(f"[layer-capture] bad magic idx={idx}: 0x{magic:08x}")
                    return False
                if version != 2:
                    print(f"[layer-capture] wire protocol mismatch: got "
                          f"version {version}, expected 2 — rebuild the "
                          f"OpenXR frame layer.")
                    return False
                if fmt != self.FMT_R8G8B8A8_SRGB:
                    print(f"[layer-capture] unsupported format idx={idx}: {fmt}")
                    return False
                if self._export and nbytes == 0:
                    pass                          # pixels come from exp_buf (GPU)
                elif nbytes != w * h * 4:
                    print(f"[layer-capture] size mismatch idx={idx}: "
                          f"nbytes={nbytes} vs w*h*4={w*h*4}")
                    return False

                # 7 float32 pose
                pose_bytes = self._recv_exact(7 * 4)
                px, py, pz, qw, qx, qy, qz = struct.unpack("<7f", pose_bytes)
                eye_poses[eye_id] = _pose_to_H(
                    Pose(pos=(px, py, pz), quat_wxyz=(qw, qx, qy, qz)))

                # Pixel recv (host path only; export mode sends nbytes=0).
                if nbytes:
                    view = memoryview(self._bufs[eye_id])
                    got = 0
                    while got < nbytes:
                        chunk = self._sock.recv_into(view[got:nbytes],
                                                     nbytes - got)
                        if chunk == 0:
                            raise EOFError(
                                f"layer socket closed mid-pixel-recv: "
                                f"{got}/{nbytes}")
                        got += chunk
                    eyes[eye_id] = np.frombuffer(self._bufs[eye_id],
                                                 dtype=np.uint8,
                                                 count=nbytes).reshape(h, w, 4)
            t_recv = t()

            if not self._export and (0 not in eyes or 1 not in eyes):
                print(f"[layer-capture] missing eye idx={idx}: {list(eyes)}")
                return False

            # Encode both eyes. FD-export pulls pixels from the GPU exportable
            # buffer (no host pixels were received; _export_encode also drains
            # the wire tail and waits on the timeline). Host mode uses numpy.
            if self._raw_images:
                # TEST: no compression — ship raw rgb8 bytes straight from the
                # host eye buffers (drop alpha). tobytes() is C-order, so this
                # is row-major rgb8 the bridge stamps into an Image.
                enc_bytes_l = np.ascontiguousarray(eyes[0][:, :, :3]).tobytes()
                enc_bytes_r = np.ascontiguousarray(eyes[1][:, :, :3]).tobytes()
            elif self._export:
                enc_bytes_l, enc_bytes_r = self._export_encode(w, h)
            else:
                enc_bytes_l = _encode_h264_gpu(eyes[0], 0)
                enc_bytes_r = _encode_h264_gpu(eyes[1], 1)
            t_enc = t()

            # Publish to the ROS bridge. World-rig form of the eye poses gives
            # the bridge the per-eye world poses it derives the baseline from.
            if self._ros_bridge is not None:
                H_eye_L = _ref_pose_to_world_rig(eye_poses[0])
                H_eye_R = _ref_pose_to_world_rig(eye_poses[1])
                self._ros_bridge.publish(snap, enc_bytes_l, enc_bytes_r,
                                         H_eye_L, H_eye_R)
            t_end = t()
            self._captured += 1

            if self._profile:
                ms = lambda a, b: (b - a) * 1000.0
                print(f"[capture #{idx:04d} time] "
                      f"sendGET {ms(t0, t_get):5.2f} ms  "
                      f"recv {ms(t_get, t_recv):6.2f} ms  "
                      f"encode {ms(t_recv, t_enc):5.2f} ms  "
                      f"publish {ms(t_enc, t_end):5.2f} ms  "
                      f"total {ms(t0, t_end):6.2f} ms")
            return True
        except Exception as e:
            self._failed += 1
            print(f"[layer-capture] capture error idx={idx}: {e}")
            return False

    def _export_encode(self, w, h):
        """FD-export live consume (Stage 3b). The layer appends an 8xint32 tail
        after the (pixel-less) eye headers: {EXPORT_MAGIC, slot, seq_lo, seq_hi,
        fd0, fd1, fd2, sem_fd}. Import the 3 exportable buffers + timeline
        semaphore once, wait on the timeline for this frame's seq, D2D-copy the
        picked slot into a GPU scratch tensor, and NVENC both eyes straight
        from the GPU. No host pixels, no H2D. Returns (h264_left, h264_right)."""
        import os, struct
        import torch
        from cuda.bindings import runtime as cudart

        tail = self._recv_exact(8 * 4)            # drained first -> wire stays synced
        (magic2, slot_idx, seq_lo, seq_hi,
         fd0, fd1, fd2, sem_fd) = struct.unpack("<8i", tail)
        if magic2 != 0x52465845:                  # "EXFR"
            raise RuntimeError(f"bad export tail magic 0x{magic2:08x}")
        seq  = ((seq_hi & 0xffffffff) << 32) | (seq_lo & 0xffffffff)
        bpe  = w * h * 4
        need = bpe * 2

        def _ck(err, what):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"{what}: {err}")

        if not self._ext:
            for si, fd in enumerate((fd0, fd1, fd2)):
                d = cudart.cudaExternalMemoryHandleDesc()
                d.type = cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd
                d.handle.fd = os.dup(fd)           # CUDA takes ownership of this dup
                d.size = need
                err, ext = cudart.cudaImportExternalMemory(d)
                _ck(err, f"cudaImportExternalMemory slot{si}")
                bd = cudart.cudaExternalMemoryBufferDesc()
                bd.offset, bd.size, bd.flags = 0, need, 0
                err, dptr = cudart.cudaExternalMemoryGetMappedBuffer(ext, bd)
                _ck(err, f"cudaExternalMemoryGetMappedBuffer slot{si}")
                self._ext[si] = (ext, dptr)
            ds = cudart.cudaExternalSemaphoreHandleDesc()
            ds.type = cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
            ds.handle.fd = os.dup(sem_fd)
            err, self._sem = cudart.cudaImportExternalSemaphore(ds)
            _ck(err, "cudaImportExternalSemaphore")
            self._gpu_buf = torch.empty(need, dtype=torch.uint8, device="cuda")
            print(f"[export] LIVE GPU-resident path: imported 3 buffers "
                  f"({need} bytes) + timeline semaphore; host readback disabled")

        # Wait for the Vulkan copy of this frame to finish (no host fence).
        wp = cudart.cudaExternalSemaphoreWaitParams()
        wp.params.fence.value = seq
        err, = cudart.cudaWaitExternalSemaphoresAsync([self._sem], [wp], 1, 0)
        _ck(err, "cudaWaitExternalSemaphoresAsync")

        # D2D copy picked slot -> scratch (synchronous; full barrier), then
        # NVENC each eye directly from the GPU tensor.
        _, dptr = self._ext[slot_idx]
        err, = cudart.cudaMemcpy(self._gpu_buf.data_ptr(), int(dptr), need,
                                 cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        _ck(err, "cudaMemcpy D2D")
        eye0 = self._gpu_buf[0:bpe].view(h, w, 4)
        eye1 = self._gpu_buf[bpe:2 * bpe].view(h, w, 4)

        if self._cmp_frames < 3:
            print(f"[export] frame {self._cmp_frames + 1} slot={slot_idx} "
                  f"seq={seq} GPU means L={float(eye0.float().mean()):.1f} "
                  f"R={float(eye1.float().mean()):.1f}")
        self._cmp_frames += 1

        return (_encode_h264_from_gpu_rgba(eye0, 0),
                _encode_h264_from_gpu_rgba(eye1, 1))

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly n bytes from the socket; raise EOFError if the
        peer closes early."""
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            chunk = self._sock.recv_into(view[got:], n - got)
            if chunk == 0:
                raise EOFError("layer socket closed")
            got += chunk
        return bytes(buf)


# ═════════════════════════════════════════════════════════════════════════════
# Pretty printers
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_intr(eye_id, k):
    label = {0: "L", 1: "R"}.get(eye_id, f"V{eye_id}")
    return (f"[eye {label}] {k['w']}×{k['h']}  "
            f"fx={k['fx']:7.2f} fy={k['fy']:7.2f}  "
            f"cx={k['cx']:7.2f} cy={k['cy']:7.2f}")


def _fmt_joints(j):
    return "[" + ", ".join(f"{v:+6.1f}°" for v in j) + "]"


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    from isaacsim.core.api import World

    # ── Prepare URDF first (fails fast if ROS env not sourced) ──
    urdf_path = prepare_urdf(args.ros_package, args.xacro_path, args.urdf_file)
    if urdf_path is None:
        print("[fatal] Could not prepare URDF. Source your ROS workspace, then retry.")
        simulation_app.close()
        sys.exit(1)

    # ── Parse CLI numerics ──
    try:
        parts = [float(v) for v in args.base_pose.split(",")]
        if len(parts) == 2:
            base_xy = np.array(parts, dtype=np.float64)
            base_z_user = None
        elif len(parts) == 3:
            base_xy = np.array(parts[:2], dtype=np.float64)
            base_z_user = float(parts[2])
        else:
            raise ValueError("need 'X,Y' or 'X,Y,Z'")
    except Exception as e:
        print(f"[fatal] --base-pose invalid: {e}")
        simulation_app.close()
        sys.exit(1)

    try:
        if args.initial_joints_file:
            joint_deg = load_initial_joints_csv(args.initial_joints_file)
            print(f"[joints] loaded initial joints from "
                  f"{args.initial_joints_file} (radians → degrees)")
        else:
            joint_deg = np.array(
                [float(v) for v in args.initial_joints.split(",")],
                dtype=np.float64,
            )
        if joint_deg.size != 7:
            raise ValueError(f"need 7 angles, got {joint_deg.size}")
    except Exception as e:
        print(f"[fatal] --initial-joints / --initial-joints-file invalid: {e}")
        simulation_app.close()
        sys.exit(1)

    # ── Build scene ──
    world = World(stage_units_in_meters=1.0)

    or_mode = args.or_environment is not None
    if or_mode:
        load_or_environment(args.or_environment)
    else:
        build_default_scene()

    # Determine base Z. Matches sim_isaac_node.py: in OR mode the trolley
    # raises the base to trolley_height; otherwise robot sits on the floor.
    # User can override with the 3-value form of --base-pose.
    if base_z_user is not None:
        base_z = base_z_user
    elif or_mode:
        base_z = float(args.trolley_height)
    else:
        base_z = 0.0
    base_xyz = np.array([base_xy[0], base_xy[1], base_z], dtype=np.float64)
    print(f"[Isaac] Base pose: ({base_xyz[0]:.3f}, {base_xyz[1]:.3f}, "
          f"{base_xyz[2]:.3f})  "
          f"[{'OR + trolley' if or_mode and base_z_user is None else 'manual'}]")

    base_yaw = 0.0
    robot_path = import_robot(urdf_path, base_xyz, base_yaw, apply_pbr=or_mode)
    spawn_trolley(robot_path, base_height=float(base_xyz[2]))

    world.reset()
    apply_joint_targets(robot_path, joint_deg)
    print(f"[Isaac] Initial joints: {_fmt_joints(joint_deg)}")

    # ── RTX viewport settings (matches sim_isaac_node.py) ──
    # Auto-exposure is the load-bearing one; others harmlessly explicit.
    try:
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool('/rtx/reflections/enabled', True)
        settings.set_bool('/rtx/ambientOcclusion/enabled', True)
        settings.set_bool('/rtx/indirectDiffuse/enabled', True)
        settings.set_bool('/rtx/shadows/enabled', True)
        settings.set_bool('/rtx/directLighting/enabled', True)
        settings.set_bool('/rtx/translucency/enabled', True)
        settings.set_bool('/rtx/post/histogram/enabled', True)  # auto-exposure
        for _ in range(5):
            simulation_app.update()
        print("[Isaac] RTX features set (reflections, AO, GI, shadows, auto-exposure)")
    except Exception as e:
        print(f"[Isaac] RTX setup failed: {e}")

    # ── XR foveation override ──────────────────────────────────────────────
    # Disable Omniverse Kit's carb-side foveation pass. Belt-and-braces;
    # the load-bearing fix is XR_FRAME_STRIP_FOVEATION=1 at the API layer.
    # See XR_CAPTURE_NOTES.md for the pipeline diagram and diagnosis.
    if args.foveation_mode != "default":
        try:
            mode = args.foveation_mode
            # Cover vr/ar/tabletar — unused profiles ignore silently.
            # Set both transient and /persistent variants.
            profiles = ("vr", "ar", "tabletar")
            paths = []
            for p in profiles:
                paths.append(f"/xr/profile/{p}/foveation/mode")
                paths.append(f"/persistent/xr/profile/{p}/foveation/mode")
            for pth in paths:
                settings.set(pth, mode)
            print(f"[Isaac] XR foveation mode = '{mode}' "
                  f"(applied to vr/ar/tabletar profiles, "
                  f"transient + persistent)")
        except Exception as e:
            print(f"[Isaac] XR foveation override failed: {e}")

    # Silence Kit's per-frame "invalid foveation warped dims 0x0" errors.
    # These are emitted by the fallback path that produces our pinhole
    # output — they're load-bearing, not a bug. See XR_CAPTURE_NOTES.md.
    try:
        settings.set("/log/channels/omni.kit.xr.system.openxr.plugin/level",
                     "fatal")
        print("[Isaac] silenced omni.kit.xr.system.openxr.plugin log channel "
              "(foveation-dim warnings)")
    except Exception as e:
        print(f"[Isaac] log channel silence failed: {e}")

    # AppLauncher injects `xr` into args; default False if missing
    xr_enabled = bool(getattr(args, "xr", False))

    controller: ControllerReader | None = None
    if xr_enabled:
        controller = ControllerReader()

    joint_mode = bool(args.joint_mode)
    if xr_enabled:
        print(f"[input] starting in {'JOINT' if joint_mode else 'TROLLEY'} mode "
              f"(hold BOTH GRIPS 0.5s to toggle)")
    else:
        print("[input] non-XR validation mode: robot stays at initial pose.")

    intrinsics: IntrinsicsOneShot | None = None
    intrinsics_reported = False

    # ── Capture state (XR mode + capture enabled) ───
    capture_mode = args.capture_mode if xr_enabled else "off"
    layer_capture: "LayerFrameCapture | None" = None
    capture_trigger: CaptureTrigger | None = None
    capture_out_dir: str | None = None
    capture_idx = 0
    continuous_capture = False    # toggled by right-A start / right-B stop
    cont_start_idx = 0            # capture_idx at start of current continuous run
    # Identifier for the current continuous run (used to name the .h265
    # files in video mode). None when not in a continuous run.
    continuous_run_id: "str | None" = None
    if capture_mode != "off":
        capture_trigger = CaptureTrigger(hold_sec=args.capture_trigger_hold)
        # Initialise the NVENC H.264 encoder (per-eye instances are built
        # lazily on the first frame, at the frame's resolution).
        _init_encoder(args.device,
                      h264_bitrate=args.h264_bitrate,
                      h264_all_intra=not args.h264_ippp)

    print("=" * 70)
    print(f"  Mode:  {'XR (CloudXR + Quest)' if xr_enabled else 'NON-XR (functional validation)'}")
    print(f"  Scene: {'OR environment' if or_mode else 'default (ground + dome + distant)'}")
    print(f"  Robot: {urdf_path}")
    if xr_enabled:
        print("  Start AR in Isaac Sim UI, connect Quest, then:")
        print("    TROLLEY mode (default):")
        print("      Left  thumb X/Y → base in world X/Y")
        print("      Right thumb X   → base yaw")
        if capture_mode != "off":
            print(f"      Left trigger held >{args.capture_trigger_hold:.1f}s → capture ONE stereo frame")
            print(f"      Right A button → START continuous capture")
            print(f"      Right B button → STOP  continuous capture")
        print("    JOINT mode:")
        print("      Left  thumb X/Y → joints A1 / A2 ±")
        print("      Right thumb X/Y → joints A3 / A4 ±")
        print("      Triggers L / R  → joint  A5 ±")
        print("      Grips    L / R  → joint  A6 ±")
        print("      Buttons  A / B  → joint  A7 ±")
        print("    Mode toggle: hold BOTH GRIPS for 0.5 s")
        print(f"  Intrinsics: one-shot after {args.intrinsics_warmup_sec:.1f}s warm-up,")
        print(f"              snapshot → {args.intrinsics_snapshot_path}")
        print(f"  Capture:    mode={capture_mode}")
    else:
        print("  Non-XR validation: robot at initial pose, scene renders only.")
        print("  Ctrl+C to exit.")
    print("=" * 70)

    JOINT_LIMIT_DEG = 170.0  # soft clamp

    # Producer frame-rate cap (XR_PUBLISH_FPS=N): throttle the WHOLE sim loop —
    # render (simulation_app.update) + capture — to at most N fps. The render is
    # the dominant SM load (producer-alone pins ~85% SM), so slowing the loop is
    # what frees the GPU for the consumer's optimiser; capping only the publish
    # does nothing because Isaac keeps rendering. 0 / unset = uncapped.
    try:
        _cap_fps = float(os.environ.get("XR_PUBLISH_FPS", "") or 0.0)
    except ValueError:
        _cap_fps = 0.0
    _frame_min_interval = (1.0 / _cap_fps) if _cap_fps > 0.0 else 0.0
    _loop_t = time.perf_counter()
    if _frame_min_interval > 0.0:
        print(f"[producer] frame-rate cap: {_cap_fps:.1f} fps "
              f"(throttles render+capture to free SM for the consumer)")

    try:
        while simulation_app.is_running():
            # Frame-rate cap: pace the loop so render+capture run at most at the
            # configured fps, leaving SM headroom for the consumer's optimiser.
            if _frame_min_interval > 0.0:
                _dt = time.perf_counter() - _loop_t
                if _dt < _frame_min_interval:
                    time.sleep(_frame_min_interval - _dt)
                _loop_t = time.perf_counter()
            # Guard clause: non-XR validation just renders the scene and
            # skips all input + capture logic below. Putting this at the
            # top dedents the rest of the loop body by one level.
            if not xr_enabled:
                simulation_app.update()
                continue

            st = controller.read(joint_mode=joint_mode)
            if st.menu_edge:
                joint_mode = not joint_mode
                print(f"[input] mode → "
                      f"{'JOINT' if joint_mode else 'TROLLEY'}")

            if joint_mode:
                # Per-tick joint deltas (only triggers/grips/buttons drive
                # joints; both-grips-held still toggles mode via menu_edge).
                joint_deg[0] += st.lx * args.joint_speed
                joint_deg[1] += st.ly * args.joint_speed
                joint_deg[2] += st.rx * args.joint_speed
                joint_deg[3] += st.ry * args.joint_speed
                joint_deg[4] += (st.r_trig - st.l_trig) * args.joint_speed
                # A6 ← right grip - left grip (analog, independent of buttons).
                # When BOTH grips are held to trigger mode-toggle, A6 net
                # delta is zero (they cancel), so the joint doesn't move.
                joint_deg[5] += (st.r_grip - st.l_grip) * args.joint_speed
                # A7 ← BUTTON_A (+) / BUTTON_B (-), digital
                joint_deg[6] += ((1.0 if st.a else 0.0)
                                 - (1.0 if st.b else 0.0)) * args.joint_speed
                np.clip(joint_deg, -JOINT_LIMIT_DEG, JOINT_LIMIT_DEG, out=joint_deg)

                if any((st.lx, st.ly, st.rx, st.ry,
                        st.l_trig, st.r_trig,
                        st.l_grip, st.r_grip, st.a, st.b)):
                    apply_joint_targets(robot_path, joint_deg)
            else:
                # Trolley mode: move base in world XY + yaw (Z is fixed)
                moved = False
                if st.lx or st.ly:
                    base_xyz[0] += st.lx * args.trolley_speed * 0.01
                    base_xyz[1] += st.ly * args.trolley_speed * 0.01
                    moved = True
                if st.rx:
                    base_yaw += st.rx * 0.02
                    moved = True
                if moved:
                    set_base_pose(robot_path, base_xyz, base_yaw)

            # ── Stereo capture (runs in BOTH modes) ──
            # Capture/publish must continue in JOINT mode as well, otherwise the
            # frame stream stops the instant the arm is articulated and the
            # consumer times out on read() and exits. base_xyz/base_yaw hold the
            # current base pose (only TROLLEY moves it) and joint_deg holds the
            # current joints (only JOINT moves them) — both are valid every tick.
            active_backend = layer_capture
            ready = (active_backend is not None
                     and capture_out_dir is not None)

            # Publish toggle: press-A starts continuous publishing, press-B
            # stops — but ONLY in TROLLEY mode. In JOINT mode A/B drive joint A7,
            # so they must not be consumed as publish toggles; continuous_capture
            # persists across a mode switch, so start publishing in TROLLEY (or
            # via --ros-publish-always) then switch to JOINT to move the arm.
            if (ready and not joint_mode and st.a_edge and not continuous_capture
                    and not args.ros_publish_always):
                continuous_capture = True
                cont_start_idx = capture_idx
                print(f"[capture] PUBLISH START (from #{capture_idx:04d}) "
                      f"— press B to stop")
            if ready and not joint_mode and st.b_edge and continuous_capture:
                continuous_capture = False
                print(f"[capture] PUBLISH STOP "
                      f"({capture_idx - cont_start_idx} frames, "
                      f"{capture_idx} total)")

            should_capture = ready and (
                continuous_capture
                or (args.ros_publish_always and ros_bridge is not None))

            if should_capture:
                # Base pose (script state) is the only per-frame snapshot the
                # bridge needs alongside the layer's eye poses + joints. The
                # eye poses themselves arrive from the layer inside capture().
                H_base_world = _xy_yaw_to_H(base_xyz, base_yaw)
                snap = {
                    "idx":          capture_idx,
                    "joint_rad":    np.radians(joint_deg).copy(),
                    "H_base_world": H_base_world,
                }
                if active_backend.capture(capture_out_dir, snap):
                    capture_idx += 1
                # else: capture failed — don't bump capture_idx.

            simulation_app.update()

            # Intrinsics capture: start once xrCamera prim appears in the stage
            if intrinsics is None and is_xr_active():
                intrinsics = IntrinsicsOneShot(
                    warmup_sec=args.intrinsics_warmup_sec,
                    save_path=args.intrinsics_snapshot_path,
                )
                print(f"[intrinsics] capture started "
                      f"(warm-up {args.intrinsics_warmup_sec:.1f}s)")

            # Intrinsics arrival is one-time: report immediately on snapshot
            if (intrinsics is not None
                    and intrinsics.is_done() and not intrinsics_reported):
                snap = intrinsics.snapshot()
                print("─" * 70)
                if snap:
                    print(f"[intrinsics] captured (frozen):")
                    for eye_id in sorted(snap):
                        print("  " + _fmt_intr(eye_id, snap[eye_id]))
                    print(f"  saved → {args.intrinsics_snapshot_path}")

                    if capture_mode == "layer":
                        try:
                            # Bridge up before LayerFrameCapture so
                            # the sidecar is ready for the first
                            # publish() call. (Architecture: see
                            # RosBridge docstring + ARCHITECTURE.md)
                            ros_bridge = None
                            if args.ros_publish:
                                try:
                                    ros_bridge = RosBridge(
                                        script_path=args.ros_bridge_script,
                                        setup_path=args.ros_bridge_setup,
                                        socket_path=args.ros_bridge_socket,
                                        python_bin=args.ros_bridge_python,
                                        namespace=args.ros_namespace,
                                        frame_id_world=args.ros_frame_id,
                                        joint_names=list(JOINT_NAMES),
                                        intrinsics_snapshot_path=args.intrinsics_snapshot_path,
                                        image_codec=args.image_codec,
                                    )
                                    ros_bridge.start()
                                    mode_tag = (
                                        "always-mode: live publish only, "
                                        "press-A disabled"
                                        if args.ros_publish_always
                                        else "press-A → publish only "
                                             "(no recording)"
                                        if args.ros_publish_no_record
                                        else "press-A → rosbag")
                                    print(f"  ROS:        publishing under "
                                          f"{args.ros_namespace}/ via sidecar "
                                          f"({mode_tag}, world: "
                                          f"{args.ros_frame_id})")
                                except RuntimeError as e:
                                    print(f"[ros] setup failed: {e}")
                                    print(f"[ros] continuing without --ros-publish")
                                    ros_bridge = None
                            layer_capture = LayerFrameCapture(
                                socket_path=args.frames_socket_path,
                                profile=args.profile_capture,
                                ros_bridge=ros_bridge,
                            )
                            layer_capture.start()
                            if not layer_capture._ready:
                                print(f"[capture] layer setup FAILED: "
                                      f"could not connect to "
                                      f"{args.frames_socket_path}. Make sure "
                                      f"XR_APILAYER_FRAME_GRAB is enabled and "
                                      f"the layer's .so built. Layer mode will "
                                      f"not work for this run.")
                                layer_capture = None
                            else:
                                capture_out_dir = setup_output_dir(args.output_dir)
                                save_camera_info_yaml_stereo(capture_out_dir, snap)
                                print(f"[capture] layer mode ready. Hold left "
                                      f"trigger >{args.capture_trigger_hold:.1f}s "
                                      f"in TROLLEY mode to capture.")
                        except Exception as e:
                            print(f"[capture] layer setup FAILED: {e}")
                            layer_capture = None
                else:
                    print(f"[intrinsics] capture FAILED — {intrinsics.status()}")
                intrinsics_reported = True

    except KeyboardInterrupt:
        print("\n[exit] interrupted")

    # Stop the capture worker (closes the ROS bridge sidecar too).
    if layer_capture is not None:
        layer_capture.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()