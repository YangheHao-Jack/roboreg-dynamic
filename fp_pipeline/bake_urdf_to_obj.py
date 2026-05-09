#!/usr/bin/env python3
"""
Bake URDF + per-link visual meshes into a single .obj at a fixed FK pose.

Same FK / mesh-loading pattern as `track_full_robot_single_frame_stereo.py`:
  - pytorch_kinematics for forward kinematics (build_chain_from_urdf)
  - urdfpy for enumerating link visuals + applying visual.origin and mesh.scale
  - Joint angles read from a CSV/XLSX with columns lbr_A1..lbr_A7

Output (in `--out_dir`):
  lbr_med7_baked.obj           Combined mesh, recentered at bbox center
                                (Isaac ROS FoundationPose requires this).
  lbr_med7_baked_offset.npy    The offset that, added to a centered-mesh
                                point, recovers the original lbr_link_0
                                coordinates: p_link0 = p_centered + offset.
                                FP outputs T_camera_to_centered, so:
                                  T_camera_to_link0 = T_camera_to_centered
                                                     @ translate(+offset)

Usage:
  source "/media/jack/新加卷/venvs/FS_env/bin/activate"
  pip install pytorch_kinematics urdfpy trimesh pandas openpyxl

  python bake_urdf_to_obj.py \\
      --urdf /home/jack/roboreg/test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf \\
      --joint_csv /path/to/joints.xlsx \\
      --out_dir /home/jack/FoundationPose_assets/

The CSV/XLSX must have columns lbr_A1..lbr_A7. The first data row is used
by default (matching your screenshot). To pick another row pass --joint_row.
"""

import argparse
from pathlib import Path
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True,
                   help="Path to URDF (e.g. lbr_med7_r800.urdf)")
    p.add_argument("--joint_csv", required=True,
                   help="CSV or XLSX with columns lbr_A1..lbr_A7")
    p.add_argument("--joint_row", type=int, default=0,
                   help="Which data row to read (0-indexed; default: 0)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--obj_name", default="lbr_med7_baked")
    return p.parse_args()


def load_joint_angles(joint_csv: str, joint_row: int) -> dict:
    """Return {'lbr_A1': float, ..., 'lbr_A7': float} from CSV/XLSX."""
    import pandas as pd
    path = Path(joint_csv)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    expected = [f"lbr_A{i}" for i in range(1, 8)]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        sys.exit(f"[bake] CSV missing columns: {missing}. "
                 f"Got columns: {list(df.columns)}")
    if joint_row >= len(df):
        sys.exit(f"[bake] joint_row={joint_row} but CSV only has {len(df)} rows")

    row = df.iloc[joint_row]
    return {c: float(row[c]) for c in expected}


def load_robot_mesh_with_fk(urdf_path, joint_dict):
    """Apply FK and concatenate all visual link meshes into base-frame mesh.
    Same pattern as track_full_robot_single_frame_stereo.py."""
    import torch
    import pytorch_kinematics as pk
    import trimesh
    from urdfpy import URDF

    urdf_str = Path(urdf_path).read_text()
    chain = pk.build_chain_from_urdf(urdf_str.encode("utf-8"))
    robot = URDF.load(str(urdf_path))

    # Sanity: make sure every actuated joint has an angle in our dict
    actuated = chain.get_joint_parameter_names()
    print(f"[bake] Actuated joints in URDF: {actuated}")
    for j in actuated:
        if j not in joint_dict:
            sys.exit(f"[bake] No angle provided for actuated joint '{j}'. "
                     f"Got: {list(joint_dict.keys())}")
    angles_dict = {j: torch.tensor(joint_dict[j], dtype=torch.float32)
                   for j in actuated}
    print(f"[bake] FK joint config:")
    for j in actuated:
        print(f"        {j} = {joint_dict[j]:+.6f} rad")

    fk = chain.forward_kinematics(angles_dict)

    pieces = []
    for link in robot.links:
        if not link.visuals:
            continue
        if link.name not in fk:
            continue
        H_link_to_base = fk[link.name].get_matrix()[0].cpu().numpy()
        for visual in link.visuals:
            geom = visual.geometry
            if geom.mesh is None or not geom.mesh.meshes:
                continue
            for m in geom.mesh.meshes:
                m_copy = m.copy()
                # Visual origin offset (URDF link -> mesh offset)
                if visual.origin is not None:
                    m_copy.apply_transform(visual.origin)
                # Mesh scale (some .dae embed scale)
                if geom.mesh.scale is not None:
                    s = np.eye(4); s[:3, :3] = np.diag(geom.mesh.scale)
                    m_copy.apply_transform(s)
                # FK to base
                m_copy.apply_transform(H_link_to_base)
                pieces.append(m_copy)
                print(f"[bake]   + {link.name}: {len(m_copy.vertices)} verts")
    if not pieces:
        sys.exit("[bake] No visual meshes found in URDF.")
    return trimesh.util.concatenate(pieces)


def main():
    a = parse_args()

    joint_dict = load_joint_angles(a.joint_csv, a.joint_row)
    print(f"[bake] Loaded joint angles from {a.joint_csv} (row {a.joint_row})")

    print(f"[bake] Loading URDF + computing FK + concatenating visuals...")
    mesh = load_robot_mesh_with_fk(a.urdf, joint_dict)
    print(f"[bake] Combined: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"[bake] Bounds in lbr_link_0:")
    print(f"         min:     {mesh.bounds[0]}")
    print(f"         max:     {mesh.bounds[1]}")
    print(f"         extents: {mesh.extents}")

    # Recenter for Isaac ROS FP. Save offset for downstream pose recovery.
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    print(f"[bake] Recentering: subtracting bbox center {bbox_center}")
    mesh.apply_translation(-bbox_center)

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    obj_path    = out_dir / f"{a.obj_name}.obj"
    offset_path = out_dir / f"{a.obj_name}_offset.npy"
    mesh.export(obj_path)
    np.save(offset_path, bbox_center.astype(np.float64))

    print(f"[bake] DONE")
    print(f"[bake]   .obj          -> {obj_path}")
    print(f"[bake]   offset .npy   -> {offset_path}")
    print(f"[bake]")
    print(f"[bake] To recover the pose in lbr_link_0 frame from FP output:")
    print(f"[bake]   T_cam_link0 = T_cam_centered @ T_centered_link0")
    print(f"[bake]   where T_centered_link0 = np.eye(4) with [:3,3] = +offset")
    print(f"[bake]   (offset = {bbox_center})")


if __name__ == "__main__":
    main()
