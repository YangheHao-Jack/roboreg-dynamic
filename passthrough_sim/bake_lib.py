#!/usr/bin/env python3
"""
bake_lib.py

Reusable robot URDF bake. Same FK / mesh-loading pattern as
bake_urdf_to_obj.py, but as an importable function so the bake can run
inside a ROS node from a live joint_state message instead of a CSV.

Public entry point:
    bake_from_joint_dict(urdf_path, joint_dict, out_dir, obj_name)
        -> (obj_path: Path, offset_path: Path, info: dict)

Output files
    <out_dir>/<obj_name>.obj            Combined mesh, recentered at bbox.
    <out_dir>/<obj_name>_offset.npy     Translation s.t.
                                        p_link0 = p_centered + offset
"""

from pathlib import Path
import sys

import numpy as np


def load_robot_mesh_with_fk(urdf_path, joint_dict, log=print):
    """Apply FK and concatenate all visual link meshes into base-frame mesh."""
    import torch
    import pytorch_kinematics as pk
    import trimesh
    import yourdfpy

    urdf_str = Path(urdf_path).read_text()
    chain = pk.build_chain_from_urdf(urdf_str.encode("utf-8"))
    robot = yourdfpy.URDF.load(str(urdf_path))

    actuated = chain.get_joint_parameter_names()
    log(f"[bake] Actuated joints in URDF: {actuated}")
    for j in actuated:
        if j not in joint_dict:
            raise ValueError(
                f"No angle provided for actuated joint '{j}'. "
                f"Got: {list(joint_dict.keys())}")

    angles_dict = {j: torch.tensor(joint_dict[j], dtype=torch.float32)
                   for j in actuated}
    log(f"[bake] FK joint config:")
    for j in actuated:
        log(f"        {j} = {joint_dict[j]:+.6f} rad")

    fk = chain.forward_kinematics(angles_dict)

    # yourdfpy's robot.link_map: name -> Link, each with .visuals
    pieces = []
    urdf_dir = Path(urdf_path).resolve().parent
    for link_name, link in robot.link_map.items():
        if not link.visuals:
            continue
        if link_name not in fk:
            continue
        H_link_to_base = fk[link_name].get_matrix()[0].cpu().numpy()
        for visual in link.visuals:
            geom = visual.geometry
            mesh_field = getattr(geom, "mesh", None)
            if mesh_field is None:
                continue
            # yourdfpy gives filename (relative to URDF dir) + optional scale
            mesh_path = mesh_field.filename
            if not Path(mesh_path).is_absolute():
                mesh_path = str((urdf_dir / mesh_path).resolve())
            try:
                loaded = trimesh.load(mesh_path, force="mesh")
            except Exception as e:
                log(f"[bake]   WARN: could not load {mesh_path}: {e}")
                continue
            if loaded is None or len(loaded.vertices) == 0:
                continue
            m_copy = loaded.copy()
            # Visual origin (URDF link -> mesh frame)
            if visual.origin is not None:
                origin = np.asarray(visual.origin, dtype=np.float64)
                if origin.shape == (4, 4):
                    m_copy.apply_transform(origin)
            # Mesh scale
            if mesh_field.scale is not None:
                s = np.eye(4)
                s[:3, :3] = np.diag(np.asarray(mesh_field.scale,
                                               dtype=np.float64))
                m_copy.apply_transform(s)
            # FK to base
            m_copy.apply_transform(H_link_to_base)
            pieces.append(m_copy)
            log(f"[bake]   + {link_name}: {len(m_copy.vertices)} verts")

    if not pieces:
        raise RuntimeError("No visual meshes found in URDF.")
    return trimesh.util.concatenate(pieces)


def bake_from_joint_dict(urdf_path: str,
                         joint_dict: dict,
                         out_dir: str,
                         obj_name: str = "lbr_med7_baked",
                         log=print):
    """Full bake. Returns (obj_path, offset_path, info_dict)."""
    log(f"[bake] Loading URDF + computing FK + concatenating visuals...")
    mesh = load_robot_mesh_with_fk(urdf_path, joint_dict, log=log)
    log(f"[bake] Combined: {len(mesh.vertices)} verts, "
        f"{len(mesh.faces)} faces")
    log(f"[bake] Bounds in lbr_link_0:")
    log(f"         min:     {mesh.bounds[0]}")
    log(f"         max:     {mesh.bounds[1]}")
    log(f"         extents: {mesh.extents}")

    # Recenter for Isaac ROS FP. Save offset for downstream pose recovery.
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    log(f"[bake] Recentering: subtracting bbox center {bbox_center}")
    mesh.apply_translation(-bbox_center)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    obj_path    = out_dir_p / f"{obj_name}.obj"
    offset_path = out_dir_p / f"{obj_name}_offset.npy"
    mesh.export(obj_path)
    np.save(offset_path, bbox_center.astype(np.float64))

    log(f"[bake] DONE: {obj_path}, {offset_path}")
    return obj_path, offset_path, {
        "n_verts": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "bbox_center": bbox_center.tolist(),
        "extents": mesh.extents.tolist(),
    }


# ── CLI shim (matches the original bake_urdf_to_obj.py interface) ────
def _cli():
    """Command-line entry: read joint angles from CSV/XLSX, call
    bake_from_joint_dict. Behaviour matches the original script."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True)
    p.add_argument("--joint_csv", required=True)
    p.add_argument("--joint_row", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--obj_name", default="lbr_med7_baked")
    a = p.parse_args()

    import pandas as pd
    path = Path(a.joint_csv)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    expected = [f"lbr_A{i}" for i in range(1, 8)]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        sys.exit(f"[bake] CSV missing columns: {missing}")
    if a.joint_row >= len(df):
        sys.exit(f"[bake] joint_row={a.joint_row} but CSV has {len(df)} rows")

    row = df.iloc[a.joint_row]
    joint_dict = {c: float(row[c]) for c in expected}
    bake_from_joint_dict(a.urdf, joint_dict, a.out_dir, a.obj_name)


if __name__ == "__main__":
    _cli()