#!/usr/bin/env python3
"""
static_fields_publisher.py — ONE process publishing all the static fields the
seed stack needs, from offline files:

  camera_info L/R + baseline  ->  {ns}/image_left|right/camera_info, {ns}/baseline
  joint states (constant)     ->  --joints-topic (default /lbr/joint_states)

    python3 static_fields_publisher.py \
        --left  ~/quest_calib/camera_info_left.yaml \
        --right ~/quest_calib/camera_info_right.yaml \
        --baseline ~/xr_captures/20260524_171931/HT_right_to_left.npy \
        --joints ~/rosbag2_2026_07_07-14_58_27/extract/joint_states.csv

Formats: camera_info YAML in either style (receiver --save-calib message-dump
or producer calibration-file); baseline as 4x4 .npy or baseline.yaml; joints
as bag_joint_extract csv (t_header,t_bag,<joint>...).
"""
import argparse
import csv as _csv

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, JointState
from geometry_msgs.msg import PoseStamped


def caminfo_from_yaml(path: str) -> CameraInfo:
    import yaml
    with open(path) as f:
        y = yaml.safe_load(f)
    ci = CameraInfo()
    if "k" in y:                                   # message-dump style
        ci.width = int(y["width"]); ci.height = int(y["height"])
        ci.k = [float(v) for v in y["k"]]
        ci.p = [float(v) for v in y["p"]]
        ci.r = [float(v) for v in y["r"]]
        ci.d = [float(v) for v in y.get("d", [])]
        ci.distortion_model = y.get("distortion_model", "plumb_bob")
        ci.header.frame_id = y.get("frame_id", "")
    else:                                          # calibration-file style
        ci.width = int(y["image_width"]); ci.height = int(y["image_height"])
        ci.k = [float(v) for v in y["camera_matrix"]["data"]]
        ci.p = [float(v) for v in y["projection_matrix"]["data"]]
        ci.r = [float(v) for v in y["rectification_matrix"]["data"]]
        ci.d = [float(v) for v in y["distortion_coefficients"]["data"]]
        ci.distortion_model = y.get("distortion_model", "plumb_bob")
        ci.header.frame_id = y.get("camera_name", "")
    print(f"[static] {path}: {ci.width}x{ci.height} fx={ci.k[0]:.2f}")
    return ci


def pose_from_baseline_file(path: str) -> PoseStamped:
    ps = PoseStamped()
    ps.pose.orientation.w = 1.0
    if path.endswith(".npy"):
        T = np.load(path)
        assert T.shape == (4, 4), f"{path}: expected 4x4, got {T.shape}"
        ps.pose.position.x = float(T[0, 3])
        ps.pose.position.y = float(T[1, 3])
        ps.pose.position.z = float(T[2, 3])
        R = T[:3, :3]
        qw = float(np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0)
        if qw > 1e-9:
            ps.pose.orientation.x = float((R[2, 1] - R[1, 2]) / (4 * qw))
            ps.pose.orientation.y = float((R[0, 2] - R[2, 0]) / (4 * qw))
            ps.pose.orientation.z = float((R[1, 0] - R[0, 1]) / (4 * qw))
            ps.pose.orientation.w = qw
    else:
        import yaml
        with open(path) as f:
            y = yaml.safe_load(f)
        t = y["translation"]; q = y.get("rotation_xyzw", [0, 0, 0, 1])
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, t)
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = map(float, q)
    p = ps.pose.position
    b = (p.x**2 + p.y**2 + p.z**2) ** 0.5
    print(f"[static] {path}: baseline={b*1000:.1f} mm "
          f"t=({p.x:+.5f}, {p.y:+.5f}, {p.z:+.5f})")
    if b < 1e-3:
        raise SystemExit("[static] baseline < 1 mm — file suspect, refusing.")
    return ps


def jointstate_from_csv(path: str, row: int = 0, mean: bool = False) -> JointState:
    with open(path, newline='') as f:
        r = _csv.reader(f)
        names = next(r)[2:]                        # skip t_header, t_bag
        rows = [[float(v) for v in line[2:]] for line in r if line]
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    vals = ([sum(c) / len(rows) for c in zip(*rows)] if mean else rows[row])
    js = JointState()
    js.name = list(names)
    js.position = [float(v) for v in vals]
    print(f"[static] {path} ({'mean' if mean else f'row {row}'}): "
          + ", ".join(f"{n}={v:+.4f}" for n, v in zip(names, vals)))
    return js


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--left', default=None, help='camera_info_left.yaml')
    ap.add_argument('--right', default=None, help='camera_info_right.yaml')
    ap.add_argument('--baseline', default=None,
                    help='HT_right_to_left.npy or baseline.yaml')
    ap.add_argument('--joints', default=None, help='joint_states.csv (extract)')
    ap.add_argument('--joints-row', type=int, default=0)
    ap.add_argument('--joints-mean', action='store_true')
    ap.add_argument('--namespace', default='/xr')
    ap.add_argument('--joints-topic', default='/lbr/joint_states')
    ap.add_argument('--calib-rate', type=float, default=1.0)
    ap.add_argument('--joints-rate', type=float, default=100.0)
    a = ap.parse_args()
    ns = a.namespace.rstrip('/')

    # Publish whatever subset of fields was given: calib trio (all three
    # together or none — partial calibration is worse than none) and/or
    # joints. Joints-only is the LIVE-pipeline mode, where the receiver
    # already publishes calibration — running the calib trio alongside it
    # would double-publish conflicting values on the same topics.
    calib_given = [a.left, a.right, a.baseline]
    if any(calib_given) and not all(calib_given):
        raise SystemExit("give --left/--right/--baseline together or not at all")
    if not any(calib_given) and not a.joints:
        raise SystemExit("nothing to publish (no calib files, no --joints)")
    ci_l = caminfo_from_yaml(a.left) if a.left else None
    ci_r = caminfo_from_yaml(a.right) if a.right else None
    ps = pose_from_baseline_file(a.baseline) if a.baseline else None
    js = (jointstate_from_csv(a.joints, a.joints_row, a.joints_mean)
          if a.joints else None)

    rclpy.init()
    node = Node('static_fields_publisher')
    parts = []
    if ci_l is not None:
        pub_l = node.create_publisher(CameraInfo, f'{ns}/image_left/camera_info', 1)
        pub_r = node.create_publisher(CameraInfo, f'{ns}/image_right/camera_info', 1)
        pub_b = node.create_publisher(PoseStamped, f'{ns}/baseline', 1)

        def tick_calib():
            now = node.get_clock().now().to_msg()
            ci_l.header.stamp = now; ci_r.header.stamp = now
            ps.header.stamp = now
            pub_l.publish(ci_l); pub_r.publish(ci_r); pub_b.publish(ps)

        node.create_timer(1.0 / max(a.calib_rate, 0.1), tick_calib)
        parts.append(f"calib on {ns}/* at {a.calib_rate} Hz")
    if js is not None:
        pub_j = node.create_publisher(JointState, a.joints_topic, 10)

        def tick_joints():
            js.header.stamp = node.get_clock().now().to_msg()
            pub_j.publish(js)

        node.create_timer(1.0 / max(a.joints_rate, 0.1), tick_joints)
        parts.append(f"joints on {a.joints_topic} at {a.joints_rate} Hz")
    node.get_logger().info("static fields up: " + "; ".join(parts))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()