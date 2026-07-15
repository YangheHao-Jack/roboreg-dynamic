#!/usr/bin/env python3
"""seed_gate.py — input valve for a resident (warm) ESS+FP seed stack.

Keeps the seed engines loaded but idle: ESS/depth consume gated copies of the
rectified topics; this node forwards them only while armed. Arm -> the seed
stack computes; disarm -> zero GPU (both engines are event-driven).

    /seed/arm   std_msgs/Bool (latched)   true = forward, false = idle
    in:   /left/image_rect  /right/image_rect  /left/camera_info_rect  /right/camera_info_rect
    out:  same names under /seed/...

Manual re-init workflow (until the consumer automates it):
    ros2 topic pub --once /seed/arm std_msgs/msg/Bool "{data: true}"
    ... recorder latches a fresh /pose_init ...
    ros2 topic pub --once /seed/arm std_msgs/msg/Bool "{data: false}"
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool


PAIRS = [
    ('/left/image_rect',        '/seed/left/image_rect',        Image),
    ('/right/image_rect',       '/seed/right/image_rect',       Image),
    ('/left/camera_info_rect',  '/seed/left/camera_info_rect',  CameraInfo),
    ('/right/camera_info_rect', '/seed/right/camera_info_rect', CameraInfo),
]


class SeedGate(Node):
    def __init__(self, start_armed=False):
        super().__init__('seed_gate')
        self.armed = bool(start_armed)
        self._fwd = 0
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/seed/arm', self._on_arm, latched)
        # Publishers persist; SUBSCRIPTIONS exist only while armed. With no
        # subscription, DDS/SHM delivers nothing at all — the disarmed gate
        # costs zero CPU/bandwidth (vs. receive-and-drop, which would
        # deserialise ~200 MB/s of raw images forever).
        self._pubs = {src: self.create_publisher(typ, dst, 5)
                      for src, dst, typ in PAIRS}
        self._subs = []
        if self.armed:
            self._make_subs()
        self.get_logger().info(
            f"seed gate up ({'ARMED — forwarding (first-init mode); the '
            'consumer disarms after handoff' if self.armed else
            'DISARMED, zero-cost'}): arm via /seed/arm to re-initialise.")

    def _make_subs(self):
        for src, dst, typ in PAIRS:
            pub = self._pubs[src]
            self._subs.append(self.create_subscription(
                typ, src,
                (lambda p: (lambda m: self._relay(p, m)))(pub), 5))

    def _drop_subs(self):
        for s in self._subs:
            self.destroy_subscription(s)
        self._subs = []

    def _on_arm(self, msg):
        if bool(msg.data) != self.armed:
            self.armed = bool(msg.data)
            self._fwd = 0
            if self.armed:
                self._make_subs()
            else:
                self._drop_subs()
            self.get_logger().info(
                f"seed gate {'ARMED — seed stack computing' if self.armed else 'disarmed — seed stack idle (zero-cost)'}")

    def _relay(self, pub, msg):
        if self.armed:
            pub.publish(msg)
            self._fwd += 1
            if self._fwd == 1:
                self.get_logger().info('first gated message forwarded')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-armed', action='store_true',
                    help='Begin forwarding immediately (warm-launch first '
                         'init: behaves exactly like the ungated path until '
                         'the consumer disarms after its handoff).')
    args, _ = ap.parse_known_args()
    rclpy.init()
    n = SeedGate(start_armed=args.start_armed)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()