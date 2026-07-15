#!/usr/bin/env python3
"""fs_input_relay.py — feed the FoundationStereo graph from the pipeline's
already-rectified stream.

The FS include applies its own internal remappings, which take precedence
over launch-scoped SetRemap rules — so instead of fighting remap precedence,
this relay publishes onto the topics the FS nodes actually subscribe
(verified via `ros2 node info`):

    <src-left-image>   -> /fs/left/image_raw
    <src-right-image>  -> /fs/right/image_raw
    <src-left-ci>      -> /fs/left/camera_info
    <src-right-ci>     -> /fs/right/camera_info

FS's internal rectify is an identity under the rectified stream's
zero-distortion camera_info; resize/pad are no-ops at 960x576. In warm mode
the sources are the /seed/* gated topics, so this relay moves nothing while
the gate is closed — zero idle cost.
"""
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


class FsInputRelay(Node):
    def __init__(self, a):
        super().__init__('fs_input_relay')
        pairs = [
            (a.src_left_image,  '/fs/left/image_raw',    Image),
            (a.src_right_image, '/fs/right/image_raw',   Image),
            (a.src_left_ci,     '/fs/left/camera_info',  CameraInfo),
            (a.src_right_ci,    '/fs/right/camera_info', CameraInfo),
        ]
        self._n = 0
        for src, dst, typ in pairs:
            pub = self.create_publisher(typ, dst, 5)
            self.create_subscription(
                typ, src,
                (lambda p: (lambda m: self._fwd(p, m)))(pub), 5)
        self.get_logger().info(
            'FS input relay up: ' +
            '; '.join(f'{s} -> {d}' for s, d, _ in pairs))

    def _fwd(self, pub, msg):
        pub.publish(msg)
        self._n += 1
        if self._n == 1:
            self.get_logger().info('first message relayed into /fs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-left-image',  default='/left/image_rect')
    ap.add_argument('--src-right-image', default='/right/image_rect')
    ap.add_argument('--src-left-ci',     default='/left/camera_info_rect')
    ap.add_argument('--src-right-ci',    default='/right/camera_info_rect')
    a, _ = ap.parse_known_args()
    rclpy.init()
    rclpy.spin(FsInputRelay(a))


if __name__ == '__main__':
    main()
