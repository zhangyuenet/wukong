# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Capture depth frames from ROS 2 and turn them into world points.

Topics (bridged by scripts/sim.sh for PX4's x500_depth):
  /depth/image        sensor_msgs/Image, 32FC1, metres along the optical axis
  /depth/camera_info  sensor_msgs/CameraInfo
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from .geometry import depth_rays, depth_to_points, Intrinsics

Pose = Tuple[float, float, float, float]   # north, east, height, heading


def image_to_depth(msg: Image) -> np.ndarray:
    if msg.encoding == '32FC1':
        a = np.frombuffer(bytes(msg.data), dtype=np.float32)
        return a.reshape(msg.height, msg.step // 4)[:, :msg.width]
    if msg.encoding == '16UC1':   # millimetres, as real OAK-D drivers publish
        a = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        a = a.reshape(msg.height, msg.step // 2)[:, :msg.width].astype(np.float32) / 1000.0
        a[a == 0] = np.inf
        return a
    raise ValueError(f'unsupported depth encoding {msg.encoding}')


class DepthCamera:
    """Capture the next depth frame after a request and convert it with the given pose."""

    def __init__(self, node: Node, image_topic: str = '/depth/image',
                 info_topic: str = '/depth/camera_info') -> None:
        self._k: Optional[Intrinsics] = None
        self._pending: Optional[tuple] = None
        self.clouds: List[np.ndarray] = []      # every successful capture, in order
        self.labels: List[str] = []
        self.poses: List[Tuple[float, ...]] = []   # vehicle pose used for each capture
        self.rays: List[tuple] = []              # (camera, ray ends, hit) per capture
        self.stamps: List[float] = []            # image time stamp, s
        node.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)
        node.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)

    @property
    def ready(self) -> bool:
        return self._k is not None

    def capture(self, label: str, pose: Pose, done: Callable[[bool], None]) -> None:
        """Convert the next frame using pose (the vehicle is holding still), then done(ok)."""
        self._pending = (label, pose, done)

    def _on_info(self, msg: CameraInfo) -> None:
        k = msg.k
        if k[0] > 0:
            self._k = Intrinsics(k[0], k[4], k[2], k[5])

    def _on_image(self, msg: Image) -> None:
        if self._pending is None or self._k is None:
            return
        label, pose, done = self._pending
        self._pending = None
        try:
            depth = image_to_depth(msg)
            pts = depth_to_points(depth, self._k, pose)
            rays = depth_rays(depth, self._k, pose)
        except ValueError:
            done(False)
            return
        self.rays.append(rays)
        self.clouds.append(pts)
        self.labels.append(label)
        self.poses.append(tuple(float(v) for v in pose))
        self.stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        done(len(pts) > 0)
