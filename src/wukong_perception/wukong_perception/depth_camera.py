# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Capture depth frames from ROS 2 and turn them into world points.

Topics (bridged by scripts/sim.sh for PX4's x500_depth):
  /depth/image        sensor_msgs/Image, 32FC1, metres along the optical axis
  /depth/camera_info  sensor_msgs/CameraInfo

Two kinds of frame. A scan is the next frame after a request (the vehicle
holds still). A frame in flight is taken every FLIGHT_STEP metres or
FLIGHT_TURN_DEG degrees and goes only to the voxel map. Given a pose_at
source, both are converted with the full pose at the moment they were
imaged; a scan without one uses the pose at the request, and so do frames in
flight not at all. 两种帧：扫描是请求之后的下一帧（飞机悬停）；飞行中每移动
FLIGHT_STEP 米或转 FLIGHT_TURN_DEG 度取一帧，只用于体素地图。有 pose_at 时两者都用
成像时刻的完整位姿转换；没有时扫描用请求时的位姿，也不取飞行中的帧。
"""

from collections import Counter, deque
import time
from typing import Callable, Deque, List, Optional, Tuple, Union

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from .geometry import (depth_rays, depth_to_points, FLIGHT_STRIDE, Intrinsics,
                       new_viewpoint)

Pose = Tuple[float, ...]   # north, east, height, heading[, roll, pitch]
# A frame's stamp (us) -> the pose then, or why it cannot be told well.
PoseAt = Callable[[float], Union[Pose, str]]

# s a frame in flight waits before it is converted, so that pose samples from
# after its stamp have arrived. 飞行中的帧等这么久再转换，让成像时刻之后的位姿样本先到。
FLIGHT_DELAY = 0.15
KEEP_POINTS = 300          # hit points kept per frame in flight, for checking
# s a scan waits for a frame with a pose at its moment, then takes the pose at
# the request (the vehicle holds still). 扫描等这么久仍没有带成像时刻位姿的帧时，
# 改用请求时的位姿（飞机在悬停）。
SCAN_POSE_WAIT = 1.0


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


def _stamp_us(msg: Image) -> float:
    return msg.header.stamp.sec * 1e6 + msg.header.stamp.nanosec * 1e-3


class DepthCamera:
    """Capture depth frames: scans on request, and, given pose_at, frames in flight."""

    def __init__(self, node: Node, image_topic: str = '/depth/image',
                 info_topic: str = '/depth/camera_info',
                 pose_at: Optional[PoseAt] = None) -> None:
        self._k: Optional[Intrinsics] = None
        self._pending: Optional[tuple] = None
        self.clouds: List[np.ndarray] = []      # every successful capture, in order
        self.labels: List[str] = []
        self.poses: List[Tuple[float, ...]] = []   # vehicle pose used for each capture
        self.rays: List[tuple] = []              # (camera, ray ends, hit) per capture
        self.stamps: List[float] = []            # image time stamp, s
        # Every frame for the voxel map, in order: the scans and the frames in
        # flight. 体素地图用的全部深度帧，按顺序：扫描和飞行中的帧。
        self.frames: List[tuple] = []
        self.flight_poses: List[Tuple[float, ...]] = []   # (stamp s, *pose) per frame in flight
        # A few of each frame's hit points, to check them against the truth.
        # 每帧的少量击中点，用于和真值核对。
        self.flight_points: List[np.ndarray] = []
        self.dropped: Counter = Counter()        # frames in flight not used, by reason
        self.received = 0                        # frames in flight looked at
        self._pose_at = pose_at
        self._queue: Deque[Tuple[Image, float]] = deque(maxlen=8)
        self._last_view: Optional[Pose] = None
        node.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)
        node.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        if pose_at is not None:
            node.create_timer(0.05, self._drain)

    @property
    def ready(self) -> bool:
        return self._k is not None

    def capture(self, label: str, pose: Pose, done: Callable[[bool], None]) -> None:
        """Convert the next frame (the vehicle is holding still at pose), then done(ok)."""
        self._pending = (label, pose, done, time.monotonic())

    def describe_flight(self) -> str:
        """How many frames in flight were kept and why others were not, for the log."""
        why = ', '.join(f'{n} {r}' for r, n in sorted(self.dropped.items()))
        return (f'{len(self.flight_poses)} kept of {self.received}'
                + (f'; without a pose: {why}' if why else ''))

    def _on_info(self, msg: CameraInfo) -> None:
        k = msg.k
        if k[0] > 0:
            self._k = Intrinsics(k[0], k[4], k[2], k[5])

    def _on_image(self, msg: Image) -> None:
        if self._k is None:
            return
        if self._pose_at is not None:
            self._queue.append((msg, time.monotonic()))
        elif self._pending is not None:
            self._scan(msg, self._pending[1])

    def _drain(self) -> None:
        """Convert the frames that have waited FLIGHT_DELAY: a scan asked for, or in flight."""
        now = time.monotonic()
        while self._queue and now - self._queue[0][1] >= FLIGHT_DELAY:
            msg, received = self._queue.popleft()
            if self._pending is not None and received >= self._pending[3]:
                pose = self._pose_at(_stamp_us(msg))
                if not isinstance(pose, str):
                    self._scan(msg, pose)
                    continue
                if now - self._pending[3] > SCAN_POSE_WAIT:
                    self.dropped[f'scan with the pose at the request ({pose})'] += 1
                    self._scan(msg, self._pending[1])
                    continue
            self._in_flight(msg)

    def _scan(self, msg: Image, pose: Pose) -> None:
        """Convert msg as the scan asked for, with pose."""
        label, _, done, _ = self._pending
        self._pending = None
        try:
            depth = image_to_depth(msg)
            pts = depth_to_points(depth, self._k, pose)
            rays = depth_rays(depth, self._k, pose)
        except ValueError:
            done(False)
            return
        self.rays.append(rays)
        self.frames.append(rays)
        self._last_view = tuple(pose)
        self.clouds.append(pts)
        self.labels.append(label)
        self.poses.append(tuple(float(v) for v in pose))
        self.stamps.append(_stamp_us(msg) * 1e-6)
        done(len(pts) > 0)

    def _in_flight(self, msg: Image) -> None:
        self.received += 1
        stamp = _stamp_us(msg)
        pose = self._pose_at(stamp)
        if isinstance(pose, str):
            self.dropped[pose] += 1
            return
        if not new_viewpoint(self._last_view, pose):
            return
        try:
            depth = image_to_depth(msg)
        except ValueError:
            self.dropped['unreadable'] += 1
            return
        cam, ends, hit = depth_rays(depth, self._k, pose, stride=FLIGHT_STRIDE)
        self.frames.append((cam, ends.astype(np.float32), hit))
        self.flight_poses.append((stamp * 1e-6, *(float(v) for v in pose)))
        pts = ends[hit]
        self.flight_points.append(pts[::max(1, len(pts) // KEEP_POINTS)].astype(np.float32))
        self._last_view = tuple(pose)
