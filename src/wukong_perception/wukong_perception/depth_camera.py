# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Capture depth frames from ROS 2 and turn them into world points.

Topics (bridged by scripts/sim.sh for PX4's x500_depth):
  /depth/image        sensor_msgs/Image, 32FC1, metres along the optical axis
  /depth/camera_info  sensor_msgs/CameraInfo

Two kinds of frame. A scan is the next frame after a request (the vehicle
holds still, or nearly). A frame in flight is taken every FLIGHT_STEP metres
or FLIGHT_TURN_DEG degrees and goes only to the voxel map. Both are
converted with the full pose at the moment they were imaged, from pose_at,
with the clock the images are stamped with said by the caller. A scan whose
pose at that moment cannot be had within SCAN_POSE_WAIT is converted with
the pose at the request instead: it still shapes the house, but it proves no
space free (to the map with free=False, and not in rays).
两种帧：扫描是请求之后的下一帧（飞机悬停或接近悬停）；飞行中每移动 FLIGHT_STEP 米或
转 FLIGHT_TURN_DEG 度取一帧，只用于体素地图。两者都用 pose_at 给出的成像时刻完整位姿
转换，图像打时间戳用的时钟由调用方说明。SCAN_POSE_WAIT 内拿不到成像时刻位姿的扫描，
改用请求时的位姿转换：仍用于房屋形状，但不能证明任何空间为空（以 free=False 进入地图，
也不进 rays）。

The voxel map takes its frames with take_frames(): at most MAP_BACKLOG wait
for it, and those taken are let go. 体素地图用 take_frames() 取帧：最多等 MAP_BACKLOG 帧，
取走的帧不再保留。
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
# A frame's stamp (us) and its clock -> the pose then, or why it cannot be told
# well. 帧的时间戳（微秒）和时钟 -> 当时的位姿，或无法确定的原因。
PoseAt = Callable[[float, str], Union[Pose, str]]
# One frame for the voxel map: camera, ray ends, hit per ray, and whether it may
# count space free (its pose is that of the moment it was imaged).
# 体素地图的一帧：相机、射线终点、每条射线是否击中，以及能否计为空（位姿是成像时刻的）。
MapFrame = Tuple[Tuple[float, float, float], np.ndarray, np.ndarray, bool]

# s a frame in flight waits before it is converted, so that pose samples from
# after its stamp have arrived. 飞行中的帧等这么久再转换，让成像时刻之后的位姿样本先到。
FLIGHT_DELAY = 0.15
KEEP_POINTS = 300          # hit points kept per frame in flight, for checking
# s a scan waits for a frame with a pose at its moment, then takes the pose at
# the request. The pose cannot be had for a few seconds after PX4 resets its
# clock offset estimate; this, plus FLIGHT_DELAY and a frame, stays within the
# mission's photo_timeout (5 s). 扫描等这么久仍没有带成像时刻位姿的帧时，改用请求时的
# 位姿。PX4 重置时钟偏移估计后的几秒内拿不到位姿；这个时间加上 FLIGHT_DELAY 和一帧，
# 仍在任务的 photo_timeout（5 秒）以内。
SCAN_POSE_WAIT = 3.0
# Frames waiting for the voxel map at most (some 60 MB); beyond, the oldest are
# dropped, which only leaves the map knowing less. The map takes them at each
# decision, and a whole photo round may be flown without one: two_storey's
# took some 400 frames in Gazebo. 等待进入体素地图的帧最多这么多（约 60 MB）；
# 再多就丢掉最早的，只会让地图知道得少一些。地图在每次决定时取帧，而整轮拍照可能
# 中间没有一次决定：Gazebo 中 two_storey 的一轮取了约 400 帧。
MAP_BACKLOG = 1024


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
    """Capture depth frames: scans on request, and frames in flight."""

    def __init__(self, node: Node, pose_at: PoseAt, clock: str,
                 image_topic: str = '/depth/image',
                 info_topic: str = '/depth/camera_info') -> None:
        """
        Subscribe to the depth topics; clock is the one the images are stamped with.

        pose_history.PX4_CLOCK for Gazebo's images, COMPANION_CLOCK for a real
        camera's driver. 图像打时间戳用的时钟：Gazebo 用 PX4_CLOCK，真实相机驱动用
        COMPANION_CLOCK。
        """
        self._k: Optional[Intrinsics] = None
        self._pending: Optional[tuple] = None
        self._log = node.get_logger()
        self.clouds: List[np.ndarray] = []      # every successful capture, in order
        self.labels: List[str] = []
        self.poses: List[Tuple[float, ...]] = []   # vehicle pose used for each capture
        # (camera, ray ends, hit) per scan converted with the pose at the image
        # time: the evidence of free space. 用成像时刻位姿转换的扫描的射线：空的证据。
        self.rays: List[tuple] = []
        self.stamps: List[float] = []            # image time stamp, s
        self.at_request = 0                      # scans converted with the pose at the request
        self.backlog_dropped = 0                 # frames the voxel map never took
        self._map: Deque[MapFrame] = deque(maxlen=MAP_BACKLOG)
        self.flight_poses: List[Tuple[float, ...]] = []   # (stamp s, *pose) per frame in flight
        # A few of each frame's hit points, to check them against the truth.
        # 每帧的少量击中点，用于和真值核对。
        self.flight_points: List[np.ndarray] = []
        self.dropped: Counter = Counter()        # frames in flight not used, by reason
        self.received = 0                        # frames in flight looked at
        self._pose_at = pose_at
        self.clock = clock
        self._queue: Deque[Tuple[Image, float]] = deque(maxlen=8)
        self._last_view: Optional[Pose] = None
        node.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)
        node.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        node.create_timer(0.05, self._drain)

    @property
    def ready(self) -> bool:
        return self._k is not None

    def capture(self, label: str, pose: Pose, done: Callable[[bool], None]) -> None:
        """Convert the next frame, then done(ok); pose is where the vehicle hovers, about still."""
        self._pending = (label, pose, done, time.monotonic())

    def take_frames(self) -> List[MapFrame]:
        """Return the frames for the voxel map since the last call, in order, and let them go."""
        out = []
        while True:
            try:
                out.append(self._map.popleft())
            except IndexError:          # empty (frames arrive from the ROS thread)
                return out

    def describe_flight(self) -> str:
        """How many frames in flight were kept and why others were not, for the log."""
        why = ', '.join(f'{n} {r}' for r, n in sorted(self.dropped.items()))
        return (f'{len(self.flight_poses)} kept of {self.received}'
                + (f'; without a pose: {why}' if why else '')
                + f'; {self.at_request} scan(s) with the pose at the request'
                + (f'; {self.backlog_dropped} never mapped (backlog full)'
                   if self.backlog_dropped else ''))

    def _on_info(self, msg: CameraInfo) -> None:
        k = msg.k
        if k[0] > 0:
            self._k = Intrinsics(k[0], k[4], k[2], k[5])

    def _on_image(self, msg: Image) -> None:
        if self._k is not None:
            self._queue.append((msg, time.monotonic()))

    def _drain(self) -> None:
        """Convert the frames that have waited FLIGHT_DELAY: a scan asked for, or in flight."""
        now = time.monotonic()
        while self._queue and now - self._queue[0][1] >= FLIGHT_DELAY:
            msg, received = self._queue.popleft()
            if self._pending is not None and received >= self._pending[3]:
                pose = self._pose_at(_stamp_us(msg), self.clock)
                if not isinstance(pose, str):
                    self._scan(msg, pose, True)
                    continue
                if now - self._pending[3] > SCAN_POSE_WAIT:
                    self._log.warning(
                        f'scan "{self._pending[0]}": no pose at the image time after '
                        f'{SCAN_POSE_WAIT:g} s ({pose}); converted with the pose at the request, '
                        f'for the house only, not as free space')
                    self.at_request += 1
                    self._scan(msg, self._pending[1], False)
                    continue
            self._in_flight(msg)

    def _to_map(self, cam, ends: np.ndarray, hit: np.ndarray, free: bool) -> None:
        if len(self._map) == MAP_BACKLOG:
            self.backlog_dropped += 1           # the append drops the oldest
        self._map.append((cam, ends, hit, free))

    def _scan(self, msg: Image, pose: Pose, at_image_time: bool) -> None:
        """
        Convert msg as the scan asked for, with pose.

        Only a pose of the image time makes the scan evidence of free space.
        只有成像时刻的位姿才让扫描成为“空”的证据。
        """
        label, _, done, _ = self._pending
        self._pending = None
        try:
            depth = image_to_depth(msg)
            pts = depth_to_points(depth, self._k, pose)
            cam, ends, hit = depth_rays(depth, self._k, pose)
        except ValueError:
            done(False)
            return
        ends = ends.astype(np.float32)
        self._to_map(cam, ends, hit, at_image_time)
        if at_image_time:
            self.rays.append((cam, ends, hit))
            self._last_view = tuple(pose)
        self.clouds.append(pts)
        self.labels.append(label)
        self.poses.append(tuple(float(v) for v in pose))
        self.stamps.append(_stamp_us(msg) * 1e-6)
        done(len(pts) > 0)

    def _in_flight(self, msg: Image) -> None:
        self.received += 1
        stamp = _stamp_us(msg)
        pose = self._pose_at(stamp, self.clock)
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
        self._to_map(cam, ends.astype(np.float32), hit, True)
        self.flight_poses.append((stamp * 1e-6, *(float(v) for v in pose)))
        pts = ends[hit]
        self.flight_points.append(pts[::max(1, len(pts) // KEEP_POINTS)].astype(np.float32))
        self._last_view = tuple(pose)
