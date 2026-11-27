# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Depth images to points, eave lines and their segments.

Pure numpy, no ROS: tested with synthetic point clouds (test/test_geometry.py).
纯 numpy、不依赖 ROS，用合成点云测试。

Frames: points are (north, east, height) in metres, in the same frame as the
vehicle pose (the plan frame); height is above the takeoff point.
点的坐标是（北, 东, 高度），与飞机位姿同一坐标系，高度相对起飞点。
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# PX4 x500_depth: OAK-D Lite 0.132 m ahead, 0.26 m above the body origin,
# level and facing forward. (forward, right, down) in metres.
# 相机安装位置：机体前方 0.132 米、上方 0.26 米，水平朝前。
CAMERA_OFFSET_FRD = (0.132, 0.0, -0.26)

# Effective depth range R (docs/OBSTACLE_AVOIDANCE.md section 3): readings
# beyond it are dropped, as if nothing had returned. A real OAK-D Lite is not
# reliable on leaves much beyond 10 m, and M4 is to work with 10 m. Not yet:
# the M3 survey expects to see a whole building from one side, and with 10 m
# it saw only the front 3 m of a 10 m deep townhouse row and put a station
# over the unseen back eave. 18.5 m (just inside Gazebo's 19.1 m clip) until
# the survey explores through seen-free space (M4, OBSTACLE_AVOIDANCE.md
# section 9). 有效深度距离 R：超出的读数一律丢弃，当作没有回波。真实的 OAK-D Lite
# 对 10 米外的树叶不可靠，M4 要在 10 米下工作。但现在还不行：M3 的扫描默认从一侧就能
# 看到整栋建筑，10 米时只看到 10 米深联排的前 3 米，把一个站点排到了没看到的后檐上方。
# 在扫描改为只穿过已确认为空的空间逐步探索之前，取 18.5 米。
DEPTH_RANGE = 18.5

# Depth frames taken in flight, for the voxel map: a new one each time the
# vehicle has moved this far or turned this much since the last frame kept
# (scans included), sampled every FLIGHT_STRIDE pixels. 飞行中给体素地图取的深度帧：
# 自上一帧（含扫描）以来移动了这么远或转了这么多时取一帧，每隔 FLIGHT_STRIDE 个像素取样。
FLIGHT_STEP = 1.0          # m
FLIGHT_TURN_DEG = 15.0
FLIGHT_STRIDE = 8

COMPASS = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']


def compass(bearing_deg: float) -> str:
    return COMPASS[int(((bearing_deg % 360) + 22.5) // 45) % 8]


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def new_viewpoint(last: Optional[Sequence[float]], pose: Sequence[float],
                  step: float = FLIGHT_STEP, turn_deg: float = FLIGHT_TURN_DEG) -> bool:
    """
    Tell whether pose (north, east, height, heading, ...) is a new viewpoint after last.

    New when there is no last one, or the vehicle has moved at least step
    metres or turned at least turn_deg since. 没有上一帧，或者移动了至少 step 米、
    转了至少 turn_deg 度时算新的视点。
    """
    if last is None:
        return True
    if math.dist(last[:3], pose[:3]) >= step:
        return True
    turn = (pose[3] - last[3] + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(turn)) >= turn_deg


def body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix taking body FRD vectors to NED (PX4 convention, Z-Y-X)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def depth_to_points(depth: np.ndarray, k: Intrinsics,
                    pose: Sequence[float],
                    offset_frd: Tuple[float, float, float] = CAMERA_OFFSET_FRD,
                    stride: int = 4, max_range: float = DEPTH_RANGE) -> np.ndarray:
    """
    Convert a depth image to world points.

    depth: (H, W) metres along the optical axis (as Gazebo and OAK-D give it);
    inf/nan/0 mean no return. pose: (north, east, height, heading[, roll,
    pitch]) of the vehicle, angles in radians. Without roll and pitch the
    camera is taken as level; 1 degree of tilt is ~0.26 m of error at 15 m.
    位姿可以带横滚和俯仰；不带时按水平处理，倾斜 1 度在 15 米处约有 0.26 米误差。
    Returns an (N, 3) array of (north, east, height).
    """
    d = depth[::stride, ::stride].astype(np.float64)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    ok = np.isfinite(d) & (d > 0.05) & (d < max_range)
    d, u, v = d[ok], u[ok], v[ok]
    # Optical frame (x right, y down, z forward) -> body FRD.
    fwd = d
    right = (u - k.cx) / k.fx * d
    down = (v - k.cy) / k.fy * d
    body = np.vstack([fwd + offset_frd[0], right + offset_frd[1], down + offset_frd[2]])
    n0, e0, h0, yaw = pose[:4]
    roll, pitch = (pose[4], pose[5]) if len(pose) >= 6 else (0.0, 0.0)
    ned = body_to_ned(roll, pitch, yaw) @ body
    return np.column_stack([n0 + ned[0], e0 + ned[1], h0 - ned[2]])


def intrinsics_from_fov(width: int, height: int, hfov_rad: float) -> Intrinsics:
    """Pinhole intrinsics of an ideal camera with square pixels (as Gazebo renders)."""
    f = (width / 2) / math.tan(hfov_rad / 2)
    return Intrinsics(f, f, width / 2, height / 2)


# PX4 x500_depth colour camera (OAK-D Lite IMX214 in Gazebo): 1920x1080, hfov 1.204 rad.
RGB_X500_DEPTH = intrinsics_from_fov(1920, 1080, 1.204)


def project(points: np.ndarray, k: Intrinsics, pose: Sequence[float],
            offset_frd: Tuple[float, float, float] = CAMERA_OFFSET_FRD) -> np.ndarray:
    """
    Project world points (N, 3) into the image: (N, 3) of (u, v, depth).

    The inverse of depth_to_points. depth <= 0 means behind the camera.
    depth_to_points 的逆运算。depth 不大于 0 表示在相机后方。
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    n0, e0, h0, yaw = pose[:4]
    roll, pitch = (pose[4], pose[5]) if len(pose) >= 6 else (0.0, 0.0)
    r = body_to_ned(roll, pitch, yaw)
    cam = np.array([n0, e0, -h0]) + r @ np.asarray(offset_frd)
    ned = np.column_stack([p[:, 0], p[:, 1], -p[:, 2]]) - cam
    body = ned @ r              # = (r.T @ ned.T).T
    fwd, right, down = body[:, 0], body[:, 1], body[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = k.cx + k.fx * right / fwd
        v = k.cy + k.fy * down / fwd
    return np.column_stack([u, v, fwd])


def depth_rays(depth: np.ndarray, k: Intrinsics, pose: Sequence[float],
               offset_frd: Tuple[float, float, float] = CAMERA_OFFSET_FRD,
               stride: int = 4, max_range: float = DEPTH_RANGE
               ) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray]:
    """
    Return the camera position and every sampled ray of a depth image.

    (camera (north, east, height), ends (N, 3), hit (N,)): a ray with a return
    ends at the point it hit; one without (nothing within max_range) ends at
    max_range along its direction, and the space it crossed is just as empty.
    返回相机位置和每条射线：有回波的终点是击中点；没有回波的终点取最大量程处，
    它经过的空间同样是空的。
    """
    d = depth[::stride, ::stride].astype(np.float64)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u, v = u.ravel() * stride, v.ravel() * stride
    d = d.ravel()
    hit = np.isfinite(d) & (d > 0.05) & (d < max_range)
    x, y = (u - k.cx) / k.fx, (v - k.cy) / k.fy
    # Rays without a return: as far as max_range along the ray (z-depth = range / norm).
    norm = np.sqrt(1 + x * x + y * y)
    d = np.where(hit, d, max_range / norm)
    body = np.vstack([d + offset_frd[0], x * d + offset_frd[1], y * d + offset_frd[2]])
    n0, e0, h0, yaw = pose[:4]
    roll, pitch = (pose[4], pose[5]) if len(pose) >= 6 else (0.0, 0.0)
    r = body_to_ned(roll, pitch, yaw)
    ned = r @ body
    cam = r @ np.array(offset_frd)
    ends = np.column_stack([n0 + ned[0], e0 + ned[1], h0 - ned[2]])
    return (n0 + cam[0], e0 + cam[1], h0 - cam[2]), ends, hit


@dataclass
class Eave:
    """One eave edge of the roof, where a gutter hangs."""

    name: str                       # e.g. "south gutter", by the side it faces
    start: Tuple[float, float]      # (north, east)
    end: Tuple[float, float]
    height: float                   # m, roof edge / gutter top
    side_bearing_deg: float         # compass bearing pointing away from the house

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    def point(self, t: float) -> Tuple[float, float]:
        """Point at fraction t (0..1) along the eave."""
        return (self.start[0] + t * (self.end[0] - self.start[0]),
                self.start[1] + t * (self.end[1] - self.start[1]))

    @property
    def outward(self) -> Tuple[float, float]:
        b = math.radians(self.side_bearing_deg)
        return math.cos(b), math.sin(b)


def _extent(x: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> Tuple[float, float]:
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def _best_angle(xy: np.ndarray) -> float:
    """Angle (rad, 0..pi/2) of the minimum-area bounding rectangle, robust to outliers."""
    def area(a):
        c, s = math.cos(a), math.sin(a)
        u = xy[:, 0] * c + xy[:, 1] * s
        v = -xy[:, 0] * s + xy[:, 1] * c
        u0, u1 = _extent(u)
        v0, v1 = _extent(v)
        return (u1 - u0) * (v1 - v0)
    coarse = np.radians(np.arange(0.0, 90.0, 1.0))
    a0 = min(coarse, key=area)
    fine = a0 + np.radians(np.arange(-1.0, 1.01, 0.1))
    return float(min(fine, key=area)) % (math.pi / 2)


Point2 = Tuple[float, float]


def segment_eave(eave: Eave, segments: int) -> List[Tuple[Point2, Point2]]:
    """Split an eave into equal parts; returns [(start, end), ...]."""
    return [(eave.point(i / segments), eave.point((i + 1) / segments))
            for i in range(segments)]


def segments_for(eave: Eave, max_length: float) -> int:
    return max(1, int(math.ceil(eave.length / max_length - 1e-6)))


def merge(clouds: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    clouds = [c for c in clouds if c is not None and len(c)]
    return np.vstack(clouds) if clouds else None
