# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Depth images to points, and points to a house with eave lines.

Pure numpy, no ROS: tested with synthetic point clouds (test/test_geometry.py).
纯 numpy、不依赖 ROS，用合成点云测试。

Frames: points are (north, east, height) in metres, in the same frame as the
vehicle pose (the plan frame); height is above the takeoff point.
点的坐标是（北, 东, 高度），与飞机位姿同一坐标系，高度相对起飞点。
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# PX4 x500_depth: OAK-D Lite 0.132 m ahead, 0.26 m above the body origin,
# level and facing forward. (forward, right, down) in metres.
# 相机安装位置：机体前方 0.132 米、上方 0.26 米，水平朝前。
CAMERA_OFFSET_FRD = (0.132, 0.0, -0.26)

COMPASS = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']


def compass(bearing_deg: float) -> str:
    return COMPASS[int(((bearing_deg % 360) + 22.5) // 45) % 8]


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


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
                    stride: int = 4, max_range: float = 18.5) -> np.ndarray:
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


@dataclass
class HouseEstimate:
    center: Tuple[float, float]     # (north, east)
    length: float                   # m, along the ridge, outline incl. overhang and gutters
    width: float                    # m, across the ridge
    ridge_bearing_deg: float        # 0..180
    ridge_height: float
    eaves: List[Eave] = field(default_factory=list)
    # Height of the ground in the same frame, from points around the house.
    # Heights here are in the vehicle's altitude frame, which drifts (barometer):
    # good for flying (same frame), subtract ground_level to compare with reality.
    # 高度与飞机高度在同一坐标系（会因气压计漂移），飞行用它没问题；与实际比较时减去地面高度。
    ground_level: float = 0.0
    n_points: int = 0
    notes: List[str] = field(default_factory=list)

    def corners(self) -> List[Tuple[float, float]]:
        b = math.radians(self.ridge_bearing_deg)
        a = (math.cos(b), math.sin(b))           # along the ridge
        c = (-math.sin(b), math.cos(b))          # across
        out = []
        for sa, sc in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            out.append((self.center[0] + sa * self.length / 2 * a[0] + sc * self.width / 2 * c[0],
                        self.center[1] + sa * self.length / 2 * a[1] + sc * self.width / 2 * c[1]))
        return out


class FitError(RuntimeError):
    """Not enough of the house was seen to fit it."""


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


def fit_house(points: np.ndarray, approx_center: Tuple[float, float], radius: float,
              min_height: float = 0.8, band: float = 0.6, edge_band: float = 0.15,
              min_points: int = 500) -> HouseEstimate:
    """
    Fit a gabled house to world points: outline, ridge direction, eave lines.

    Keeps points within radius of approx_center and above min_height (drops
    the ground). The outline is the minimum-area rectangle around them, which
    includes the overhang and gutters. Of its four edges, the two where the
    building is lowest are the eaves; the other two are gable ends.
    以最小面积外接矩形作为房屋轮廓；四条边中最低的两条是屋檐，另外两条是山墙。
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('points must be (N, 3)')
    rel = p[:, :2] - np.asarray(approx_center)
    dist = np.hypot(rel[:, 0], rel[:, 1])
    around = (dist > radius) & (np.abs(p[:, 2]) < 1.5)
    ground = float(np.median(p[around, 2])) if around.sum() >= 50 else 0.0
    keep = (dist <= radius) & (p[:, 2] >= ground + min_height)
    p, rel = p[keep], rel[keep]
    if len(p) < min_points:
        raise FitError(f'only {len(p)} points on the house (need {min_points}); '
                       'was it in view of the scans?')

    a = _best_angle(rel)
    c, s = math.cos(a), math.sin(a)
    # Axis u at angle a from north (towards east); v at a + 90 deg.
    u = rel[:, 0] * c + rel[:, 1] * s
    v = -rel[:, 0] * s + rel[:, 1] * c
    u0, u1 = _extent(u)
    v0, v1 = _extent(v)
    z = p[:, 2]

    def edge_height(sel) -> Tuple[float, int]:
        n = int(sel.sum())
        return (float(np.percentile(z[sel], 90)) if n else float('nan')), n

    def eave_height(key) -> float:
        """Median height right at the outer edge: the gutter lip / roof edge."""
        coord, lo, hi = (u, u0, u1) if key[0] == 'u' else (v, v0, v1)
        sel = (coord <= lo + edge_band) if key[1] == '0' else (coord >= hi - edge_band)
        if sel.sum() >= 20:
            return float(np.median(z[sel]))
        return edges[key][0]

    edges = {
        'u0': edge_height(u <= u0 + band), 'u1': edge_height(u >= u1 - band),
        'v0': edge_height(v <= v0 + band), 'v1': edge_height(v >= v1 - band),
    }
    notes = []
    for k, (hgt, n) in edges.items():
        if n < 30:
            notes.append(f'edge {k} barely seen ({n} points)')
    mean_u = np.nanmean([edges['u0'][0], edges['u1'][0]])
    mean_v = np.nanmean([edges['v0'][0], edges['v1'][0]])
    # Eaves are the low pair. Edges at u0/u1 run along v, so the ridge runs along v.
    # 低的一对是屋檐。u0/u1 两条边沿 v 方向，此时屋脊沿 v。
    if mean_u < mean_v:
        ridge_axis_deg = math.degrees(a) + 90.0
        eave_keys = ('u0', 'u1')
        length, width = v1 - v0, u1 - u0
    else:
        ridge_axis_deg = math.degrees(a)
        eave_keys = ('v0', 'v1')
        length, width = u1 - u0, v1 - v0
    if abs(mean_u - mean_v) < 0.4:
        notes.append(f'eaves and gables at similar height ({mean_u:.1f} vs {mean_v:.1f} m): '
                     'roof shape unclear')

    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    center = (approx_center[0] + cu * c - cv * s, approx_center[1] + cu * s + cv * c)

    def to_world(uu, vv):
        return (round(approx_center[0] + uu * c - vv * s, 3),
                round(approx_center[1] + uu * s + vv * c, 3))

    eaves = []
    for key in eave_keys:
        hgt = eave_height(key)
        if key in ('u0', 'u1'):
            uu = u0 if key == 'u0' else u1
            start, end = to_world(uu, v0), to_world(uu, v1)
            out_deg = math.degrees(a) + (180.0 if key == 'u0' else 0.0)
        else:
            vv = v0 if key == 'v0' else v1
            start, end = to_world(u0, vv), to_world(u1, vv)
            out_deg = math.degrees(a) + (270.0 if key == 'v0' else 90.0)
        out_deg %= 360.0
        eaves.append(Eave(compass(out_deg) + ' gutter', start, end, round(hgt, 2),
                          round(out_deg, 1)))
    if eaves[0].name == eaves[1].name:   # can only happen through rounding
        eaves[1].name += ' (2)'
    return HouseEstimate(center=(round(center[0], 3), round(center[1], 3)),
                         length=round(length, 3), width=round(width, 3),
                         ridge_bearing_deg=round(ridge_axis_deg % 180.0, 2),
                         ridge_height=round(float(np.percentile(z, 99.5)), 2),
                         eaves=eaves, n_points=int(len(p)), notes=notes,
                         ground_level=round(ground, 3))


Point2 = Tuple[float, float]


def segment_eave(eave: Eave, segments: int) -> List[Tuple[Point2, Point2]]:
    """Split an eave into equal parts; returns [(start, end), ...]."""
    return [(eave.point(i / segments), eave.point((i + 1) / segments))
            for i in range(segments)]


def segments_for(eave: Eave, max_length: float) -> int:
    return max(1, int(math.ceil(eave.length / max_length - 1e-6)))


def to_rotated_box_args(est: HouseEstimate) -> dict:
    """Arguments for wukong_flight.skills.RotatedBox (kept here free of that import)."""
    return {'center': est.center, 'length': est.length, 'width': est.width,
            'bearing_deg': est.ridge_bearing_deg, 'top': est.ridge_height}


def merge(clouds: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    clouds = [c for c in clouds if c is not None and len(c)]
    return np.vstack(clouds) if clouds else None
