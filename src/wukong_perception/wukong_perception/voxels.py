# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Three-state voxel map: occupied, seen free, unknown (docs/OBSTACLE_AVOIDANCE.md, L2).

A route may only pass through voxels seen free. Free means that depth rays
actually crossed the voxel; nothing is inferred about the space around a ray,
above it or behind what it hit. (The older heightmap.FreeSpace keeps, per
column, the lowest height a ray crossed and takes everything in the column
to be lower: wrong under a branch, an eave or a porch roof.)
只有深度射线实际穿过的格子才算确认为空；不推断射线旁边、上方或击中点后方的空间。
（heightmap.FreeSpace 按列记录射线穿过的最低高度，认为整列都比它低：树枝、屋檐、
门廊顶下面就错了。）

Occupied wins over free: a hit makes a voxel occupied at once, whatever was
seen there before (something may have come in). It counts as free again only
after clear_ratio free crossings per hit, all made after the last hit. A
stereo camera may match the wall behind a thin branch, and that ray then
seems to pass through the branch. 已占据优先：一次击中立即让格子变成占据，不管之前
看到过什么（可能有东西进来了）。之后要在最后一次击中以后、每次击中再有 clear_ratio
次空射线穿过，才重新算空。双目相机可能在细枝处匹配到后面的墙，这条射线看起来就
“穿过”了细枝。

Pure numpy, no ROS. Points are (north, east, height), as in geometry.py.
纯 numpy，不依赖 ROS。
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import DEPTH_RANGE

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


@dataclass(frozen=True)
class VoxelConfig:
    res: float = 0.25          # m, voxel edge
    size: float = 100.0        # m, square around the takeoff point
    floor: float = -2.0        # m, lowest height kept (ground below the takeoff point)
    top: float = 32.0          # m, highest (the 30 m ceiling plus a margin)
    min_free: int = 1          # scans whose rays must cross a voxel before it is free
    clear_ratio: int = 3       # free crossings per hit to clear an occupied voxel
    # How far a ray without a return counts as free. In Gazebo no return means
    # nothing within the camera's clip range, so the whole ray is free. A real
    # stereo camera also returns nothing where it cannot match (sky, glare, a
    # blank wall), and there no return proves nothing: this must be 0 there
    # until measured (OBSTACLE_AVOIDANCE.md section 8). 没有回波的射线算空的距离。
    # Gazebo 中没有回波表示相机截止距离内什么都没有；真实的双目相机在匹配不上的地方
    # （天空、强光、无纹理的墙）也没有回波，那里不能证明什么，实测之前应取 0。
    no_return_free: float = DEPTH_RANGE


class VoxelMap:
    """Per voxel, around the takeoff point: hits, and free crossings since the last hit."""

    def __init__(self, center: Tuple[float, float], cfg: VoxelConfig = VoxelConfig()) -> None:
        self.cfg = cfg
        self.res = cfg.res
        self.origin = np.array([center[0] - cfg.size / 2, center[1] - cfg.size / 2, cfg.floor])
        n = int(math.ceil(cfg.size / cfg.res))
        self.shape = (n, n, int(math.ceil((cfg.top - cfg.floor) / cfg.res)))
        # Saturating counts, one per scan at most. 饱和计数，每次扫描最多加一。
        self.hits = np.zeros(self.shape, np.uint8)
        self.frees = np.zeros(self.shape, np.uint8)
        self.scans = 0

    # -- indexing ------------------------------------------------------------------------

    def _index(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (flat index, inside) of the voxels holding points (N, 3)."""
        ijk = np.floor((np.asarray(points, np.float64) - self.origin) / self.res).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)
        flat = np.ravel_multi_index(tuple(np.clip(ijk, 0, np.array(self.shape) - 1).T),
                                    self.shape)
        return flat, inside

    # -- adding scans --------------------------------------------------------------------

    def add_rays(self, cam: Sequence[float], ends: np.ndarray, hit: np.ndarray,
                 chunk: int = 2000) -> None:
        """
        Add one depth frame: rays from cam to ends, hit[i] if ray i ended on a surface.

        The format of geometry.depth_rays. A ray with a hit marks the voxels it
        crossed free, stopping one voxel short of the hit, and the hit voxel
        occupied; a ray without one marks free only its first no_return_free
        metres. Each voxel counts at most once per frame, and a voxel hit in
        the frame is not also counted free in it; a hit drops the free
        crossings seen before it. 有回波的射线：沿途到击中点前一格算空，击中的格子算
        占据；没有回波的射线只有前 no_return_free 米算空。每帧每个格子最多计一次，本帧被
        击中的格子不再计为空；击中会清掉此前的空射线计数。
        """
        cam = np.asarray(cam, np.float64)
        ends = np.asarray(ends, np.float64)
        hit = np.asarray(hit, bool)
        free_idx, hit_idx = [], []
        for k in range(0, len(ends), chunk):
            f, h = self._trace(cam, ends[k:k + chunk], hit[k:k + chunk])
            free_idx.append(f)
            hit_idx.append(h)
        hits = np.unique(np.concatenate(hit_idx)) if hit_idx else np.zeros(0, np.int64)
        frees = np.unique(np.concatenate(free_idx)) if free_idx else np.zeros(0, np.int64)
        frees = np.setdiff1d(frees, hits, assume_unique=True)
        _bump(self.hits, hits)
        # Only crossings after the last hit can clear a voxel. 只有最后一次击中之后的
        # 空射线才能清除。
        self.frees.reshape(-1)[hits] = 0
        _bump(self.frees, frees)
        self.scans += 1

    def _trace(self, cam: np.ndarray, ends: np.ndarray, hit: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        vec = ends - cam
        length = np.linalg.norm(vec, axis=1)
        if not len(length):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        step = self.res / 2
        reach = np.where(hit, length - self.res, np.minimum(length, self.cfg.no_return_free))
        steps = int(np.ceil(max(reach.max(), 0.0) / step))
        free = np.zeros(0, np.int64)
        if steps > 0:
            t = np.arange(steps + 1) * step                          # metres along the ray
            ok = t[None, :] <= reach[:, None]
            unit = vec / np.maximum(length, 1e-9)[:, None]
            pts = cam[None, None, :] + t[None, :, None] * unit[:, None, :]
            flat, inside = self._index(pts[ok])
            free = np.unique(flat[inside])
        hit_flat, inside = self._index(ends[hit])
        return free, np.unique(hit_flat[inside])

    def assume_free(self, inside: 'Region') -> None:
        """
        Count the voxels of a region as crossed once by a free ray, without a scan.

        For space known to be empty without a depth ray: the column above the
        takeoff point that the operator confirmed open, and the space the
        drone's body has been in. A later hit still makes them occupied.
        用于不靠深度射线也知道是空的空间：操作员确认开阔的起飞点上方竖直通道、飞机机体
        到过的地方。之后若被击中，照样变成占据。
        """
        _bump(self.frees, self._voxels_in(inside))

    def _voxels_in(self, region: 'Region') -> np.ndarray:
        lo = np.floor((np.asarray(region.low) - self.origin) / self.res).astype(np.int64)
        hi = np.floor((np.asarray(region.high) - self.origin) / self.res).astype(np.int64)
        lo = np.clip(lo, 0, np.array(self.shape) - 1)
        hi = np.clip(hi, 0, np.array(self.shape) - 1)
        axes = [np.arange(lo[k], hi[k] + 1) for k in range(3)]
        ijk = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        # Only voxels wholly inside: free is the claim that must be safe.
        # 只取完全在区域内的体素：“空”是必须保证安全的那一方。
        keep = np.ones(len(ijk), bool)
        for corner in np.stack(np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij'),
                               axis=-1).reshape(-1, 3):
            keep &= region.contains(self.origin + (ijk + corner) * self.res)
        return np.ravel_multi_index(tuple(ijk[keep].T), self.shape)

    # -- queries -------------------------------------------------------------------------

    def states(self, points: np.ndarray) -> np.ndarray:
        """Return UNKNOWN, FREE or OCCUPIED for each point (N, 3); outside the map: UNKNOWN."""
        flat, inside = self._index(np.atleast_2d(points))
        return np.where(inside, self._state(flat), UNKNOWN)

    def _state(self, flat: np.ndarray) -> np.ndarray:
        h = self.hits.ravel()[flat].astype(np.int32)
        f = self.frees.ravel()[flat].astype(np.int32)
        occupied = (h > 0) & (f < self.cfg.clear_ratio * h)
        free = ~occupied & (f >= np.where(h > 0, self.cfg.clear_ratio * h, self.cfg.min_free))
        return np.where(occupied, OCCUPIED, np.where(free, FREE, UNKNOWN))

    def first_blocked(self, a: Sequence[float], b: Sequence[float], radius: float,
                      half_height: Optional[float] = None) -> Optional[float]:
        """
        Return how far from a along a -> b the swept body first meets a voxel not seen free.

        The body is an upright cylinder of the given radius reaching half_height
        above and below the path (default: radius): a multirotor is flat, and
        its height error is smaller than its position error. None if all it
        sweeps from a to b (the body and its margin) lies in free voxels; a voxel
        counts if any part of it is inside. 机体按竖直圆柱计算（半径 radius，上下各
        half_height）：多旋翼是扁的，高度误差也比水平误差小。扫过的空间全部在确认为空
        的格子里时返回 None，否则返回第一次碰到其他格子的位置离 a 多远。

        The path is cut into steps of at most res / 2. What the body sweeps in
        one step lies inside the body at the step's midpoint grown by half the
        step (across and up separately), so every point of the path is
        covered, not only sample points; the distance returned is the start of
        the first step that may touch, never past the first contact.
        路径切成不超过 res / 2 的小段。一小段里机体扫过的空间，包含在这一段中点处、按半段
        位移加大（水平、竖直分别加）的机体里，所以路径上每一点都被覆盖，而不只是采样点；
        返回的是第一个可能接触的小段的起点，不会超过第一次接触的位置。
        """
        if half_height is None:
            half_height = radius
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        length = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(length / (self.res / 2))))
        t = np.linspace(0.0, length, n + 1)[:-1]                         # where each step starts
        half_move = (b - a) / n / 2
        centres = a + (t / max(length, 1e-9))[:, None] * (b - a) + half_move   # (steps, 3)
        radius = radius + float(np.linalg.norm(half_move[:2]))
        half_height = half_height + abs(float(half_move[2]))
        rh = int(math.ceil(radius / self.res)) + 1
        rv = int(math.ceil(half_height / self.res)) + 1
        dh, dv = np.arange(-rh, rh + 1), np.arange(-rv, rv + 1)
        box = np.stack(np.meshgrid(dh, dh, dv, indexing='ij'), axis=-1).reshape(-1, 3)
        ijk = np.floor((centres - self.origin) / self.res).astype(np.int64)
        cells = ijk[:, None, :] + box[None, :, :]                        # (samples, box, 3)
        mid = self.origin + (cells + 0.5) * self.res
        # Nearest gaps from the sample to each voxel. 采样点到各体素的最近距离。
        gap = np.maximum(np.abs(centres[:, None, :] - mid) - self.res / 2, 0.0)
        near = (np.linalg.norm(gap[..., :2], axis=2) < radius) & (gap[..., 2] < half_height)
        inside = np.all((cells >= 0) & (cells < np.array(self.shape)), axis=2)
        flat = np.ravel_multi_index(
            tuple(np.clip(cells, 0, np.array(self.shape) - 1).reshape(-1, 3).T), self.shape)
        ok = ~near | ((self._state(flat).reshape(near.shape) == FREE) & inside)
        if length > 0:
            # The drone is already at a: the half of its margin behind it is never
            # entered (a level camera does not see beside or behind the drone).
            # 飞机已经在 a 处，身后那一半余量不会进入（水平的相机看不到机身旁边和后方）。
            u = (b - a) / length
            ok |= ((mid - a) @ u) < -self.res / 2
        bad = np.flatnonzero(~ok.all(axis=1))
        return float(t[bad[0]]) if len(bad) else None


@dataclass(frozen=True)
class Ball:
    """A sphere: where the drone's body has been."""

    centre: Tuple[float, float, float]
    radius: float

    @property
    def low(self) -> Tuple[float, float, float]:
        return tuple(c - self.radius for c in self.centre)

    @property
    def high(self) -> Tuple[float, float, float]:
        return tuple(c + self.radius for c in self.centre)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return whether each point (N, 3) is inside."""
        return np.linalg.norm(points - np.asarray(self.centre), axis=1) <= self.radius


@dataclass(frozen=True)
class Column:
    """An upright cylinder: the open column above the takeoff point."""

    centre: Tuple[float, float]
    radius: float
    bottom: float
    top: float

    @property
    def low(self) -> Tuple[float, float, float]:
        return (self.centre[0] - self.radius, self.centre[1] - self.radius, self.bottom)

    @property
    def high(self) -> Tuple[float, float, float]:
        return (self.centre[0] + self.radius, self.centre[1] + self.radius, self.top)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return whether each point (N, 3) is inside."""
        d = np.linalg.norm(points[:, :2] - np.asarray(self.centre), axis=1)
        return (d <= self.radius) & (points[:, 2] >= self.bottom) & (points[:, 2] <= self.top)


Region = Union[Ball, Column]


def _bump(counts: np.ndarray, flat: np.ndarray) -> None:
    """Add one to counts at the flat indices, saturating at the dtype's maximum."""
    if not len(flat):
        return
    c = counts.reshape(-1)
    c[flat] = np.minimum(c[flat].astype(np.int32) + 1, np.iinfo(counts.dtype).max)
