# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Buildings of any shape from a height map: the nearest building and all its eaves.

The merged survey points go into a 2.5D grid (highest point per cell). Raised
cells that touch form a building; the one nearest the takeoff point is the
target. An eave is where the roof ends in a drop and the roof slopes down
towards that drop; where the slope runs along the drop it is a rake (gable
edge), which has no gutter. Eaves at different heights (upper roof, garage,
porch) come out the same way, so multi-storey houses need no special case.
二维半高度图：相连的高出地面的格子组成一栋建筑，离起飞点最近的是目标。
屋面在边缘处陡降、且屋面朝这个边缘向下倾斜的地方是屋檐；坡度沿边缘方向的是山墙边。
多层屋檐用同样的方法得到。
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import compass, Eave

Cell = Tuple[int, int]


def ground_height(z: np.ndarray, span: float = 1.5, bin_size: float = 0.05) -> float:
    """
    Height of the ground: the peak of the height histogram near zero.

    The ground is one large flat surface, so it is by far the tallest peak;
    a median would be pulled up by the many wall points near the ground.
    地面是一大片平面，在高度直方图中是最高的峰；用中位数会被墙面点拉高。
    """
    z = z[np.abs(z) < span]
    if len(z) < 20:
        return 0.0
    hist, edges = np.histogram(z, bins=np.arange(-span, span + bin_size, bin_size))
    k = int(np.argmax(hist))
    sel = z[(z >= edges[k]) & (z < edges[k + 1])]
    return float(np.median(sel))


@dataclass
class HeightMap:
    """Highest point per cell. Heights relative to the ground (ground_level removed)."""

    origin: Tuple[float, float]          # (north, east) of cell (0, 0)'s corner
    res: float
    top: np.ndarray                      # (rows=north, cols=east); nan = no points
    ground_level: float = 0.0

    @classmethod
    def from_points(cls, points: np.ndarray, res: float = 0.2,
                    ground_level: Optional[float] = None) -> 'HeightMap':
        p = np.asarray(points, dtype=np.float64)
        if ground_level is None:
            ground_level = ground_height(p[:, 2])
        h = p[:, 2] - ground_level
        n0, e0 = p[:, 0].min() - res, p[:, 1].min() - res
        i = ((p[:, 0] - n0) / res).astype(int)
        j = ((p[:, 1] - e0) / res).astype(int)
        top = np.full((i.max() + 2, j.max() + 2), -np.inf)
        np.maximum.at(top, (i, j), h)
        # Far-away roofs are sampled sparsely, so some roof cells hold only ground
        # points seen under the eaves. A grey closing (max then min over 3x3)
        # fills such one-cell holes without moving the roof's outer edge.
        # 远处屋顶的点稀疏，有的格子只有屋檐下的地面点；灰度闭运算补上这种单格空洞。
        top = _grey_erode(_grey_dilate(top))
        top[np.isneginf(top)] = np.nan
        return cls((n0, e0), res, top, ground_level)

    def cell(self, n: float, e: float) -> Cell:
        return int((n - self.origin[0]) / self.res), int((e - self.origin[1]) / self.res)

    def center(self, c: Cell) -> Tuple[float, float]:
        return (self.origin[0] + (c[0] + 0.5) * self.res, self.origin[1] + (c[1] + 0.5) * self.res)

    def raised(self, min_height: float = 1.0) -> np.ndarray:
        return np.nan_to_num(self.top, nan=-1.0) >= min_height


@dataclass
class Building:
    cells: np.ndarray                    # boolean mask over the height map
    distance: float                      # m, nearest cell to the takeoff point
    top: float                           # m above ground
    footprint_ne: List[Tuple[float, float]] = field(default_factory=list)


def buildings(hm: HeightMap, takeoff: Tuple[float, float], min_height: float = 1.5,
              min_area: float = 8.0, gap: float = 0.5) -> List[Building]:
    """Find connected raised regions, nearest first. Gaps up to `gap` m are bridged."""
    raised = hm.raised(min_height)
    # Close small holes (unseen cells under eaves, between scans). 补上小空洞。
    k = max(1, int(round(gap / hm.res)))
    grown = _dilate(raised, k)
    labels, n = _label(grown)
    out = []
    for lab in range(1, n + 1):
        mask = (labels == lab) & raised
        if mask.sum() * hm.res ** 2 < min_area:
            continue
        idx = np.argwhere(mask)
        centers = np.column_stack([hm.origin[0] + (idx[:, 0] + 0.5) * hm.res,
                                   hm.origin[1] + (idx[:, 1] + 0.5) * hm.res])
        dist = float(np.min(np.hypot(centers[:, 0] - takeoff[0], centers[:, 1] - takeoff[1])))
        out.append(Building(mask, dist, float(np.nanmax(hm.top[mask]))))
    out.sort(key=lambda b: b.distance)
    return out


def _shifts(a: np.ndarray, fill: float):
    p = np.pad(a, 1, constant_values=fill)
    r, c = a.shape
    return [p[1 + di:1 + di + r, 1 + dj:1 + dj + c] for di in (-1, 0, 1) for dj in (-1, 0, 1)]


def _grey_dilate(a: np.ndarray) -> np.ndarray:
    return np.max(_shifts(a, -np.inf), axis=0)


def _grey_erode(a: np.ndarray) -> np.ndarray:
    return np.min(_shifts(a, -np.inf), axis=0)


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    out = mask.copy()
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            if di * di + dj * dj <= k * k:
                out |= np.roll(np.roll(mask, di, 0), dj, 1)
    return out


def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label 8-connected components (BFS; the maps here are a few hundred cells a side)."""
    labels = np.zeros(mask.shape, int)
    n = 0
    rows, cols = mask.shape
    for start in map(tuple, np.argwhere(mask)):
        if labels[start]:
            continue
        n += 1
        labels[start] = n
        q = deque([start])
        while q:
            i, j = q.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < rows and 0 <= b < cols and mask[a, b] and not labels[a, b]:
                        labels[a, b] = n
                        q.append((a, b))
    return labels, n


# -- eaves --------------------------------------------------------------------------------

@dataclass
class EdgeCell:
    cell: Cell
    height: float
    out_dir: Tuple[float, float]         # unit (dn, de) towards the drop (coarse: 8 directions)
    kind: str                            # 'eave', 'rake' or 'other'
    # Outward direction for grouping: the roof's downhill direction at eaves
    # (continuous, works for any heading), else out_dir.
    # 分组用的朝外方向：屋檐处用屋面下坡方向（连续值，任意朝向都准确）。
    direction: Tuple[float, float] = (0.0, 0.0)


def _gradient(top: np.ndarray, i: int, j: int, r: int) -> Optional[Tuple[float, float]]:
    """Downhill direction of the roof around (i, j): plane fit over a (2r+1)^2 window."""
    sub = top[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
    ii, jj = np.mgrid[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
    ii, jj = ii[:sub.shape[0], :sub.shape[1]], jj[:sub.shape[0], :sub.shape[1]]
    hz = top[i, j]
    # Stay on this roof: not the wall below, not a taller part of the building next to it.
    ok = np.isfinite(sub) & (np.abs(sub - hz) < 0.8)
    if ok.sum() < 6:
        return None
    a = np.column_stack([ii[ok] - i, jj[ok] - j, np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(a, sub[ok], rcond=None)
    gi, gj = coef[0], coef[1]                    # uphill per cell
    m = math.hypot(gi, gj)
    if m < 0.03:                                 # flat (< ~8 deg at 0.2 m cells)
        return None
    return -gi / m, -gj / m


def edge_cells(hm: HeightMap, mask: np.ndarray, drop: float = 1.0,
               min_height: float = 1.8) -> List[EdgeCell]:
    """Find cells of the building where the roof ends in a drop of more than `drop` m."""
    top = hm.top
    out = []
    rows, cols = top.shape
    look = max(3, int(round(1.2 / hm.res)))      # look this far outwards for the drop
    for i, j in np.argwhere(mask):
        h = top[i, j]
        if not np.isfinite(h) or h < min_height:
            continue
        dn = de = 0.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            lower = False
            for s in range(1, look + 1):
                a, b = i + di * s, j + dj * s
                if not (0 <= a < rows and 0 <= b < cols):
                    lower = True
                    break
                t = top[a, b]
                if not np.isfinite(t):
                    continue                       # no data here: look further out
                if t >= h - 0.3:
                    break                          # roof continues
                if t < h - drop:
                    lower = True
                    break
            if lower:
                dn += di
                de += dj
        if dn == 0 and de == 0:
            continue
        m = math.hypot(dn, de)
        out_dir = (dn / m, de / m)
        g = _gradient(top, i, j, 3)
        if g is None:
            # Flat at the edge: kept as an eave candidate (a low-pitched porch can
            # read as flat); eave_lines drops apexes by checking the roof rises.
            # 边缘处看起来平：仍作为屋檐候选（坡度小的门廊可能显得平），
            # 山墙顶尖由 eave_lines 检查屋面是否上升来排除。
            kind, d = 'eave', out_dir
        else:
            cos = g[0] * out_dir[0] + g[1] * out_dir[1]
            kind = 'eave' if cos > 0.7 else ('rake' if abs(cos) < 0.5 else 'other')
            d = g if kind == 'eave' else out_dir
        out.append(EdgeCell((int(i), int(j)), float(h), out_dir, kind, d))
    return out


def eave_lines(hm: HeightMap, cells: List[EdgeCell], min_length: float = 2.0,
               height_tol: float = 0.35, angle_tol_deg: float = 20.0) -> List[Eave]:
    """Group eave cells into straight eaves: same outward direction, height, and adjacency."""
    eaves = [c for c in cells if c.kind == 'eave']
    used = [False] * len(eaves)
    index: Dict[Cell, int] = {c.cell: k for k, c in enumerate(eaves)}
    cos_tol = math.cos(math.radians(angle_tol_deg))
    reach = max(2, int(round(0.5 / hm.res)))
    groups = []
    for k0 in range(len(eaves)):
        if used[k0]:
            continue
        used[k0] = True
        group = [k0]
        # Compare with the group's mean direction, not the neighbour's: at a hip
        # corner the direction turns gradually and a neighbour chain would walk
        # round the corner, merging two eaves. 与组的平均方向比较，避免在四坡顶的
        # 转角处一格一格地拐过去，把两条屋檐连成一条。
        mean = np.array(eaves[k0].direction, dtype=float)
        q = deque([k0])
        while q:
            c = eaves[q.popleft()]
            i, j = c.cell
            for di in range(-reach, reach + 1):
                for dj in range(-reach, reach + 1):
                    k = index.get((i + di, j + dj))
                    if k is None or used[k]:
                        continue
                    o = eaves[k]
                    m = mean / (np.linalg.norm(mean) or 1.0)
                    if (abs(o.height - c.height) <= height_tol
                            and o.direction[0] * m[0] + o.direction[1] * m[1] >= cos_tol):
                        used[k] = True
                        group.append(k)
                        mean += o.direction
                        q.append(k)
        groups.append(group)

    lines = []
    for g in groups:
        pts = np.array([hm.center(eaves[k].cell) for k in g])
        hs = np.array([eaves[k].height for k in g])
        if len(g) < 3:
            continue
        # Line direction from the cells themselves (PCA); outward = the side the
        # roof slopes down to. 直线方向用主成分分析，朝外方向取屋面下坡的一侧。
        centred = pts - pts.mean(0)
        mean_dir = np.mean([eaves[k].direction for k in g], axis=0)
        mean_dir /= np.linalg.norm(mean_dir) or 1.0
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        along = vt[0]
        if np.ptp(centred @ along) < 1.0:
            # Too few cells for a reliable direction: across the roof's slope.
            # 格子太少时方向不可靠：取与屋面坡向垂直的方向。
            along = np.array([-mean_dir[1], mean_dir[0]])
        od = np.array([-along[1], along[0]])
        if od @ mean_dir < 0:
            od = -od
        s = centred @ along
        # Short pieces are kept: a sparsely seen eave (a low porch far away)
        # comes in pieces that only reach min_length once merged below.
        # 短的片段保留：点稀疏的屋檐（远处低矮的门廊）分成几段，合并后才够长。
        if s.max() - s.min() < 0.4:
            continue
        # Gutters hang on level eaves. A line whose height climbs along it is a
        # rake or a hip seen through misaligned scans, not an eave.
        # 水槽只装在水平的屋檐上；沿线高度在变化的是山墙斜边或斜脊，不是屋檐。
        if not _level(s, hs):
            continue
        # The eave line: the median across the edge cells. Scans that do not
        # line up exactly (position-estimate drift) widen the edge outwards, so
        # an outer percentile would follow the worst scan. 屋檐线取边缘格子的中位数：
        # 扫描没对齐时边缘会向外变宽，取外侧百分位会跟着最差的那次扫描走。
        off = (pts - pts.mean(0)) @ od
        base = pts.mean(0) + od * np.percentile(off, 50)
        start = base + along * s.min()
        end = base + along * s.max()
        bearing = math.degrees(math.atan2(od[1], od[0])) % 360
        lines.append(Eave(compass(bearing) + ' gutter',
                          (round(float(start[0]), 3), round(float(start[1]), 3)),
                          (round(float(end[0]), 3), round(float(end[1]), 3)),
                          round(float(np.median(hs)) + hm.ground_level, 3), round(bearing, 1)))
    lines = _merge_collinear(lines)
    lines = [_extend(hm, e) for e in lines]
    lines = _merge_collinear(lines)       # extended pieces may now overlap
    lines = _snap_corners(lines)
    # The rise test is only reliable enough for short lines, which is where the
    # false ones are (a gable apex is about 2 m); long eaves are kept as they are.
    # 屋面上升的检查只用于短线（假屋檐都短，山墙顶尖约 2 米），长屋檐不检查。
    lines = [e for e in lines if e.length >= min_length
             and (e.length >= 3.0 or _roof_rises(hm, e, min_rise=0.08))]
    lines.sort(key=lambda e: (-e.height, e.side_bearing_deg))
    _name_levels(lines, hm.ground_level)
    return lines


def _roof_rises(hm: HeightMap, e: Eave, inward: float = 1.0, min_rise: float = 0.15) -> bool:
    """
    Whether the roof rises going inwards from the eave, as it does above every eave.

    A gable's apex or a ridge end also ends in a drop and can look level, but
    inwards from it the roof stays level or falls. 每条屋檐往屋内方向屋面都会上升；
    山墙顶尖或屋脊端往里是平的或下降的。
    """
    o = np.array(e.outward)
    rows, cols = hm.top.shape
    rises = []
    for t in np.linspace(0.15, 0.85, 7):
        p = np.array(e.point(t)) - o * inward
        i, j = hm.cell(p[0], p[1])
        if 0 <= i < rows and 0 <= j < cols and np.isfinite(hm.top[i, j]):
            rises.append(hm.top[i, j] + hm.ground_level - e.height)
    if len(rises) < 3:
        return True                 # not enough seen to judge: keep it
    return float(np.median(rises)) >= min_rise


def _level(s: np.ndarray, h: np.ndarray, max_slope: float = 0.2, max_rise: float = 0.9) -> bool:
    """
    Whether heights h along positions s stay level (as along an eave).

    Loose enough for a gutter full of debris (a pile up to ~0.4 m high is just
    what must not be missed); rakes and hips climb at 0.4 to 0.6 m per metre.
    阈值要容得下堆满杂物的水槽（杂物堆约 0.4 米高，正是不能漏的）；
    山墙斜边和斜脊每米升高 0.4 到 0.6 米。
    """
    if len(s) < 3 or np.ptp(s) < 1e-6:
        return True
    slope = np.polyfit(s, h, 1)[0]
    rise = np.percentile(h, 90) - np.percentile(h, 10)
    return abs(slope) <= max_slope and rise <= max_rise


def _merge_collinear(lines: List[Eave], angle_deg: float = 10.0, offset: float = 0.7,
                     gap: float = 3.0, height_tol: float = 0.35) -> List[Eave]:
    """Join pieces of one eave split by gaps in the data (合并被数据空隙断开的同一条屋檐)."""
    lines = list(lines)
    merged = True
    while merged:
        merged = False
        for a in range(len(lines)):
            for b in range(a + 1, len(lines)):
                e, f = lines[a], lines[b]
                d = abs((e.side_bearing_deg - f.side_bearing_deg + 180) % 360 - 180)
                # A short piece's direction is uncertain: allow more angle.
                # 短片段的方向不准，允许更大的角度差。
                tol = angle_deg if min(e.length, f.length) >= 3.0 else 25.0
                if d > tol or abs(e.height - f.height) > height_tol:
                    continue
                o = np.array(e.outward)
                along = np.array([-o[1], o[0]])
                p0 = np.array(e.start)
                if abs((np.array(f.point(0.5)) - p0) @ o) > offset:
                    continue
                se = sorted(((np.array(e.start) - p0) @ along, (np.array(e.end) - p0) @ along))
                sf = sorted(((np.array(f.start) - p0) @ along, (np.array(f.end) - p0) @ along))
                if max(se[0], sf[0]) - min(se[1], sf[1]) > gap:
                    continue
                lo, hi = min(se[0], sf[0]), max(se[1], sf[1])
                le, lf = e.length, f.length
                w = le / (le + lf)
                bearing = (e.side_bearing_deg + ((f.side_bearing_deg - e.side_bearing_deg + 180)
                                                 % 360 - 180) * (1 - w)) % 360
                start, end = p0 + along * lo, p0 + along * hi
                lines[a] = Eave(e.name, (round(float(start[0]), 3), round(float(start[1]), 3)),
                                (round(float(end[0]), 3), round(float(end[1]), 3)),
                                round(w * e.height + (1 - w) * f.height, 3), round(bearing, 1))
                del lines[b]
                merged = True
                break
            if merged:
                break
    return lines


def _snap_corners(lines: List[Eave], reach: float = 1.5, height_tol: float = 0.4) -> List[Eave]:
    """
    Where two eaves at the same height meet at a corner, end both exactly there.

    At hip corners the last metre is often unseen (grazing view), so eaves stop
    short; the true corner is where their lines intersect.
    四坡顶转角处最后一米常常看不清；两条屋檐直线的交点就是真正的转角。
    """
    ends = [[np.array(e.start, float), np.array(e.end, float)] for e in lines]
    for a in range(len(lines)):
        for b in range(a + 1, len(lines)):
            e, f = lines[a], lines[b]
            if abs(e.height - f.height) > height_tol:
                continue
            d = abs((e.side_bearing_deg - f.side_bearing_deg + 180) % 360 - 180)
            if not 60 <= d <= 120:
                continue
            p, r = ends[a][0], ends[a][1] - ends[a][0]
            q, s = ends[b][0], ends[b][1] - ends[b][0]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-9:
                continue
            t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
            x = p + t * r
            ka = min((0, 1), key=lambda k: np.linalg.norm(ends[a][k] - x))
            kb = min((0, 1), key=lambda k: np.linalg.norm(ends[b][k] - x))
            if (np.linalg.norm(ends[a][ka] - x) <= reach
                    and np.linalg.norm(ends[b][kb] - x) <= reach):
                ends[a][ka] = x.copy()
                ends[b][kb] = x.copy()
    return [Eave(e.name, (round(float(s0[0]), 3), round(float(s0[1]), 3)),
                 (round(float(s1[0]), 3), round(float(s1[1]), 3)), e.height, e.side_bearing_deg)
            for e, (s0, s1) in zip(lines, ends)]


def _extend(hm: HeightMap, e: Eave, max_extend: float = 3.0) -> Eave:
    """
    Grow an eave along its line while the roof edge continues at its height.

    Cells right at a hip corner slope diagonally and are not classified as
    eave, so lines stop short of the corners; this walks them out to the end
    of the roof edge. 四坡顶转角处的格子坡向是斜的，屋檐线会在转角前停下；这里沿线延伸到屋面边缘的尽头。
    """
    o = np.array(e.outward)
    h = e.height - hm.ground_level
    rows, cols = hm.top.shape

    def top_at(p):
        i, j = hm.cell(*p)
        if 0 <= i < rows and 0 <= j < cols and np.isfinite(hm.top[i, j]):
            return hm.top[i, j]
        return -np.inf

    def roof_edge_at(p):
        inside = top_at(p - o * 0.15)
        outside = top_at(p + o * 0.6)
        return abs(inside - h) < 0.5 and outside < h - 1.0

    ends = [np.array(e.start, float), np.array(e.end, float)]
    d = ends[1] - ends[0]
    d /= np.linalg.norm(d) or 1.0
    for k, sign in ((0, -1.0), (1, 1.0)):
        p = ends[k]
        moved = 0.0
        misses = 0
        while moved < max_extend:
            q = p + sign * d * hm.res
            if roof_edge_at(q):
                p, misses = q, 0
            else:
                misses += 1
                if misses > 2:      # allow one or two missing cells
                    break
                p = q
            moved += hm.res
        ends[k] = p - sign * d * hm.res * misses
    return Eave(e.name, (round(float(ends[0][0]), 3), round(float(ends[0][1]), 3)),
                (round(float(ends[1][0]), 3), round(float(ends[1][1]), 3)),
                e.height, e.side_bearing_deg)


def _name_levels(lines: List[Eave], ground: float) -> None:
    """Give unique, readable names such as 'upper south gutter', 'lower south gutter (2)'."""
    if not lines:
        return
    hs = sorted(e.height - ground for e in lines)
    two_levels = hs[-1] - hs[0] > 1.5
    seen: Dict[str, int] = {}
    for e in lines:
        level = ''
        if two_levels:
            level = 'upper ' if e.height - ground > (hs[0] + hs[-1]) / 2 else 'lower '
        base = level + e.name
        seen[base] = seen.get(base, 0) + 1
        e.name = base if seen[base] == 1 else f'{base} ({seen[base]})'


@dataclass
class Rectangle:
    """An oriented rectangle: centre (north, east), length along bearing_deg, width across."""

    center: Tuple[float, float]
    length: float
    width: float
    bearing_deg: float

    def to_uv(self, p) -> Tuple[float, float]:
        b = math.radians(self.bearing_deg)
        dn, de = p[0] - self.center[0], p[1] - self.center[1]
        return dn * math.cos(b) + de * math.sin(b), -dn * math.sin(b) + de * math.cos(b)

    def to_ne(self, u: float, v: float) -> Tuple[float, float]:
        b = math.radians(self.bearing_deg)
        return (self.center[0] + u * math.cos(b) - v * math.sin(b),
                self.center[1] + u * math.sin(b) + v * math.cos(b))


def building_rectangle(hm: HeightMap, b: Building) -> Rectangle:
    """Minimum-area rectangle around a building's cells (outline incl. overhangs)."""
    from .geometry import _best_angle
    idx = np.argwhere(b.cells)
    pts = np.column_stack([hm.origin[0] + (idx[:, 0] + 0.5) * hm.res,
                           hm.origin[1] + (idx[:, 1] + 0.5) * hm.res])
    c = pts.mean(0)
    a = _best_angle(pts - c)
    u = (pts - c) @ np.array([math.cos(a), math.sin(a)])
    v = (pts - c) @ np.array([-math.sin(a), math.cos(a)])
    half = hm.res / 2
    u0, u1, v0, v1 = u.min() - half, u.max() + half, v.min() - half, v.max() + half
    rect = Rectangle((0.0, 0.0), u1 - u0, v1 - v0, math.degrees(a))
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    center = (float(c[0] + cu * math.cos(a) - cv * math.sin(a)),
              float(c[1] + cu * math.sin(a) + cv * math.cos(a)))
    return Rectangle(center, float(rect.length), float(rect.width), float(rect.bearing_deg))


class FreeSpace:
    """
    The lowest height at which a ray crossed each cell without hitting anything.

    Whatever stands in a cell is lower than that (minus a little). A fixed grid
    around the takeoff point, updated scan by scan. 每个格子上方最低的“空射线”高度：
    格子里的东西一定比它矮。以起飞点为中心的固定栅格，逐次扫描更新。
    """

    def __init__(self, center: Tuple[float, float], size: float = 120.0, res: float = 0.5,
                 step: float = 0.4) -> None:
        self.res, self.step = res, step
        self.origin = (center[0] - size / 2, center[1] - size / 2)
        n = int(size / res)
        self.ceiling = np.full((n, n), np.inf)

    def add(self, cam: Tuple[float, float, float], ends: np.ndarray, hit: np.ndarray,
            chunk: int = 3000) -> None:
        """Add one scan's rays (in chunks, to keep memory small)."""
        for k in range(0, len(ends), chunk):
            self._add(cam, ends[k:k + chunk], hit[k:k + chunk])

    def _add(self, cam, ends: np.ndarray, hit: np.ndarray) -> None:
        cam = np.asarray(cam, dtype=np.float64)
        vec = ends - cam
        length = np.linalg.norm(vec, axis=1)
        steps = int(np.ceil(length.max() / self.step)) if len(length) else 0
        if steps == 0:
            return
        t = np.arange(1, steps + 1) * self.step                       # metres along the ray
        frac = t[None, :] / np.maximum(length[:, None], 1e-6)
        # Stop short of a hit (the surface itself is not free space). 击中点前留一点余量。
        limit = np.where(hit, (length - self.step) / np.maximum(length, 1e-6), 1.0)
        ok = frac <= limit[:, None]
        pts = cam[None, None, :] + frac[..., None] * vec[:, None, :]
        pts = pts[ok]
        i = ((pts[:, 0] - self.origin[0]) / self.res).astype(int)
        j = ((pts[:, 1] - self.origin[1]) / self.res).astype(int)
        n = self.ceiling.shape[0]
        inside = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        np.minimum.at(self.ceiling, (i[inside], j[inside]), pts[inside, 2])

    def ceiling_at(self, north: float, east: float) -> float:
        i = int((north - self.origin[0]) / self.res)
        j = int((east - self.origin[1]) / self.res)
        n = self.ceiling.shape[0]
        if 0 <= i < n and 0 <= j < n:
            return float(self.ceiling[i, j])
        return float('inf')
