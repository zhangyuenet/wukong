# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Line scans up with each other before merging them.

Each scan is converted with the vehicle's position estimate at that moment.
GPS noise makes that estimate wander by tens of centimetres in simulation (and
metres on a real drone), so scans of the same wall land in different places;
merged, they widen every roof edge and make false steps. Here each scan is
shifted onto what the earlier scans built (point-to-plane ICP, translation
only): ground and roofs pin the height and the roof-slope directions, walls
the horizontal position.
每次扫描用当时的位置估计转换，而位置估计会随 GPS 噪声漂移。这里用点到平面的
ICP（只求平移）把每次扫描对齐到之前已对齐的扫描：地面和屋顶约束高度和坡面方向，
墙面约束水平位置。

Translation only: heading errors are small (well under a degree) next to
position errors. 只做平移：朝向误差远小于位置误差。
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Registration:
    shifts: List[Tuple[float, float, float]]   # (north, east, up) applied to each scan, m
    scores: List[float]                        # fraction of points that matched a plane


def _voxel_keys(p: np.ndarray, res: float) -> np.ndarray:
    v = np.floor(p / res).astype(np.int64) + (1 << 20)
    return (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]


def _downsample(p: np.ndarray, res: float) -> np.ndarray:
    """Keep one point (the mean) per voxel."""
    keys = _voxel_keys(p, res)
    order = np.argsort(keys)
    _, start = np.unique(keys[order], return_index=True)
    sums = np.add.reduceat(p[order], start, axis=0)
    counts = np.diff(np.append(start, len(order)))[:, None]
    return sums / counts


class PlaneModel:
    """Per voxel of the reference cloud: centre and plane normal (where it is planar)."""

    def __init__(self, p: np.ndarray, res: float, min_points: int = 6) -> None:
        keys = _voxel_keys(p, res)
        order = np.argsort(keys)
        k, start = np.unique(keys[order], return_index=True)
        counts = np.diff(np.append(start, len(order)))
        keep_keys, centres, normals = [], [], []
        ps = p[order]
        for key, s, n in zip(k, start, counts):
            if n < min_points:
                continue
            q = ps[s:s + n]
            c = q.mean(0)
            w, v = np.linalg.eigh(np.cov((q - c).T))
            if w[0] > 0.2 * w[1]:          # not planar enough (a corner, clutter)
                continue
            keep_keys.append(key)
            centres.append(c)
            normals.append(v[:, 0])
        self.keys = np.array(keep_keys, dtype=np.int64)
        self.centres = np.array(centres).reshape(-1, 3)
        self.normals = np.array(normals).reshape(-1, 3)
        self.res = res

    def lookup(self, p: np.ndarray) -> np.ndarray:
        """Return the index of the plane in each point's voxel, or -1."""
        if not len(self.keys):
            return np.full(len(p), -1)
        keys = _voxel_keys(p, self.res)
        i = np.clip(np.searchsorted(self.keys, keys), 0, len(self.keys) - 1)
        return np.where(self.keys[i] == keys, i, -1)


def align(src: np.ndarray, model: PlaneModel, reach: float = 1.0, iterations: int = 8,
          damping: float = 20.0, max_residual: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Find the translation that moves src onto the model's planes (point to plane).

    Directions the planes do not pin down (along a single wall) are held near
    zero by damping. Returns (shift, fraction of points matched).
    约束不足的方向（例如沿着唯一一面墙）由阻尼项保持在零附近。
    """
    t = np.zeros(3)
    frac = 0.0
    for _ in range(iterations):
        q = src + t
        idx = model.lookup(q)
        ok = idx >= 0
        if ok.sum() < 30:
            return np.zeros(3), 0.0
        n = model.normals[idx[ok]]
        r = np.einsum('ij,ij->i', n, q[ok] - model.centres[idx[ok]])
        good = np.abs(r) < max_residual
        if good.sum() < 30:
            return np.zeros(3), 0.0
        a, b = n[good], -r[good]
        step = np.linalg.solve(a.T @ a + damping * np.eye(3), a.T @ b)
        t = np.clip(t + step, -reach, reach)
        frac = good.sum() / len(src)
        if np.linalg.norm(step) < 0.005:
            break
    return t, float(frac)


def register_scans(clouds: Sequence[np.ndarray], voxel: float = 0.3, min_match: float = 0.15,
                   coarse: Tuple[float, ...] = (1.0, 0.6)
                   ) -> Tuple[List[np.ndarray], Registration]:
    """
    Shift every scan onto the ones before it; return the shifted scans.

    Scans that match too little stay where they are. Finally every shift is
    offset by the median shift, so the merged map stays in the frame the
    vehicle is flying in rather than in the first scan's.
    依次把每次扫描对齐到之前已对齐的扫描；匹配太少的不动。最后减去平移量的中位数，
    使合并后的地图仍处在飞机正在使用的坐标系中。
    """
    clouds = [np.asarray(c, dtype=np.float64) for c in clouds]
    shifts, scores = [], []
    ref: List[np.ndarray] = []
    for c in clouds:
        if not len(c):
            shifts.append(np.zeros(3))
            scores.append(0.0)
            continue
        small = _downsample(c, voxel / 2)
        t, frac = np.zeros(3), 0.0
        if ref:
            # Coarse to fine: large voxels catch errors of half a metre (walls,
            # roofs and ground are still planes at that size), small ones refine.
            # 由粗到细：大体素能抓住半米的误差，小体素再细调。
            allref = np.vstack(ref)
            for res in coarse + (voxel,):
                step, frac = align(small + t, PlaneModel(allref, res), reach=1.0)
                t = t + step
            if frac < min_match:
                t = np.zeros(3)
        shifts.append(t)
        scores.append(frac)
        ref.append(small + t)
    med = np.median(np.array(shifts), axis=0) if shifts else np.zeros(3)
    shifts = [s - med for s in shifts]
    out = [c + s for c, s in zip(clouds, shifts)]
    return out, Registration([tuple(round(float(v), 3) for v in s) for s in shifts],
                             [round(float(v), 3) for v in scores])


def merge_registered(clouds: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    clouds = [c for c in clouds if c is not None and len(c)]
    if not clouds:
        return None
    aligned, _ = register_scans(clouds)
    return np.vstack(aligned)
