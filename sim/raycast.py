#!/usr/bin/env python3
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
A tiny depth camera for generated worlds: ray casting in numpy, no Gazebo.

Reads the static models of a world SDF (boxes and OBJ meshes, as sim/scenario.py
writes them), and renders depth images like the drone's OAK-D sees them,
occlusion included. For developing and unit-testing perception in seconds;
the real simulator still has the final word.
用 numpy 光线投射渲染深度图，不需要 Gazebo；用于快速开发和测试感知算法。

    python3 sim/raycast.py sim/generated/two_storey/wukong_two_storey.sdf --survey out.npz
"""

import argparse
import math
import re
import sys
from typing import List, Tuple
import xml.etree.ElementTree as ET

import numpy as np

# OAK-D Lite depth in Gazebo: 640x480, hfov 1.274 rad, 0.2..19.1 m.
DEPTH_W, DEPTH_H, DEPTH_HFOV, DEPTH_FAR = 640, 480, 1.274, 19.1
CAMERA_OFFSET_FRD = (0.132, 0.0, -0.26)

_BOX_TRIS = np.array([  # unit cube corners (+-0.5), 12 triangles
    [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
    [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
_BOX_V = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def load_triangles(world_sdf: str, skip_debris: bool = False) -> np.ndarray:
    """All visual triangles of static models in ENU world coordinates: (T, 3, 3)."""
    root = ET.parse(world_sdf).getroot()
    tris: List[np.ndarray] = []
    for model in root.iter('model'):
        if model.findtext('static', '').strip() != 'true':
            continue
        if model.get('name') == 'ground_plane':
            continue
        mpose = np.array([float(v) for v in (model.findtext('pose') or '0 0 0 0 0 0').split()])
        mr = _rot(*mpose[3:])
        for vis in model.iter('visual'):
            if skip_debris and vis.get('name', '').startswith('debris'):
                continue
            pose = np.array([float(v) for v in (vis.findtext('pose') or '0 0 0 0 0 0').split()])
            r = _rot(*pose[3:])
            box = vis.find('geometry/box/size')
            mesh = vis.find('geometry/mesh/uri')
            if box is not None:
                size = np.array([float(v) for v in box.text.split()])
                v = (_BOX_V * size) @ r.T + pose[:3]
                t = v[_BOX_TRIS]
            elif mesh is not None:
                path = re.sub(r'^file://', '', mesh.text.strip())
                verts, faces = [], []
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('v '):
                            verts.append([float(x) for x in line.split()[1:4]])
                        elif line.startswith('f '):
                            faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
                v = np.array(verts) @ r.T + pose[:3]
                t = v[np.array(faces)]
            else:
                continue
            tris.append(t @ mr.T + mpose[:3])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def render_depth(tris: np.ndarray, pose, stride: int = 4,
                 width: int = DEPTH_W, height: int = DEPTH_H, hfov: float = DEPTH_HFOV,
                 far: float = DEPTH_FAR) -> np.ndarray:
    """
    Depth image (metres along the optical axis, inf = nothing) seen from pose.

    pose: (north, east, height, heading[, roll, pitch]) of the vehicle, as the
    drone's perception uses it. Only every `stride`-th pixel is cast; the result
    is full size with the other pixels set to inf (depth_to_points with the same
    stride reads exactly the cast pixels). Ground (z = 0) is included.
    """
    f = (width / 2) / math.tan(hfov / 2)
    n0, e0, h0, yaw = pose[:4]
    roll, pitch = (pose[4], pose[5]) if len(pose) >= 6 else (0.0, 0.0)
    r_ned = _rot(roll, pitch, yaw)                      # body FRD -> NED
    cam_ned = np.array([n0, e0, -h0]) + r_ned @ np.array(CAMERA_OFFSET_FRD)
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    body = np.stack([np.ones_like(u, float), (u - width / 2) / f, (v - height / 2) / f], -1)
    d_ned = body.reshape(-1, 3) @ r_ned.T              # ray directions, z-depth = 1 per unit
    # NED -> ENU for ray casting against the world triangles.
    o = np.array([cam_ned[1], cam_ned[0], -cam_ned[2]])
    d = np.column_stack([d_ned[:, 1], d_ned[:, 0], -d_ned[:, 2]])
    best = np.full(len(d), np.inf)
    # Ground plane z = 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = -o[2] / d[:, 2]
    best = np.where((tg > 0), tg, best)
    # Moller-Trumbore, chunked over triangles. 分块计算，控制内存。
    for i in range(0, len(tris), 256):
        t = tris[i:i + 256]
        e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        p = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum('rtk,tk->rt', p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - t[:, 0]
        uu = np.einsum('rtk,tk->rt', p, s) * inv
        q = np.cross(s, e1)
        vv = (d @ q.T) * inv
        tt = (e2 * q).sum(1)[None, :] * inv
        hit = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 1e-6)
        tt = np.where(hit, tt, np.inf).min(1)
        best = np.minimum(best, tt)
    best[(best > far) | (best < 0.2)] = np.inf
    out = np.full((height, width), np.inf, np.float32)
    out[::stride, ::stride] = best.reshape(v.shape).astype(np.float32)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('world')
    ap.add_argument('--pose', nargs='+', type=float, default=[0, 0, 5, 0],
                    help='north east height heading_deg')
    ap.add_argument('--png', default=None, help='save a depth picture')
    args = ap.parse_args(argv)
    tris = load_triangles(args.world)
    p = list(args.pose)
    p[3] = math.radians(p[3])
    import time
    t0 = time.time()
    depth = render_depth(tris, p, stride=4)
    print(f'{len(tris)} triangles, rendered in {time.time() - t0:.2f}s, '
          f'{np.isfinite(depth).sum()} hits')
    if args.png:
        from PIL import Image
        d = depth[::4, ::4]
        img = np.where(np.isfinite(d), 255 - np.clip(d / DEPTH_FAR * 255, 0, 255), 0)
        Image.fromarray(img.astype(np.uint8)).resize((640, 480)).save(args.png)
    return 0


if __name__ == '__main__':
    sys.exit(main())
