#!/usr/bin/env python3
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Check eave detection on a scenario without Gazebo: ray-cast a survey, compare with truth.

    python3 sim/perception_check.py two_storey row_end
    python3 sim/perception_check.py --png /tmp/out two_storey    # also save top-down plots

Survey stations are placed around the target from the ground truth (this checks
perception only, not the discovery of the target). Needs the workspace sourced
(wukong_perception) and the scenario generated (sim/generated/<name>/).
不用 Gazebo 检查屋檐识别：光线投射模拟环绕扫描，与真值比较。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raycast  # noqa: E402

from wukong_perception.geometry import depth_to_points, intrinsics_from_fov  # noqa: E402
from wukong_perception.heightmap import (buildings, edge_cells, eave_lines,  # noqa: E402
                                         HeightMap)

K = intrinsics_from_fov(640, 480, raycast.DEPTH_HFOV)


def survey_poses(center, radius, altitudes, stations=8):
    poses = []
    for alt in altitudes:
        for i in range(stations):
            b = 2 * math.pi * i / stations
            p = (center[0] + radius * math.cos(b), center[1] + radius * math.sin(b))
            heading = math.atan2(center[1] - p[1], center[0] - p[0])
            poses.append((p[0], p[1], alt, heading))
    return poses


def target_box(truth):
    if 'buildings' not in truth:          # M2 (v1) truth: one house
        h = truth['house']
        pts = np.array(h['footprint_with_overhang'])
        c = pts.mean(0)
        return (float(c[0]), float(c[1])), float(np.max(np.hypot(*(pts - c).T))), \
            h['ridge_height']
    pts = []
    for b in truth['buildings']:
        if b['target']:
            for f in b['footprints']:
                pts += f['corners']
            top = b['height']
    pts = np.array(pts)
    c = pts.mean(0)
    r = float(np.max(np.hypot(*(pts - c).T)))
    return (float(c[0]), float(c[1])), r, top


def _project(p, a, b):
    d = np.array(b) - np.array(a)
    L = np.linalg.norm(d)
    u = d / L
    r = np.array(p) - np.array(a)
    return float(r @ u), float(abs(-r[0] * u[1] + r[1] * u[0])), L


def compare(eaves, truth_gutters):
    """Match each truth gutter to the detected eave lying along it."""
    rows = []
    used = set()
    for g in truth_gutters:
        best = None
        for k, e in enumerate(eaves):
            s0, d0, L = _project(e.start, g['from'], g['to'])
            s1, d1, _ = _project(e.end, g['from'], g['to'])
            if max(d0, d1) > 0.8 or abs(e.height - g['height']) > 0.8:
                continue
            lo, hi = sorted((s0, s1))
            overlap = max(0.0, min(hi, L) - max(lo, 0.0))
            if overlap > 0.3 * L and (best is None or overlap > best[1]):
                best = (k, overlap, L, max(d0, d1), lo, hi, e)
        if best is None:
            rows.append((g['name'], 'MISSED', ''))
            continue
        k, overlap, L, dist, lo, hi, e = best
        used.add(k)
        rows.append((g['name'], 'found',
                     f'{e.name}: covers {overlap / L:4.0%}, ends off by {-lo:+.2f}/{hi - L:+.2f} m, '
                     f'{dist:.2f} m from the line, height {e.height - g["height"]:+.2f} m'))
    extra = [e for k, e in enumerate(eaves) if k not in used]
    return rows, extra


def check(name, png_dir=None):
    gen = os.path.join(HERE, 'generated', name)
    truth = json.load(open(os.path.join(gen, 'truth.json')))
    world = os.path.join(gen, f'wukong_{name}.sdf')
    tris = raycast.load_triangles(world)
    center, r, top = target_box(truth)
    poses = survey_poses(center, r + 5.5, (4.5, top + 2.5))
    t0 = time.time()
    clouds = [depth_to_points(raycast.render_depth(tris, p, stride=4), K, p, stride=4)
              for p in poses]
    pts = np.vstack(clouds)
    t1 = time.time()
    hm = HeightMap.from_points(pts, res=0.2)
    blds = buildings(hm, (0.0, 0.0))
    target = blds[0]
    cells = edge_cells(hm, target.cells)
    eaves = eave_lines(hm, cells)
    t2 = time.time()
    tg = [g for g in truth['gutters'] if g.get('target', True)]
    if truth.get('version') == 2 and any(b['kind'] == 'row' and b['target']
                                         for b in truth['buildings']):
        # Rows: the whole row is one building; compare with all its gutters for now.
        owner = tg[0]['owner'].split('/')[0]
        tg = [g for g in truth['gutters'] if g['owner'].startswith(owner + '/')]
    rows, extra = compare(eaves, tg)
    print(f'== {name}: {len(poses)} scans, {len(pts)} points ({t1 - t0:.1f}s), '
          f'{len(blds)} building(s), nearest {target.distance:.1f} m away, '
          f'{len(eaves)} eaves found ({t2 - t1:.1f}s); kinds: '
          + ', '.join(f'{k}={sum(c.kind == k for c in cells)}' for k in ('eave', 'rake', 'other')))
    for gname, status, detail in rows:
        print(f'   {gname:<28} {status:<7} {detail}')
    for e in extra:
        print(f'   EXTRA {e.name}: {e.start} -> {e.end} at {e.height:.2f} m')
    missed = sum(s == 'MISSED' for _, s, _ in rows)
    if os.environ.get('PLAN', '1') == '1' and not plan_check(hm, eaves, poses[0], tg):
        return False
    if png_dir:
        _plot(hm, target, cells, eaves, tg, os.path.join(png_dir, f'{name}_eaves.png'))
    return missed == 0 and not extra


def plan_check(hm, eaves, here_pose, truth_gutters):
    """Plan the inspection from the first survey station; check it like the drone would."""
    from wukong_flight.skills import check_plan, Limits, ReturnHome, TakePhoto
    from wukong_mission.inspection import plan_inspection, surfaces_from_heightmap
    hf = surfaces_from_heightmap(hm)
    limits = Limits(max_altitude=18, max_distance=45, max_skills=200, margin=2.0, no_fly=(hf,))
    here = (here_pose[0], here_pose[1], here_pose[2])
    t0 = time.time()
    skills, targets, skipped = plan_inspection(eaves, [hf], here, (0.0, 0.0), limits)
    t1 = time.time()
    check = check_plan(skills + [ReturnHome()], limits, (0.0, 0.0), initial=here)
    photos = sum(isinstance(s, TakePhoto) for s in skills)
    print(f'   plan: {len(skills)} skills, {photos} photos, {len(skipped)} skipped, '
          f'{t1 - t0:.1f}s; check {"OK" if check.ok else "FAILED"}')
    for sk in skipped:
        print(f'     skipped {sk.label}: {sk.reason}')
    for err in check.errors[:5]:
        print(f'     check: {err}')
    # Which viewpoints were not the standard one (4.5 m out, 0.6 m up)?
    for label, t in targets.items():
        e = next(e for e in eaves if e.name == t.gutter)
        mid = ((t.start[0] + t.end[0]) / 2, (t.start[1] + t.end[1]) / 2)
        out = math.dist(mid, t.viewpoint[:2])
        up = t.viewpoint[2] - e.height
        if abs(out - 4.5) > 0.05 or abs(up - 0.6) > 0.05:
            print(f'     adapted {label}: {out:.1f} m out, {up:.1f} m above the gutter')
    return check.ok


def _plot(hm, target, cells, eaves, truth_gutters, path):
    from PIL import Image, ImageDraw
    s = 4
    top = np.nan_to_num(hm.top, nan=0.0)
    g = np.clip(top / 10.0 * 200, 0, 200).astype(np.uint8)
    img = Image.fromarray(np.flipud(g)).convert('RGB').resize((g.shape[1] * s, g.shape[0] * s),
                                                               Image.NEAREST)
    d = ImageDraw.Draw(img)
    rows = hm.top.shape[0]

    def px(n, e):
        i = (n - hm.origin[0]) / hm.res
        j = (e - hm.origin[1]) / hm.res
        return j * s, (rows - i) * s
    colors = {'eave': (0, 200, 0), 'rake': (200, 0, 200), 'other': (200, 200, 0)}
    for c in cells:
        x, y = px(*hm.center(c.cell))
        d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=colors[c.kind])
    for gt in truth_gutters:
        d.line([px(*gt['from']), px(*gt['to'])], fill=(0, 120, 255), width=3)
    for e in eaves:
        d.line([px(*e.start), px(*e.end)], fill=(255, 0, 0), width=2)
    img.save(path)
    print(f'   plot: {path}  (green eave cells, magenta rake, blue truth, red found)')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('scenarios', nargs='+')
    ap.add_argument('--png', default=None)
    args = ap.parse_args(argv)
    ok = all([check(n, args.png) for n in args.scenarios])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
