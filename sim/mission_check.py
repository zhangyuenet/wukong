#!/usr/bin/env python3
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Run the M3 mission logic end to end without Gazebo: discovery, survey, inspection plan.

    python3 sim/mission_check.py two_storey row_end

The real HouseTask code runs; the vehicle is ideal (it reaches every setpoint)
and depth scans are ray-cast from the generated world. Decide answers are
inserted exactly as Mission does: only if the rest of the plan passes
check_plan() from where the vehicle is. Then the eaves found and the photos
planned are compared with the ground truth.
不用 Gazebo 跑通 M3 的任务逻辑：发现、分层扫描、巡检规划。飞机按理想情况到达每个设定点，
深度图由光线投射生成；Decide 的结果与 Mission 一样，先通过检查才插入。
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402
import numpy as np  # noqa: E402
from perception_check import compare, K  # noqa: E402
import raycast  # noqa: E402

from wukong_flight.skills import (check_plan, Decide, GoTo, ReturnHome, Scan,  # noqa: E402
                                  Takeoff, TakePhoto)
from wukong_mission.classify import TruthClassifier  # noqa: E402
from wukong_mission.gutter_task import GutterRequest  # noqa: E402
from wukong_mission.house_task import HouseTask, OVERVIEW  # noqa: E402
from wukong_mission.judge import FakeJudge  # noqa: E402
from wukong_mission.world import load_world  # noqa: E402
from wukong_perception.geometry import depth_rays, depth_to_points  # noqa: E402


class RayCastDepth:
    """Stands in for DepthCamera: renders the generated world instead of reading ROS."""

    def __init__(self, tris, drift: float = 0.0, seed: int = 0):
        self.tris = tris
        self.clouds, self.labels, self.poses = [], [], []
        # Position-estimate error, a random walk like GPS noise: the image is
        # rendered from the true pose, converted with the estimated one.
        # 模拟位置估计漂移：用真实位姿渲染，用带误差的位姿转换。
        self.drift = drift
        self.rng = np.random.default_rng(seed)
        self.err = np.zeros(3)

    def capture(self, label, pose, done):
        depth = raycast.render_depth(self.tris, pose, stride=4)
        if self.drift:
            self.err = np.clip(self.err + self.rng.normal(0, self.drift / 3, 3),
                               -self.drift, self.drift)
            self.err[2] *= 0.5
        est = (pose[0] + self.err[0], pose[1] + self.err[1], pose[2] + self.err[2]) + \
            tuple(pose[3:])
        self.clouds.append(depth_to_points(depth, K, est, stride=4))
        self.rays = getattr(self, 'rays', []) + [depth_rays(depth, K, est, stride=4)]
        self.errors = getattr(self, 'errors', []) + [tuple(self.err)]
        self.labels.append(label)
        self.poses.append(tuple(pose) + (0.0,) * (6 - len(pose)))
        done(True)


CLEARANCE = 1.0   # m: closer than this to a real building counts as a collision


def _seg_dist(p, a, b):
    """Distances from points p (P,3) to segments a->b (T,3): (P,T)."""
    ab = b - a
    t = np.einsum('ptk,tk->pt', p[:, None, :] - a[None], ab) / np.maximum(
        np.einsum('tk,tk->t', ab, ab), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=2)


def mesh_distance(points, tris):
    """Distance from each point (P,3) to the nearest triangle of tris (T,3,3)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rel = points[:, None, :] - a[None]
    h = np.einsum('ptk,tk->pt', rel, n)                       # signed height over the plane
    q = points[:, None, :] - h[..., None] * n[None]          # projection onto the plane
    inside = np.ones(h.shape, bool)
    for u, v in ((a, b), (b, c), (c, a)):
        side = np.einsum('ptk,tk->pt', np.cross(v - u, q - u[None]), n)
        inside &= side >= -1e-9
    d = np.where(inside, np.abs(h), np.inf)
    for u, v in ((a, b), (b, c), (c, a)):
        d = np.minimum(d, _seg_dist(points, u, v))
    return d.min(axis=1)


def _too_close(a, b, tris):
    """Where the leg a -> b comes within CLEARANCE of the real world, or None."""
    if tris is None or len(tris) == 0:
        return None
    n = max(1, int(math.dist(a, b) / 0.25))
    k = np.linspace(0.0, 1.0, n + 1)[:, None]
    ned = np.array(a) + k * (np.array(b) - np.array(a))       # (north, east, altitude)
    enu = ned[:, [1, 0, 2]]
    d = mesh_distance(enu, tris)
    i = int(np.argmin(d))
    return (tuple(ned[i]), float(d[i])) if d[i] < CLEARANCE else None


def fly(task, plan, start, limits, tris=None):
    """Execute a plan with an ideal vehicle; returns (photos, problems, scans)."""
    pos = (start[0], start[1], 0.0)
    heading = 0.0
    photos, problems, scans = [], [], 0
    i = 0
    while i < len(plan):
        s = plan[i]
        if isinstance(s, Takeoff):
            pos = (start[0], start[1], s.altitude)
        elif isinstance(s, GoTo):
            new = (s.north, s.east, s.altitude)
            hit = _too_close(pos, new, tris)
            if hit:
                (n, e, z), d = hit
                problems.append(f'skill {i + 1}: leg to N {new[0]:.1f} E {new[1]:.1f} alt '
                                f'{new[2]:.1f} passes {d:.1f} m from the house at '
                                f'N {n:.1f} E {e:.1f} alt {z:.1f}')
            pos = new
            if s.look_at is not None:
                heading = math.atan2(s.look_at[1] - s.east, s.look_at[0] - s.north)
        elif isinstance(s, Scan):
            task.depth.capture(s.label, (*pos, heading), lambda ok: None)
            scans += 1
        elif isinstance(s, TakePhoto):
            if s.label == OVERVIEW:
                task.on_photo(s.label, '', (*pos, heading, 0.0, 0.0))
            else:
                photos.append(s.label)
        elif isinstance(s, Decide):
            t0 = time.time()
            skills, new_limits, note = task.on_decide(s.label, (*pos, heading))
            dt = time.time() - t0
            print(f'   decide "{s.label}" ({dt:.1f}s): {note}')
            if skills:
                lim = new_limits or limits
                # Same rule as Mission: skills that end the mission replace the rest.
                terminal = isinstance(skills[-1], (ReturnHome,))
                rest = list(skills) if terminal else list(skills) + plan[i + 1:]
                check = check_plan(rest, lim, start, initial=pos)
                if check.ok:
                    plan = plan[:i + 1] + rest
                    limits = lim
                else:
                    problems += [f'decide {s.label} rejected: {e}' for e in check.errors[:3]]
        elif isinstance(s, ReturnHome):
            break
        i += 1
    return photos, problems, scans


def planned_report(task):
    """
    Return the planned photos as a report: each claims what the plan says is the target's.

    For a townhouse unit that is the part between the party walls as read (the
    photos run on past them so the boundary shows; that part counts only if a
    photo shows the boundary there, which needs the real judge). Scored with
    sim/evaluate.py's rules for gaps and intrusion; the answers mean nothing.
    把计划拍摄的照片当成报告：每段只声明计划认定属于目标户的部分，按 evaluate.py 的
    漏拍和越界规则检查（判断结果无意义）。
    """
    photos = []
    for t in task.targets.values():
        a, b = t.claim if t.claim is not None else (t.start, t.end)
        if math.dist(a, b) < 0.05:
            continue                     # wholly past a party wall: the neighbour's
        photos.append({'needs_attention': 'no', 'findings': '',
                       'extra': {'gutter': t.gutter, 'segment': t.segment,
                                 'from': list(a), 'to': list(b),
                                 'height': t.height - task_ground(task)}})
    return {'flight_ok': True, 'photos': photos}


def task_ground(task):
    return getattr(task, '_ground', 0.0)


def check(name, verbose=False, drift=0.0, register=False, classify=True):
    gen = os.path.join(HERE, 'generated', name)
    truth = json.load(open(os.path.join(gen, 'truth.json')))
    world = load_world(os.path.join(gen, 'map.yaml'))
    if not world.find_house:      # an M2 map: run the M3 flow on it anyway
        import dataclasses
        world = dataclasses.replace(world, rough_houses=(), search_radius=30.0,
                                    limits=dataclasses.replace(world.limits, no_fly=()))
    tris = raycast.load_triangles(os.path.join(gen, f'wukong_{name}.sdf'))
    out = tempfile.mkdtemp(prefix=f'mission_{name}_')
    start = (0.0, 0.0)
    task = HouseTask(world, GutterRequest('all gutters', 'nearest', []), start, out,
                     FakeJudge(), 'check all gutters', log=lambda m: print('  ', m))
    task.depth = RayCastDepth(tris, drift)
    if classify and 'buildings' in truth:
        task.classifier = TruthClassifier(truth)
    task.register = register
    plan = task.plan()
    print(f'== {name}')
    t0 = time.time()
    solid = raycast.load_triangles(os.path.join(gen, f'wukong_{name}.sdf'), skip_debris=True)
    photos, problems, scans = fly(task, plan, start, world.limits, solid)
    eaves_json = os.path.join(out, 'eaves.json')
    task._ground = json.load(open(eaves_json))['ground_level'] if os.path.exists(eaves_json) else 0
    print(f'   {scans} scans, {len(photos)} photos planned, {len(task.skipped)} skipped, '
          f'{time.time() - t0:.0f}s')
    for p in problems:
        print('   PROBLEM', p)
    target = [g for g in truth['gutters'] if g.get('target', True)]
    rows, extra = compare(task.eaves, target)
    missed = [r for r in rows if r[1] == 'MISSED']
    good = not problems and not missed
    if getattr(task, 'refusal', None):
        print(f'   not inspected: {task.refusal}')
    _, scored, _ = evaluate.evaluate(planned_report(task), truth)
    for p in scored:
        print('  ', p)
        good = False
    for r in missed:
        print('   MISSED eave', r[0])
    for e in extra:
        print(f'   EXTRA eave {e.name}: {e.start} -> {e.end} at {e.height:.2f} m')
        good = False
    print('   ' + ('OK' if good else 'NOT OK'))
    return good


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('scenarios', nargs='+')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--drift', type=float, default=0.0,
                    help='position-estimate error in m (random walk, like GPS noise)')
    ap.add_argument('--register', action='store_true',
                    help='align scans to each other before merging (experimental, off in flight)')
    ap.add_argument('--no-classify', action='store_true',
                    help='skip the house-type judgement (the drone must then refuse to inspect)')
    args = ap.parse_args(argv)
    # Every scenario is checked, also after one fails. 一个场景失败后其余的照样检查。
    results = [check(n, args.verbose, args.drift, args.register, not args.no_classify)
               for n in args.scenarios]
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
