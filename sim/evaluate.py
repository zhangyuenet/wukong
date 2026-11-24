#!/usr/bin/env python3
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Score a gutter inspection report against the scenario's ground truth.

    python3 sim/evaluate.py log/missions/<run>/report.json sim/generated/<scenario>/truth.json
    python3 sim/evaluate.py REPORT TRUTH --poses truth_poses.jsonl

Three things are checked, separately:

  1. Judgements. The drone splits each gutter into segments itself, so its
     boundaries do not line up with the scenario's. Each reported segment is
     scored by where the debris really is, on every gutter the segment shows
     (a neighbour's included: whether it should have been photographed at all
     is check 3):
       - debris inside the segment, more than TOLERANCE from both ends
             -> the report must say "needs attention";
       - no debris within TOLERANCE of the segment -> it must say it is fine;
       - debris only within TOLERANCE of an end -> either answer is accepted.
     "unclear", unjudged and uninspected segments count as wrong. Segments
     the drone itself set aside as a neighbour's make no claim and are not
     scored, but they do not count as photographed either.
  2. Nothing missed: no stretch of a target gutter (its visible part) longer
     than MAX_GAP goes unphotographed, and every piece of debris on it lies in
     a photographed stretch.
  3. Nothing else: at most MAX_INTRUSION of any gutter that is not the
     target's (a neighbour's, another building's) is photographed and judged.

What a photo showed is, by default, the stretch the report claims for it.
With --poses (the vehicle's true poses, recorded in simulation by
sim/truth_pose.py) it is recomputed: the part of each gutter that really falls
inside the judged part of the picture, seen from where the vehicle really was.
Position drift then shows up as gaps and intrusion instead of being taken on
trust. 默认按报告声明的范围计算；给出 --poses（仿真记录的真实位姿）时，按真实位姿重新
计算照片判断区域里实际看到的是哪一段水槽，位置漂移会表现为漏拍或越界。

Exit code 0 only if everything is right.
"""

import argparse
import bisect
import json
import math
import os
import sys

TOLERANCE = 0.3        # m around segment ends, for debris
LINE_DISTANCE = 1.0    # m: a claimed stretch lies along a gutter within this of its line
HEIGHT_DISTANCE = 1.2  # m: gutters on different floors can share a line seen from above
MAX_GAP = 0.5          # m: longest stretch of a target gutter that may go unphotographed
MAX_INTRUSION = 0.5    # m: most of any other gutter that may be photographed and judged
POSE_MAX_AGE = 1.5     # s between a photo and the nearest recorded true pose
STEP = 0.05            # m between the points sampled along a gutter
# What a camera above the gutter sees is its outer lip (and debris sticking up
# above it); the truth's height is the bottom plate, 0.13 m lower and hidden
# behind the lip (sim/scenario.py). 从上方看到的是水槽外沿（和高出外沿的杂物）；
# 真值高度是底板，比外沿低 0.13 米，被外沿挡住。
LIP = 0.13
NEAREST_BAND = 1.5     # m: gutters this much further than the nearest one are not judged
NOT_INSPECTED = ('skipped', 'neighbour')


def _param(p, a, b):
    """Position of p along the line a -> b, in metres from a; and distance from the line."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    rx, ry = p[0] - a[0], p[1] - a[1]
    return rx * ux + ry * uy, abs(-rx * uy + ry * ux)


def _length(g):
    return math.dist(g['from'], g['to'])


def _overlap(lo, hi, intervals):
    return sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in intervals)


def _merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _gaps(a, b, covered):
    """Uncovered stretches of a..b, longest first."""
    gaps, pos = [], a
    for lo, hi in _merged(covered):
        if hi <= pos:
            continue
        if lo > pos:
            gaps.append((pos, min(lo, b)))
        pos = max(pos, hi)
        if pos >= b:
            break
    if pos < b:
        gaps.append((pos, b))
    return sorted([g for g in gaps if g[1] - g[0] > 1e-6], key=lambda g: g[0] - g[1])


def on_line(x, g):
    """Whether the stretch x claims lies along gutter g (same line, same floor, same direction)."""
    if 'height' in x and abs(x['height'] - g['height']) > HEIGHT_DISTANCE:
        return False
    s0, d0 = _param(x['from'], g['from'], g['to'])
    s1, d1 = _param(x['to'], g['from'], g['to'])
    if max(d0, d1) > LINE_DISTANCE:
        return False
    # A short stretch near a corner is close to both lines: compare directions.
    # 转角附近的短段离两条线都近：再比较方向。
    claimed = math.dist(x['from'], x['to'])
    return claimed < 0.5 or abs(s1 - s0) > 0.9 * claimed


def claimed(x, gutters):
    """[(gutter, lo, hi)]: the stretch the report claims, on every gutter it lies along."""
    out = []
    for g in gutters:
        if not on_line(x, g):
            continue
        s0 = _param(x['from'], g['from'], g['to'])[0]
        s1 = _param(x['to'], g['from'], g['to'])[0]
        lo, hi = max(0.0, min(s0, s1)), min(_length(g), max(s0, s1))
        if hi - lo > 1e-3:
            out.append((g, lo, hi))
    return out


class TruePoses:
    """The vehicle's true poses recorded by sim/truth_pose.py, looked up by wall time."""

    def __init__(self, path):
        rows = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: r['t'])
        self.t = [r['t'] for r in rows]
        self.rows = rows

    def at(self, t):
        """(north, east, height, heading, roll, pitch) nearest to t, or None if none is close."""
        if not self.t or t is None:
            return None
        i = bisect.bisect_left(self.t, t)
        near = [k for k in (i - 1, i) if 0 <= k < len(self.t)]
        k = min(near, key=lambda k: abs(self.t[k] - t))
        if abs(self.t[k] - t) > POSE_MAX_AGE:
            return None
        r = self.rows[k]
        return (r['n'], r['e'], r['alt'], r['heading'], r['roll'], r['pitch'])


def _parallel(x, g):
    """Whether gutter g runs the same way as the stretch x claims, on the same floor."""
    if 'height' in x and abs(x['height'] - g['height']) > HEIGHT_DISTANCE:
        return False
    cx, cy = x['to'][0] - x['from'][0], x['to'][1] - x['from'][1]
    gx, gy = g['to'][0] - g['from'][0], g['to'][1] - g['from'][1]
    n = math.hypot(cx, cy) * math.hypot(gx, gy)
    return n < 1e-9 or abs(cx * gx + cy * gy) / n > 0.9


def seen(x, gutters, pose):
    """
    [(gutter, lo, hi)]: what the judged part of the picture really showed.

    Samples the top of the outer lip of every gutter running the way the
    claimed stretch runs (same floor), projects the samples from the true pose
    and keeps those inside the judged columns and the crop's rows. Of those,
    only the nearest gutter (and any in line with it) counts: the judge is told
    to judge the gutter right in front and ignore ones further away. Not tied
    to the claimed line: a drifted position estimate puts the claim off the
    gutter the camera really looked at (1.5 m in Gazebo).
    按真实位姿投影与声明方向平行、同一楼层的各水槽外沿顶部的采样点，保留落在判断区域内
    的部分；只算最近的那条（以及与它在同一条线上的）。不限于声明的那条线：位置估计漂移时，
    声明的位置会偏离相机实际拍到的水槽（Gazebo 中偏过 1.5 米）。
    """
    import numpy as np
    from wukong_perception.geometry import project, RGB_X500_DEPTH
    c0, c1 = x['columns']
    r0, r1 = x['rows']
    found = []
    for g in gutters:
        if not _parallel(x, g):
            continue
        length = _length(g)
        s = np.linspace(0.0, length, max(2, int(length / STEP) + 1))
        a, b = np.array(g['from'], float), np.array(g['to'], float)
        xy = a + np.outer(s / length, b - a)
        pts = np.column_stack([xy, np.full(len(s), float(g['height']) + LIP)])
        u, v, d = project(pts, RGB_X500_DEPTH, pose).T
        ok = (d > 0.5) & (u >= c0) & (u <= c1) & (v >= r0) & (v <= r1)
        if ok.any():
            found.append((float(np.median(d[ok])), g, s, ok))
    if not found:
        return []
    nearest = min(f[0] for f in found)
    out = []
    for depth, g, s, ok in found:
        if depth > nearest + NEAREST_BAND:
            continue                       # a gutter further back, not the one judged
        # Runs of consecutive samples. 连续的采样点构成一段。
        k = 0
        while k < len(s):
            if not ok[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(s) and ok[j + 1]:
                j += 1
            out.append((g, float(s[k]), float(s[j])))
            k = j + 1
    return out


def evaluate(report: dict, truth: dict, poses: 'TruePoses' = None):
    """Return (rows, problems, note): one row per judged segment, and what failed."""
    rows, problems = [], []
    gutters = truth['gutters']
    shown = {g['name']: [] for g in gutters}    # judged stretches, per truth gutter
    verified = unverified = 0
    for p in report.get('photos', []):
        x = p.get('extra', {})
        if not x.get('from'):
            continue
        said = p['needs_attention']
        if poses is not None and x.get('columns') and x.get('rows'):
            pose = poses.at(x.get('t'))
        else:
            pose = None
        if pose is not None:
            parts = seen(x, gutters, pose)
            verified += 1
        else:
            parts = claimed(x, gutters)
            unverified += said not in NOT_INSPECTED
        if said == 'neighbour':
            # Set aside by the drone itself: no claim, nothing photographed.
            # 无人机自己判定为邻居家：不评分，也不算拍到。
            rows.append((x['gutter'], x['segment'], '(not ours)', said, True, ''))
            continue
        if said != 'skipped':
            for g, lo, hi in parts:
                shown[g['name']].append((lo, hi))
        if not parts:
            if any(on_line(x, g) for g in gutters) or pose is not None:
                why = 'the judged part of the picture shows no gutter'
            else:
                why = 'not on any gutter of the scenario'
            rows.append((x['gutter'], x['segment'], '-', said, False, why))
            continue
        inner = near = 0.0
        for g, lo, hi in parts:
            debris = [tuple(sorted((_param(d['from'], g['from'], g['to'])[0],
                                    _param(d['to'], g['from'], g['to'])[0])))
                      for d in g.get('debris', [])]
            inner += _overlap(lo + TOLERANCE, hi - TOLERANCE, debris)
            near += _overlap(lo - TOLERANCE, hi + TOLERANCE, debris)
        expect = 'yes' if inner > 0.0 else ('no' if near == 0.0 else 'either')
        ok = said in ('yes', 'no') and (expect == 'either' or said == expect)
        # Named after the target gutter it shows most of (else whatever it shows most of).
        # 以它拍到最多的目标水槽命名，没有目标水槽时以拍到最多的水槽命名。
        main = max(parts, key=lambda q: (q[0].get('target', True), q[2] - q[1]))[0]
        others = sorted({g['name'] for g, _, _ in parts} - {main['name']})
        why = p.get('findings', '') + (f' [also shows {", ".join(others)}]' if others else '')
        rows.append((main['name'], x['segment'], f'{expect} ({inner:.2f} m inside)', said, ok,
                     why))

    for g in gutters:
        if g.get('target', True):
            # Nothing missed: every piece of debris, and no long unphotographed stretch.
            # 不漏：每处杂物都拍到，也没有超过 MAX_GAP 的漏拍。
            for d in g.get('debris', []):
                a = _param(d['from'], g['from'], g['to'])[0]
                b = _param(d['to'], g['from'], g['to'])[0]
                lo, hi = min(a, b), max(a, b)
                if _overlap(lo, hi, _merged(shown[g['name']])) < 0.5 * (hi - lo):
                    problems.append(f'MISSED: {g["name"]}: debris at {lo:.1f}..{hi:.1f} m '
                                    f'(scenario segment {d["segment"]}) was not photographed')
            ends = g.get('visible') or [g['from'], g['to']]
            a, b = sorted(_param(q, g['from'], g['to'])[0] for q in ends)
            gaps = _gaps(a, b, shown[g['name']])
            if gaps and gaps[0][1] - gaps[0][0] > MAX_GAP:
                total = sum(hi - lo for lo, hi in gaps)
                lo, hi = gaps[0]
                problems.append(f'MISSED: {g["name"]}: {hi - lo:.2f} m at {lo:.1f}..{hi:.1f} m '
                                f'not photographed ({total:.2f} m of {b - a:.1f} m in all; '
                                f'limit {MAX_GAP:g} m)')
        else:
            # Nothing else: not more than a little of anyone else's gutter.
            # 不越界：别人家的水槽最多只能带进一点。
            took = sum(hi - lo for lo, hi in _merged(shown[g['name']]))
            if took > MAX_INTRUSION:
                problems.append(f'INTRUDED: {g["name"]} (not the target): {took:.2f} m '
                                f'photographed and judged (limit {MAX_INTRUSION:g} m)')
    if not report.get('flight_ok', True):
        problems.append(f'FLIGHT: did not complete: {report.get("flight_note") or "?"}')
    if poses is None:
        note = 'what the photos showed: as claimed by the report (no true poses given)'
    else:
        note = (f'what the photos showed: checked with true poses for {verified} photo(s)'
                + (f', {unverified} taken on trust (no pose or crop recorded)'
                   if unverified else ''))
    return rows, problems, note


def _find_poses(report_path):
    """truth_poses.jsonl next to the report or one directory up (eval runs), if any."""
    d = os.path.dirname(os.path.abspath(report_path))
    for c in (os.path.join(d, 'truth_poses.jsonl'),
              os.path.join(os.path.dirname(d), 'truth_poses.jsonl')):
        if os.path.isfile(c):
            return c
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('report')
    ap.add_argument('truth')
    ap.add_argument('--poses', help='true poses recorded by sim/truth_pose.py (default: '
                                    'truth_poses.jsonl next to the report or one level up)')
    args = ap.parse_args(argv)
    with open(args.report, encoding='utf-8') as f:
        report = json.load(f)
    with open(args.truth, encoding='utf-8') as f:
        truth = json.load(f)
    path = args.poses or _find_poses(args.report)
    poses = TruePoses(path) if path else None
    rows, problems, note = evaluate(report, truth, poses)
    print(f'{"gutter":<20} {"seg":>3}  {"expected":<22} {"report":<9} result')
    for name, i, expect, said, ok, why in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f'{name:<20} {i:>3}  {expect:<22} {said:<9} {"OK" if ok else "WRONG"}'
              + ('' if ok else f'  {why[:90]}'))
    for p in problems:
        print(p)
    print(note)
    scored = [r for r in rows if r[3] != 'neighbour']
    right = sum(r[4] for r in scored)
    good = right == len(scored) and scored and not problems
    print(f'\n{right}/{len(scored)} reported segments right, {len(problems)} problem(s)'
          + ('' if good else ' -- report does NOT match the scenario'))
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
