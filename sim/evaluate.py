#!/usr/bin/env python3
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Score a gutter inspection report against the scenario's ground truth.

    python3 sim/evaluate.py log/missions/<run>/report.json sim/generated/<scenario>/truth.json

The drone splits each gutter into segments itself, so its boundaries do not
line up exactly with the scenario's. Scoring is therefore by where the debris
really is (truth.json lists every piece), for each segment the report gives:

  - debris inside the segment, more than TOLERANCE from both ends
        -> the report must say "needs attention";
  - no debris within TOLERANCE of the segment
        -> the report must say it is fine;
  - debris only within TOLERANCE of a boundary -> either answer is accepted.

Every piece of debris must also lie in some reported segment (nothing missed),
most of every target gutter must be photographed (COVERAGE), the flight must
have completed, and "unclear", unjudged or uninspected segments count as wrong.
无人机自己分段，边界与场景不完全对齐，所以按杂物的真实位置打分：段内部有杂物必须
报“需要清理”，附近都没有杂物必须报“正常”，杂物只在边界附近时两种答案都算对。
Exit code 0 only if everything is right.
"""

import argparse
import json
import math
import sys

TOLERANCE = 0.3        # m around segment boundaries
LINE_DISTANCE = 1.0    # m, a reported segment belongs to the nearest gutter within this
COVERAGE = 0.8         # share of each target gutter that must be photographed


def _param(p, a, b):
    """Position of p along the line a -> b, in metres from a; and distance from the line."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    rx, ry = p[0] - a[0], p[1] - a[1]
    return rx * ux + ry * uy, abs(-rx * uy + ry * ux)


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


def evaluate(report: dict, truth: dict):
    rows, problems = [], []
    photos = [p for p in report.get('photos', []) if p.get('extra', {}).get('from')]
    covered = {g['name']: [] for g in truth['gutters']}
    for p in photos:
        x = p['extra']
        # Which truth gutter is this segment on? 这段属于哪条水槽？
        best = None
        for g in [g for g in truth['gutters'] if g.get('target', True)]:
            # Gutters on different floors can lie on the same line seen from
            # above: compare heights too when the report has them.
            # 不同楼层的水槽在俯视图里可能在同一条线上：报告有高度时也比较高度。
            if 'height' in x and abs(x['height'] - g['height']) > 1.2:
                continue
            s0, d0 = _param(x['from'], g['from'], g['to'])
            s1, d1 = _param(x['to'], g['from'], g['to'])
            if max(d0, d1) <= LINE_DISTANCE and (best is None or max(d0, d1) < best[3]):
                best = (g, min(s0, s1), max(s0, s1), max(d0, d1))
        if best is None:
            rows.append((x['gutter'], x['segment'], '-', p['needs_attention'], False,
                         'not on any gutter of the scenario'))
            continue
        g, lo, hi, _ = best
        if p['needs_attention'] != 'skipped':      # a photo was taken
            covered[g['name']].append((lo, hi))
        debris = [tuple(sorted((_param(d['from'], g['from'], g['to'])[0],
                                _param(d['to'], g['from'], g['to'])[0])))
                  for d in g.get('debris', [])]
        inner = _overlap(lo + TOLERANCE, hi - TOLERANCE, debris)
        near = _overlap(lo - TOLERANCE, hi + TOLERANCE, debris)
        expect = 'yes' if inner > 0.0 else ('no' if near == 0.0 else 'either')
        said = p['needs_attention']
        ok = said in ('yes', 'no') and (expect == 'either' or said == expect)
        rows.append((g['name'], x['segment'], f'{expect} ({inner:.2f} m inside)', said, ok,
                     p['findings']))
    # Nothing missed: every piece of debris on the target lies in a photographed
    # stretch (other buildings' gutters are not to be inspected).
    for g in truth['gutters']:
        if not g.get('target', True):
            continue
        for d in g.get('debris', []):
            a = _param(d['from'], g['from'], g['to'])[0]
            b = _param(d['to'], g['from'], g['to'])[0]
            lo, hi = min(a, b), max(a, b)
            if _overlap(lo, hi, covered[g['name']]) < 0.5 * (hi - lo):
                problems.append(f'{g["name"]}: debris at {lo:.1f}..{hi:.1f} m '
                                f'(scenario segment {d["segment"]}) was not photographed')
    # Everything inspected: the photographed stretches must cover most of each
    # target gutter (its visible part), whether or not it has debris.
    # 每条目标水槽（可见部分）的大部分都要拍到，不管有没有杂物。
    for g in truth['gutters']:
        if not g.get('target', True):
            continue
        ends = g.get('visible') or [g['from'], g['to']]
        a, b = sorted(_param(p, g['from'], g['to'])[0] for p in ends)
        if b - a < 1e-6:
            continue
        share = _overlap(a, b, _merged(covered[g['name']])) / (b - a)
        if share < COVERAGE:
            problems.append(f'{g["name"]}: only {share:.0%} of it was photographed')
    if not report.get('flight_ok', True):
        problems.append(f'flight did not complete: {report.get("flight_note") or "?"}')
    return rows, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('report')
    ap.add_argument('truth')
    args = ap.parse_args(argv)
    with open(args.report, encoding='utf-8') as f:
        report = json.load(f)
    with open(args.truth, encoding='utf-8') as f:
        truth = json.load(f)
    rows, problems = evaluate(report, truth)
    print(f'{"gutter":<18} {"seg":>3}  {"expected":<22} {"report":<8} result')
    for name, i, expect, said, ok, why in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f'{name:<18} {i:>3}  {expect:<22} {said:<8} {"OK" if ok else "WRONG"}'
              + ('' if ok else f'  {why[:70]}'))
    for p in problems:
        print('MISSED:', p)
    right = sum(r[4] for r in rows)
    good = right == len(rows) and rows and not problems
    print(f'\n{right}/{len(rows)} reported segments right, {len(problems)} problem(s)'
          + ('' if good else ' -- report does NOT match the scenario'))
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
