# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the acceptance scorer (sim/evaluate.py).

    python3 -m pytest sim/test_evaluate.py      # with the workspace sourced
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate  # noqa: E402
import pytest  # noqa: E402

# Two homes of a row, gutters along north 10.5; the neighbour's has debris.
# 联排中的两户，水槽在北 10.5 一线；邻居家的有杂物。
TRUTH = {'gutters': [
    {'name': 'unit 3 south gutter', 'target': True, 'from': [10.5, -3.0], 'to': [10.5, 3.0],
     'height': 5.3, 'debris': []},
    {'name': 'unit 4 south gutter', 'target': False, 'from': [10.5, 3.0], 'to': [10.5, 9.0],
     'height': 5.3, 'debris': [{'from': [10.5, 5.0], 'to': [10.5, 5.5], 'segment': 2}]},
]}


def _seg(i, east0, east1, said='no', **extra):
    x = {'gutter': 'south gutter', 'segment': i, 'from': [10.5, east0], 'to': [10.5, east1],
         'height': 5.3}
    x.update(extra)
    return {'needs_attention': said, 'findings': '', 'extra': x}


def _score(*photos, poses=None):
    rows, problems, _ = evaluate.evaluate({'flight_ok': True, 'photos': list(photos)}, TRUTH,
                                          poses)
    return all(r[4] for r in rows) and not problems, rows, problems


def test_exact_home_passes():
    assert _score(_seg(1, -3.0, 0.0), _seg(2, 0.0, 3.0))[0]


def test_a_little_past_the_party_wall_is_tolerated():
    assert _score(_seg(1, -3.0, 0.0), _seg(2, 0.0, 3.4))[0]


def test_the_neighbours_gutter_is_an_intrusion_even_when_judged_right():
    # Review finding: one segment over the target and a whole clean neighbour passed.
    # 评审发现：一段覆盖目标户和整户干净邻居的报告可以通过。
    ok, rows, problems = _score(_seg(1, -3.0, 0.0), _seg(2, 0.0, 9.0, said='yes'))
    assert not ok and any(p.startswith('INTRUDED: unit 4') for p in problems)
    assert all(r[4] for r in rows)     # the answer itself was right: debris was in the photo


def test_a_fifth_of_a_clean_gutter_missed_fails():
    # Review finding: a clean gutter reported only for its first 80% passed.
    ok, _, problems = _score(_seg(1, -3.0, 1.8))
    assert not ok and any(p.startswith('MISSED: unit 3') for p in problems)


def test_stretches_set_aside_as_the_neighbours_are_not_scored():
    ok, rows, _ = _score(_seg(1, -3.0, 0.0), _seg(2, 0.0, 3.0), _seg(3, 3.0, 5.0, 'neighbour'))
    assert ok and [r for r in rows if r[3] == 'neighbour']


def test_true_pose_shows_what_the_photo_really_covered(tmp_path):
    # The drone believes it photographed the whole home from in front of it;
    # it really was 1 m further east. 无人机以为拍到了整户，实际位置偏东 1 米。
    import numpy as np
    from wukong_perception.geometry import project, RGB_X500_DEPTH
    believed = (6.0, 0.0, 6.2, 0.0, 0.0, 0.0)
    u = project(np.array([[10.5, -3.0, 5.3], [10.5, 3.0, 5.3]]), RGB_X500_DEPTH, believed)
    photo = _seg(1, -3.0, 3.0, t=100.0, columns=[float(u[0, 0]), float(u[1, 0])],
                 rows=[300, 1000])
    path = tmp_path / 'truth_poses.jsonl'
    with open(path, 'w') as f:
        for t in (99.0, 100.1, 101.0):
            f.write(json.dumps({'t': t, 'n': 6.0, 'e': 1.0, 'alt': 6.2, 'heading': 0.0,
                                'roll': 0.0, 'pitch': 0.0}) + '\n')
    assert _score(photo)[0]                                    # taken on trust: fine
    ok, _, problems = _score(photo, poses=evaluate.TruePoses(str(path)))
    assert not ok
    assert any(p.startswith('MISSED: unit 3') for p in problems)
    gap = [p for p in problems if p.startswith('INTRUDED: unit 4')]
    assert gap and '1.0' in gap[0]


def test_no_true_pose_near_the_photo_falls_back_to_the_claim(tmp_path):
    path = tmp_path / 'truth_poses.jsonl'
    path.write_text(json.dumps({'t': 0.0, 'n': 0, 'e': 0, 'alt': 0, 'heading': 0, 'roll': 0,
                                'pitch': 0}) + '\n')
    photo = _seg(1, -3.0, 3.0, t=100.0, columns=[0.0, 1920.0], rows=[0, 1080])
    assert evaluate.TruePoses(str(path)).at(100.0) is None
    assert _score(photo, poses=evaluate.TruePoses(str(path)))[0]


@pytest.mark.parametrize('cover, longest', [([(0, 6)], 0.0), ([(0, 2), (2.4, 6)], 0.4),
                                            ([(1, 6)], 1.0), ([], 6.0)])
def test_gaps(cover, longest):
    gaps = evaluate._gaps(0.0, 6.0, cover)
    assert (gaps[0][1] - gaps[0][0] if gaps else 0.0) == pytest.approx(longest)
