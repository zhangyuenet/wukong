# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Judging one photo: what part of it was judged, and boundaries between homes."""

import math

import pytest
from wukong_flight.skills import Limits
from wukong_mission.gutter_task import apply_boundaries, GutterRequest
from wukong_mission.house_task import HouseTask
from wukong_mission.inspection import PhotoTarget
from wukong_mission.judge import Verdict
from wukong_mission.report import summary_line
from wukong_mission.world import WorldMap

# Facing north from 4.5 m south of a gutter running east-west at north 10.5:
# east is to the right in the picture. 面朝正北，东在画面右侧。
POSE = (6.0, 0.0, 6.2, 0.0, 0.0, 0.0)


class _Judge:
    """Says the gutter is fine; sees a boundary between homes where it is told to."""

    def __init__(self, boundary=None):
        self.boundary = boundary
        self.asked = []

    def judge(self, instruction, label, question, path):
        self.asked.append(path)
        return Verdict(label, path, 'no', 'high', 'clean')

    def find_boundary(self, path, home_position):
        self.home_position = home_position
        return self.boundary, 'wall colour changes'


def _photo(tmp_path):
    from PIL import Image
    path = str(tmp_path / 'photo_01.jpg')
    Image.new('RGB', (1920, 1080), (120, 120, 120)).save(path)
    return path


def _task(judge):
    w = WorldMap('test', (), Limits(), 69, 42, search_radius=30.0)
    return HouseTask(w, GutterRequest('x', 'nearest', []), (0.0, 0.0), '/tmp', judge, 'x')


def _target(start=(10.5, -2.0), end=(10.5, 2.0), home=None, claim=None):
    return PhotoTarget('s1', 'south gutter', 1, 3, start, end, 5.3, (6.0, 0.0, 6.2), home,
                       claim)


def test_segment_not_in_the_picture_is_not_inspected(tmp_path):
    judge = _Judge()
    behind = _target(start=(1.0, -2.0), end=(1.0, 2.0))      # behind the camera
    v = _task(judge)._inspect_one('s1', _photo(tmp_path), behind, POSE, 0.0)
    assert v.needs_attention == 'skipped' and not judge.asked   # not judged from the whole photo


def test_report_says_what_was_judged_and_from_where(tmp_path):
    v = _task(_Judge())._inspect_one('s1', _photo(tmp_path), _target(), POSE, 123.0)
    x = v.extra
    assert v.needs_attention == 'no'
    assert x['from'] == pytest.approx([10.5, -2.0], abs=0.06)
    assert x['to'] == pytest.approx([10.5, 2.0], abs=0.06)
    assert x['t'] == 123.0 and x['pose'] == list(POSE)
    assert 0 < x['columns'][0] < 960 < x['columns'][1] < 1920


def test_boundary_seen_keeps_only_the_home_side(tmp_path):
    judge = _Judge(boundary=0.5)
    # Home to the west (left), the plan's party wall at east 1.0, 1 m short of the
    # segment's end. 目标户在西侧（左），计划的共用墙在东 1.0。
    t = _target(home=(10.5, -4.0), claim=((10.5, -2.0), (10.5, 1.0)))
    v = _task(judge)._inspect_one('s1', _photo(tmp_path), t, POSE, 0.0)
    assert judge.home_position < 0.1                          # told the home is to the left
    assert v.needs_attention == 'no'
    assert v.extra['to'][1] == pytest.approx(0.0, abs=0.1)   # cut at the boundary seen
    assert v.extra['boundary'] == pytest.approx([10.5, 0.0], abs=0.1)
    assert "this home's side" in v.findings


def test_no_boundary_seen_keeps_only_the_claim(tmp_path):
    t = _target(home=(10.5, -4.0), claim=((10.5, -2.0), (10.5, 1.0)))
    v = _task(_Judge())._inspect_one('s1', _photo(tmp_path), t, POSE, 0.0)
    assert v.needs_attention == 'no'
    assert v.extra['from'][1] == pytest.approx(-2.0, abs=0.1)
    assert v.extra['to'][1] == pytest.approx(1.0, abs=0.1)   # not past the party wall as read


def test_stretch_past_the_boundary_is_the_neighbours(tmp_path):
    judge = _Judge(boundary=0.02)                             # at the left edge
    t = _target(home=(10.5, -5.0), claim=((10.5, -2.0), (10.5, 2.0)))  # home further left
    v = _task(judge)._inspect_one('s1', _photo(tmp_path), t, POSE, 0.0)
    assert v.needs_attention == 'neighbour' and not judge.asked


def test_boundary_at_the_homes_own_point_is_in_doubt(tmp_path):
    judge = _Judge(boundary=0.6)                              # about east +0.4
    t = _target(home=(10.5, 0.5), claim=((10.5, -2.0), (10.5, 2.0)))
    v = _task(judge)._inspect_one('s1', _photo(tmp_path), t, POSE, 0.0)
    assert v.needs_attention == 'unclear' and 'in doubt' in v.findings and not judge.asked


def test_boundary_beside_the_homes_point_keeps_its_side(tmp_path):
    # Gazebo: the picture showed both the home's point and, 1.1 m away near its
    # edge, the real boundary; that is not a contradiction. Gazebo 中画面里同时有目标户
    # 的参考点和 1.1 米外靠边的真实分界，这并不矛盾。
    judge = _Judge(boundary=0.07)                             # about east -1.7
    t = _target(home=(10.5, 0.0), claim=((10.5, -2.0), (10.5, 2.0)))
    v = _task(judge)._inspect_one('s1', _photo(tmp_path), t, POSE, 0.0)
    assert v.needs_attention == 'no'
    assert v.extra['from'][1] == pytest.approx(-1.7, abs=0.15)
    assert v.extra['to'][1] == pytest.approx(2.0, abs=0.1)


def test_end_unit_ignores_a_boundary_toward_the_rows_end(tmp_path):
    # Gazebo: the row's end corner was taken for a boundary, cutting 0.6 m off.
    # Gazebo 中把整排尽头的转角当成了分界，截掉了 0.6 米。
    judge = _Judge(boundary=0.8)                              # about east +1.2
    t = _target(home=(10.5, -4.0), claim=((10.5, -2.0), (10.5, 2.0)))
    t.walls = (True, False)                                   # the row ends to the east
    v = _task(judge)._inspect_one('s1', _photo(tmp_path), t, POSE, 0.0)
    assert v.needs_attention == 'no' and 'boundary' not in v.extra
    assert v.extra['to'][1] == pytest.approx(2.0, abs=0.1)


def _v(label, frm, to, said='no', boundary=None):
    x = {'gutter': 'south gutter', 'segment': int(label[1]), 'segments': 3, 'from': frm,
         'to': to, 'home': [10.5, 0.0]}
    if boundary:
        x['boundary'] = boundary
    return Verdict(label, '', said, 'high', 'clean', extra=x)


def test_a_boundary_seen_in_one_photo_bounds_the_others():
    a = _v('s1', [10.5, 1.0], [10.5, 3.25], boundary=[10.5, 3.25])
    b = _v('s2', [10.5, 2.5], [10.5, 4.5])                    # partly past it
    c = _v('s3', [10.5, 3.5], [10.5, 6.0])                    # wholly past it
    d = _v('s4', [10.5, -3.0], [10.5, 0.5])                   # the other way: untouched
    apply_boundaries([a, b, c, d])
    assert a.needs_attention == 'no' and d.needs_attention == 'no'
    assert b.needs_attention == 'unclear' and b.extra['to'] == pytest.approx([10.5, 3.25])
    assert c.needs_attention == 'neighbour'


def test_report_keeps_neighbours_stretches_apart():
    a = _v('s1', [10.5, 1.0], [10.5, 3.0])
    c = _v('s3', [10.5, 3.5], [10.5, 6.0], said='neighbour')
    line = summary_line([a, c])
    assert line.startswith('No gutter segment needs cleaning.')
    assert "Left out as the neighbour's: south gutter segment 3." in line
    assert 'Could not tell' not in line
    assert math.isfinite(len(line))
