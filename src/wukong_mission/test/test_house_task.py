# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""M3 survey geometry: ring stations, townhouse windows, discovery plan."""

import math

import pytest
from wukong_flight.skills import Decide, GoTo, ReturnHome, Scan, Takeoff
from wukong_mission.gutter_task import GutterRequest
from wukong_mission.house_task import clip_to_window, HouseTask, ring_stations
from wukong_mission.judge import FakeJudge
from wukong_mission.world import WorldMap
from wukong_perception.geometry import Eave
from wukong_perception.heightmap import Rectangle
from wukong_flight.skills import Limits  # noqa: I100


def world(townhouse=False):
    return WorldMap('test', (), Limits(max_altitude=18, max_distance=45), 69, 42,
                    search_radius=30.0, townhouse=townhouse)


def test_ring_stations_surround_and_face_the_building():
    r = Rectangle((15.0, 0.0), 12.0, 8.0, 90.0)          # long axis east-west
    st = ring_stations(r, 7.0, 9.0)
    assert len(st) >= 8
    for (n, e), (fn, fe) in st:
        u, v = r.to_uv((n, e))
        assert abs(u) >= 6.0 + 7.0 - 1e-6 or abs(v) >= 4.0 + 7.0 - 1e-6   # on the ring
        fu, fv = r.to_uv((fn, fe))
        assert abs(fu) <= 6.0 + 1e-6 and abs(fv) <= 4.0 + 1e-6            # faces the outline
        assert math.dist((n, e), (fn, fe)) >= 7.0 - 1e-6


def test_ring_end_offsets_stop_the_ring_at_party_walls():
    r = Rectangle((15.0, 0.0), 10.0, 8.0, 90.0)          # east -5 .. +5
    st = ring_stations(r, 10.0, 9.0, end_offsets=(0.0, 0.0))
    for (n, e), _ in st:
        assert abs(e) <= 5.0 + 1e-6                        # never in front of a neighbour
        assert abs(n - 15.0) >= 4.0 + 10.0 - 1e-6 or abs(e) >= 5.0 - 1e-6
    one = ring_stations(r, 10.0, 9.0, end_offsets=(0.0, 10.0))   # an end unit, row ends east
    es = [e for (n, e), _ in one]
    assert min(es) >= -5.0 - 1e-6 and max(es) == pytest.approx(15.0)


def _row_task(row, unit):
    t = HouseTask(world(townhouse=True), GutterRequest('x', 'nearest', []), (0.0, 0.0), '/tmp',
                  FakeJudge(), 'x')
    t.is_row, t.full_rect = True, row
    t.house = type('C', (), {'unit': unit, 'end_unit': None})()
    t.rect = t._row_window(row)
    return t


def test_row_window_ring_grows_only_at_the_row_end():
    row = Rectangle((16.0, 0.0), 32.5, 10.0, 90.0)          # east -16.25 .. +16.25
    middle = _row_task(row, ((16.0, -3.25), (16.0, 3.25)))
    assert middle._ring_end_offsets(10.0) == (0.0, 0.0)
    end = _row_task(row, ((16.0, 9.75), (16.0, 16.25)))
    end.house.end_unit = True
    lo, hi = end._ring_end_offsets(10.0)
    w = end.rect
    east_end = 1 if w.to_ne(w.length / 2, 0.0)[1] > w.center[1] else 0
    assert (lo, hi)[east_end] == 10.0 and (lo, hi)[1 - east_end] == 0.0
    detached = _row_task(row, None)
    detached.is_row = False
    assert detached._ring_end_offsets(10.0) == (10.0, 10.0)


def test_end_unit_reaches_the_row_end_even_when_misread():
    row = Rectangle((16.0, -10.7), 24.4, 11.8, 90.0)        # east -22.9 .. +1.5
    # Read 3.7 m short of the row's east end (pose error plus drift, as in Gazebo).
    # 读数比整排东端短 3.7 米（Gazebo 中位姿误差加漂移造成）。
    t = _row_task(row, ((11.9, -9.9), (11.3, -2.2)))
    t.house.end_unit = True
    t.rect = t._row_window(row)
    ends = sorted(t.rect.to_ne(s * t.rect.length / 2, 0.0)[1] for s in (-1, 1))
    assert ends[1] == pytest.approx(1.5, abs=0.01)           # the gable is in the window
    lo, hi = t._ring_end_offsets(10.0)
    w = t.rect
    east = 1 if w.to_ne(w.length / 2, 0.0)[1] > w.center[1] else 0
    assert (lo, hi)[east] == 10.0 and (lo, hi)[1 - east] == 0.0


def test_overview_aims_at_the_home_in_front_of_the_takeoff_point():
    t = HouseTask(world(townhouse=True), GutterRequest('x', 'nearest', []), (0.0, 0.0), '/tmp',
                  FakeJudge(), 'x')
    # Discovery saw mostly the west part of a row: its centre is 6.5 m west.
    # 环视主要看到联排的西段，中心在西边 6.5 米处。
    t.full_rect = Rectangle((15.0, -6.5), 15.5, 8.0, 90.0)   # front: north 11, east -14.25..1.25
    n, e = t._front_point()
    assert n == pytest.approx(11.0) and e == pytest.approx(1.25 - 3.0)   # kept 3 m in
    t.full_rect = Rectangle((15.0, 0.0), 30.0, 8.0, 90.0)
    assert t._front_point() == pytest.approx((11.0, 0.0))   # straight ahead


def test_discovery_plan_looks_all_round_from_above_home():
    t = HouseTask(world(), GutterRequest('x', 'nearest', []), (1.0, 2.0), '/tmp', FakeJudge(), 'x')
    plan = t.plan()
    assert isinstance(plan[0], Takeoff) and isinstance(plan[-1], ReturnHome)
    assert isinstance(plan[-2], Decide)
    looks = [s for s in plan if isinstance(s, GoTo)]
    assert all((g.north, g.east) == (1.0, 2.0) for g in looks)       # straight up only
    assert sum(isinstance(s, Scan) for s in plan) == 8
    bearings = sorted(math.degrees(math.atan2(g.look_at[1] - 2.0, g.look_at[0] - 1.0)) % 360
                      for g in looks)
    assert bearings == pytest.approx([0, 45, 90, 135, 180, 225, 270, 315], abs=0.5)


def test_row_window_is_centred_on_the_takeoff_point():
    t = HouseTask(world(townhouse=True), GutterRequest('x', 'nearest', []), (0.0, 0.0), '/tmp',
                  FakeJudge(), 'x')
    row = Rectangle((16.0, 0.0), 40.0, 10.0, 90.0)          # east -20 .. +20
    w = t._row_window(row)
    u0, _ = w.to_uv((0.0, 0.0))
    assert w.length == pytest.approx(24.0) and abs(u0) < 1e-6
    end = t._row_window(Rectangle((16.0, -14.5), 26.0, 10.0, 90.0))   # east end at -1.5
    hi = end.length / 2
    e_hi = end.to_ne(hi, 0)[1]
    assert e_hi == pytest.approx(-1.5) and end.length < 24.0          # clipped by the row end


def test_clip_to_window_keeps_only_the_middle_of_a_row():
    window = Rectangle((16.0, 0.0), 10.0, 10.0, 90.0)       # east -5 .. 5
    long_eave = Eave('south gutter', (10.5, -30.0), (10.5, 30.0), 5.3, 180.0)
    side = Eave('east gutter', (11.0, 4.0), (21.0, 4.0), 5.3, 90.0)    # across the row
    far = Eave('east gutter (2)', (11.0, 20.0), (21.0, 20.0), 5.3, 90.0)
    out = clip_to_window([long_eave, side, far], window, margin=1.0)
    names = [e.name for e in out]
    assert names == ['south gutter', 'east gutter']
    s = out[0]
    assert (s.start[1], s.end[1]) == pytest.approx((-6.0, 6.0))


class _Classifier:
    """Answers like ClaudeClassifier would, or fails like it can."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error = answer, error

    def classify_house(self, photo, pose, front, height, out_dir):
        if self.error:
            raise self.error
        return self.answer


def _classified(answer=None, error=None, look=(11.0, 0.0)):
    """Return a task that has just photographed a row (east -16.25 .. +16.25) to judge it."""
    t = HouseTask(world(), GutterRequest('x', 'nearest', []), (0.0, 0.0), '/tmp', FakeJudge(),
                  'x')
    t.full_rect = t.rect = Rectangle((16.0, 0.0), 32.5, 10.0, 90.0)
    t.overview = ('/tmp/none.jpg', (5.0, 0.0, 6.0, 0.0, 0.0, 0.0))
    t.look = look
    t.classifier = _Classifier(answer, error)
    return t


def _row_answer(unit=((11.0, -3.25), (11.0, 3.25)), confidence='high', end_unit=False):
    from wukong_mission.classify import Classification
    return Classification('townhouse_row', 2, 'none', unit, end_unit, confidence, 'test',
                          'model')


@pytest.mark.parametrize('answer, error, why', [
    (None, RuntimeError('network down'), 'could not judge'),
    (_row_answer(unit=None), None, 'begins and ends'),
    (_row_answer(confidence='low'), None, 'unsure'),
    (_row_answer(unit=((11.0, 6.0), (11.0, 12.0))), None, 'not the one the drone looked at'),
])
def test_unsure_house_is_not_inspected(answer, error, why):
    # Regression: a failed judgement fell back to the site setting, and a row
    # without unit boundaries was inspected around the takeoff point: the
    # neighbours' gutters too. 回归测试：判断失败时沿用场地设置；联排没有户界时按起飞点
    # 附近的范围检查，把邻居家的水槽也检查了。
    t = _classified(answer, error)
    skills, limits, note = t.on_decide('what kind of house', (5.0, 0.0, 6.0, 0.0))
    assert skills is None and limits is None      # the checked way home stays the plan
    assert note.startswith('not inspecting') and why in t.refusal


def test_clear_row_answer_goes_on_with_the_survey():
    t = _classified(_row_answer())
    assert t._classify() is None and t.is_row


def test_party_walls_are_carried_square_to_each_eave():
    # Regression: walls carried to the back along an outline 4 deg off landed
    # 1.8 m out in Gazebo. 回归测试：沿偏了 4 度的外接矩形把户界搬到背面，偏了 1.8 米。
    t = _classified(_row_answer())
    t.house = _row_answer()                                    # walls at east -3.25, +3.25
    t.full_rect = Rectangle((16.0, 0.0), 32.5, 10.0, 94.0)    # outline 4 deg off
    front = Eave('south gutter', (11.0, -10.0), (11.0, 10.0), 5.3, 180.0)
    back = Eave('north gutter', (21.0, 10.0), (21.0, -10.0), 5.3, 0.0)
    out = {e.name: e for e in t._unit_eaves([front, back], t.full_rect, t._party_walls())}
    m = t.survey.party_wall_margin
    for name in ('south gutter', 'north gutter'):
        ends = sorted((out[name].start[1], out[name].end[1]))
        assert ends == pytest.approx([-3.25 - m, 3.25 + m], abs=0.01)


def test_end_unit_corner_gets_the_end_margin_not_the_party_wall():
    end = _row_answer(unit=((11.0, 9.75), (11.0, 16.25)), end_unit=True)
    t = _classified(end, look=(11.0, 13.0))
    t.house = end                                              # row ends at east +16.25
    front = Eave('south gutter', (11.0, 5.0), (11.0, 16.25), 5.3, 180.0)
    side = Eave('east gutter', (11.0, 16.25), (21.0, 16.25), 5.3, 90.0)    # across the row
    walls = t._party_walls()
    assert walls == [(11.0, 9.75)]
    out = {e.name: e for e in t._unit_eaves([front, side], t.full_rect, walls)}
    m, extra = t.survey.party_wall_margin, t.cfg.end_margin
    assert out['south gutter'].start[1] == pytest.approx(9.75 - m)
    assert out['south gutter'].end[1] == pytest.approx(16.25 + extra)
    assert out['east gutter'].start[0] == pytest.approx(11.0 - extra)
    assert out['east gutter'].end[0] == pytest.approx(21.0 + extra)


def test_photos_along_a_row_know_the_home_and_their_claim():
    from wukong_mission.inspection import PhotoTarget
    t = _classified(_row_answer())
    targets = {
        'a': PhotoTarget('a', 'south gutter', 1, 3, (11.0, -4.25), (11.0, -1.4), 5.3, (6, 0, 6)),
        'b': PhotoTarget('b', 'south gutter', 3, 3, (11.0, 1.4), (11.0, 4.25), 5.3, (6, 0, 6)),
        'c': PhotoTarget('c', 'east gutter', 1, 1, (11.0, 16.0), (21.0, 16.0), 5.3, (6, 0, 6)),
    }
    t.house = _row_answer()
    t._mark_home(targets, t._party_walls())
    a, b, c = targets['a'], targets['b'], targets['c']
    assert a.home == pytest.approx((11.0, 0.0)) and b.home == pytest.approx((11.0, 0.0))
    assert [*a.claim[0], *a.claim[1]] == pytest.approx([11.0, -3.25, 11.0, -1.4])   # to wall
    assert [*b.claim[0], *b.claim[1]] == pytest.approx([11.0, 1.4, 11.0, 3.25])
    assert c.home is None and c.claim is None                          # across the row
    assert a.walls == (True, True) and b.walls == (True, True)
