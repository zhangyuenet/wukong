# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""M3 survey geometry: ring stations, townhouse windows, discovery plan."""

import math

import numpy as np
import pytest
from wukong_flight.skills import (check_plan, Decide, GoTo, HeightField, ReturnHome, Scan,
                                  Takeoff, TakePhoto)
from wukong_mission.explore import Stop
from wukong_mission.gutter_task import GutterRequest
from wukong_mission.house_task import (clip_to_window, CONTINUE, DISCOVERY, HouseTask, LOOK_AHEAD,
                                       ring_stations)
from wukong_mission.judge import FakeJudge
from wukong_mission.world import RoughHouse, WorldMap
from wukong_perception.geometry import DEPTH_RANGE, Eave
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


class _Scans:
    """Stands in for DepthCamera: the clouds and depth frames discovery has seen."""

    def __init__(self, *clouds, rays=()):
        self.clouds = [np.asarray(c, float) for c in clouds]
        self.labels = [f'look around {k + 1}' for k in range(len(self.clouds))]
        self.rays = list(rays)


def _open_ground(center=(0.0, 0.0), radius=9.0):
    """Return a ring of ground points around center: nothing taller in view."""
    a = np.linspace(0, 2 * math.pi, 400)
    r = np.repeat([radius - 1.0, radius], 200)
    return np.column_stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a), 0.0 * a])


def _frame(cam, heading):
    """Return one depth frame over flat open ground, as geometry.depth_rays gives it."""
    yaw, pitch = np.meshgrid(np.radians(np.linspace(-36.5, 36.5, 60)),   # the OAK-D's view
                             np.radians(np.linspace(-29.0, 29.0, 46)))
    yaw, pitch = heading + yaw.ravel(), pitch.ravel()
    d = np.column_stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw),
                         np.sin(pitch)])
    to_ground = np.where(d[:, 2] < 0, cam[2] / np.maximum(-d[:, 2], 1e-9), np.inf)
    hit = to_ground < DEPTH_RANGE
    ends = np.asarray(cam) + np.minimum(to_ground, DEPTH_RANGE)[:, None] * d
    return tuple(cam), ends, hit


def _look_around(cam, headings=8):
    return [_frame(cam, 2 * math.pi * k / headings) for k in range(headings)]


def test_discovery_moves_on_the_way_it_faced_on_the_ground():
    """Nothing within depth range: step on (east here) until the scans reach search_radius."""
    t = HouseTask(world(), GutterRequest('x', 'nearest', []), (1.0, 2.0), '/tmp', FakeJudge(),
                  'x', facing=math.pi / 2)
    t.depth = _Scans(_open_ground((1.0, 2.0)), rays=_look_around((1.0, 2.0, 6.0)))
    pose = (1.0, 2.0, 6.0, 0.0)
    steps = []
    while True:
        skills, limits, note = t.on_decide(DISCOVERY, pose)
        if skills is None:
            break
        go, scan, decide = skills[:3]
        assert isinstance(scan, Scan) and scan.label.startswith(LOOK_AHEAD)
        assert isinstance(decide, Decide) and isinstance(skills[-1], ReturnHome)
        assert go.look_at[0] == pytest.approx(go.north) and go.look_at[1] > go.east  # facing east
        pose = (go.north, go.east, go.altitude, math.pi / 2)
        steps.append((go.north, go.east, go.altitude))
        t.depth.rays.append(_frame(pose[:3], math.pi / 2))                     # the look ahead
    assert steps and all(s[0] == 1.0 and s[2] == 6.0 for s in steps)
    # The last step is shorter, so that the scans reach search_radius ahead.
    # 最后一步短一些，让扫描正好覆盖到前方的搜索半径。
    assert [s[1] - 2.0 for s in steps] == pytest.approx([3.0, 6.0, 9.0, 11.5])
    assert t.world.search_radius - DEPTH_RANGE == pytest.approx(11.5)
    assert note == 'no building within 18.5 m all round, nor 30.0 m ahead on the takeoff heading'


@pytest.mark.parametrize('frames', [
    [],                                                    # no frame came in
    [_frame((0.0, 0.0, 6.0), math.pi)],                    # one came in, looking the other way
])
def test_discovery_does_not_move_on_where_it_has_not_seen(frames):
    """Nothing in the way is not enough: the way itself must have been seen free."""
    t = HouseTask(world(), GutterRequest('x', 'nearest', []), (0.0, 0.0), '/tmp', FakeJudge(),
                  'x', facing=0.0)
    t.depth = _Scans(_open_ground(), rays=frames)
    skills, _, note = t.on_decide(DISCOVERY, (0.0, 0.0, 6.0, 0.0))
    assert skills is None
    assert 'seen free for only' in note


def _hinted(center, radius, max_height=7.0):
    """Return an M2 map: one house known only to lie within radius of center."""
    rough = RoughHouse('house', center, radius, max_height)
    return WorldMap('test', (), Limits(max_altitude=15, max_distance=45, no_fly=(rough.zone(),)),
                    69, 42, rough_houses=(rough,))


def _block(center, size, height, step=0.1):
    """Return points over a flat roof: what scans see of a box-shaped building from above."""
    n, e = np.meshgrid(np.arange(-size[0] / 2, size[0] / 2, step),
                       np.arange(-size[1] / 2, size[1] / 2, step))
    return np.column_stack([center[0] + n.ravel(), center[1] + e.ravel(),
                            np.full(n.size, height)])


def test_with_a_rough_position_discovery_takes_the_building_there():
    """Not the nearest building: the one in the map's circle, even with another nearer."""
    t = HouseTask(_hinted((1.0, 14.0), 7.0), GutterRequest('x', 'house', []), (0.0, 0.0),
                  '/tmp', FakeJudge(), 'x', facing=0.0)
    assert t.facing == pytest.approx(math.atan2(14.0, 1.0))      # moves on towards it
    shed = _block((8.0, 0.0), (6.0, 6.0), 4.0)                    # nearer, outside the circle
    house = _block((0.0, 15.0), (8.0, 8.0), 5.0)
    t.depth = _Scans(_open_ground(radius=20.0), shed, house)
    skills, _, note = t.on_decide(DISCOVERY, (0.0, 0.0, 6.0, 0.0))
    assert skills is None and 'no way to judge' in note           # no classifier here
    assert t.rect.center[0] == pytest.approx(0.0, abs=0.5)
    assert t.rect.center[1] == pytest.approx(15.0, abs=0.5)
    # Its unseen cells are as tall as the map says this house is at most.
    # 它没看到的格子按地图给出的这栋房屋的最高高度禁飞。
    unknown = t._unknown_cells(t._heightmap(), 0.5)
    assert unknown.cells and max(unknown.cells.values()) <= 7.0 + 1e-6


def test_with_a_rough_position_discovery_moves_on_towards_it_short_of_its_circle():
    """Out of depth range: step on towards the circle, but not into it or its margin."""
    world = _hinted((0.0, 32.0), 10.0)
    t = HouseTask(world, GutterRequest('x', 'house', []), (0.0, 0.0), '/tmp', FakeJudge(), 'x',
                  facing=0.0)                                     # faced north on the ground
    t.depth = _Scans(_open_ground(), rays=_look_around((0.0, 0.0, 6.0)))
    pose = (0.0, 0.0, 6.0, 0.0)
    steps = []
    while True:
        skills, limits, note = t.on_decide(DISCOVERY, pose)
        if skills is None:
            break
        assert check_plan(skills, limits, initial=pose[:3]).ok, note
        go = skills[0]
        pose = (go.north, go.east, go.altitude, math.pi / 2)
        steps.append(go.east)
        assert go.north == pytest.approx(0.0)                     # east, towards the circle
        t.depth.rays.append(_frame(pose[:3], math.pi / 2))
    # The circle's edge is 22 m away; 2 m margin and 0.5 m to spare.
    # 圆的边缘在 22 米处，留出 2 米安全距离和 0.5 米余量。
    assert steps == pytest.approx([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 19.5])
    assert note == 'no building seen yet within 10 m of the rough position (0.0, 32.0) ' \
        '(12.5 m away)'


@pytest.mark.parametrize('start, height, alt', [
    ((0.0, 0.0), 5.0, 10.0 * math.tan(math.radians(19.0))),         # lower than the drone
    ((0.0, 0.0), 8.0, 6.0),                                        # seen face on already
    ((0.0, 3.0), 5.0, 6.0),                                        # outside the open column
])
def test_the_overview_of_a_low_house_is_taken_from_lower_down(start, height, alt):
    """Its whole front in the picture: the bottom edge meets the ground 1 m in front of it."""
    t = HouseTask(_hinted((1.0, 14.0), 7.0, max_height=9.0), GutterRequest('x', 'house', []),
                  (0.0, 0.0), '/tmp', FakeJudge(), 'x')
    t.classifier = _Classifier()
    t.depth = _Scans(_open_ground(radius=20.0), _block((0.0, 15.0), (8.0, 8.0), height))
    skills, limits, _ = t.on_decide(DISCOVERY, (*start, 6.0, 0.0))
    go, photo, decide = skills[:3]
    assert isinstance(photo, TakePhoto) and isinstance(decide, Decide)
    assert (go.north, go.east) == start
    assert go.altitude == pytest.approx(alt, abs=0.1)
    assert check_plan(skills, limits, initial=(*start, 6.0)).ok


def test_a_house_not_on_the_map_is_refused_before_takeoff():
    with pytest.raises(ValueError, match='no house "garage"'):
        HouseTask(_hinted((1.0, 14.0), 7.0), GutterRequest('x', 'garage', []), (0.0, 0.0),
                  '/tmp', FakeJudge(), 'x')


def test_discovery_does_not_move_on_towards_something_in_the_way():
    t = HouseTask(world(), GutterRequest('x', 'nearest', []), (0.0, 0.0), '/tmp', FakeJudge(),
                  'x', facing=0.0)
    pole = [(5.0, 0.2, z) for z in np.arange(0.0, 8.0, 0.05)]     # 5 m north, too thin for a house
    t.depth = _Scans(_open_ground(), pole)
    skills, _, note = t.on_decide(DISCOVERY, (0.0, 0.0, 6.0, 0.0))
    assert skills is None
    assert 'something 5.0 m ahead' in note


def _class_a(frames):
    """Return a task that flies only where seen free, after the given depth frames."""
    import dataclasses
    t = HouseTask(world(), GutterRequest('x', 'nearest', []), (0.0, 0.0), '/tmp', FakeJudge(),
                  'x', facing=0.0)
    t.survey = dataclasses.replace(t.survey, seen_free_only=True)
    t.depth = _Scans(_open_ground(), rays=frames)
    t._record_trail((0.0, 0.0, 6.0))                        # above the takeoff point
    return t


def _photo(t, label, goal, face):
    from wukong_mission.inspection import PhotoTarget
    t.targets[label] = PhotoTarget(label, 'north gutter', 1, 1, face, face, 3.0, goal)
    return Stop(goal, face, (TakePhoto(label),), label)


def test_class_a_flies_what_it_has_seen_and_looks_before_going_on():
    """One frame looking north: the photo there is taken; the way east must be looked at first."""
    t = _class_a([_frame((0.0, 0.0, 6.0), 0.0)])
    t._queue = [_photo(t, 'a', (8.0, 0.0, 6.0), (14.0, 0.0)),
                _photo(t, 'b', (8.0, 14.0, 6.0), (14.0, 14.0))]
    t._queue_kind = 'inspect'
    skills, _, note = t._run_queue((0.0, 0.0, 6.0, 0.0), world().limits, 'photos')
    kinds = [type(k).__name__ for k in skills]
    assert TakePhoto('a') in skills and TakePhoto('b') not in skills
    look, scan, decide = skills[kinds.index('Scan') - 1:kinds.index('Scan') + 2]
    assert decide == Decide(CONTINUE) and scan.label.startswith('way check')
    assert look.look_at[1] > look.east + 5.0                           # facing east, towards b
    assert isinstance(skills[-1], ReturnHome) and 'looking along the way to b' in note
    # Nothing new seen from there: b is skipped and the drone goes home.
    # 在那里没有看到新的东西：跳过 b，返航。
    skills, _, _ = t.on_decide(CONTINUE, (look.north, look.east, look.altitude, math.pi / 2))
    assert [k for k in skills if isinstance(k, (TakePhoto, Scan))] == []
    assert isinstance(skills[-1], ReturnHome)
    assert [k.label for k in t.skipped] == ['b'] and 'not seen free' in t.skipped[0].reason
    assert 'b' not in t.targets


def test_class_a_checks_decisions_without_the_unseen_part_of_the_house():
    """The voxel map settles unseen space; that zone only steers the planner."""
    import dataclasses
    from wukong_mission.house_task import UNSEEN_HOUSE
    t = _class_a([])
    zones = (HeightField((0.0, 0.0), 1.0, {(10, 0): 6.0}, 'surroundings (seen)'),
             HeightField((0.0, 0.0), 1.0, {(12, 0): 18.0}, UNSEEN_HOUSE))
    lim = dataclasses.replace(world().limits, no_fly=zones)
    assert [z.name for z in t._flight_limits(lim).no_fly] == ['surroundings (seen)']
    t.survey = dataclasses.replace(t.survey, seen_free_only=False)
    assert t._flight_limits(lim) is lim
