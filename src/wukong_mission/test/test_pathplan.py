# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Paths and viewpoints around buildings of any shape; everything must pass check_plan."""

import math

import pytest

from wukong_flight.skills import (check_plan, GoTo, HeightField, Limits, ReturnHome, Takeoff,
                                  TakePhoto)
from wukong_mission.inspection import InspectionConfig, plan_inspection, runs_into_taller
from wukong_mission.pathplan import choose_viewpoint, line_of_sight, Planner
from wukong_perception.geometry import Eave


def l_house():
    """Build a 6 m house (north 10..18, east -5..5) with a 3 m annex to its east (east 5..10)."""
    cells = {}
    for i in range(20, 36):
        for j in range(-10, 10):
            cells[(i, j)] = 6.0
    for i in range(20, 30):
        for j in range(10, 20):
            cells[(i, j)] = 3.0
    return HeightField((0.0, 0.0), 0.5, cells, 'house')


LIMITS = Limits(max_altitude=15, max_distance=45, max_skills=200, no_fly=(l_house(),))


def test_straight_when_clear():
    p = Planner(LIMITS)
    assert p.path((0, 0, 4), (5, -3, 4)) == [(5, -3, 4)]


def test_goes_around_the_house_and_passes_the_check():
    p = Planner(LIMITS)
    start, goal = (5.0, 0.0, 4.0), (24.0, 0.0, 4.0)          # south side to north side
    route = p.path(start, goal)
    assert route is not None and route[-1] == goal and len(route) >= 2
    # There and back along the same route: the whole plan must pass the check.
    back = [GoTo(*q) for q in reversed([start] + route[:-1])]
    skills = [Takeoff(4.0), GoTo(*start)] + [GoTo(*q) for q in route] + back + [ReturnHome()]
    assert check_plan(skills, LIMITS).ok


def test_no_path_into_a_blocked_goal():
    assert Planner(LIMITS).path((0, 0, 4), (14, 0, 4)) is None      # inside the house


def test_line_of_sight_is_blocked_by_the_roof():
    hf = l_house()
    assert line_of_sight((5.0, 0.0, 5.0), (9.9, 0.0, 5.9), [hf])
    assert not line_of_sight((24.0, 0.0, 4.0), (9.9, 0.0, 5.9), [hf])    # through the house


def test_viewpoint_over_the_annex_is_raised():
    planner = Planner(LIMITS)
    # East edge of the main house above the annex roof: eave at 5.8 m, facing east.
    vp = choose_viewpoint(planner, (16.0, 5.0), 5.8, (0.0, 1.0), [l_house()])
    assert vp is not None
    assert vp[2] >= 3.0 + 2.0          # above the annex roof plus margin
    assert math.degrees(math.atan2(vp[2] + 0.26 - 5.85, math.dist(vp[:2], (16, 5)))) <= 16.0


def test_plan_inspection_photographs_and_comes_home():
    eave = Eave('south gutter', (10.0, -5.0), (10.0, 5.0), 5.8, 180.0)
    skills, targets, skipped = plan_inspection([eave], [l_house()], (4.0, 0.0, 6.4), (0.0, 0.0),
                                               LIMITS)
    assert not skipped and len(targets) == 4
    assert sum(isinstance(s, TakePhoto) for s in skills) == 4
    last = skills[-1]
    assert (last.north, last.east) == (0.0, 0.0)
    assert check_plan(skills + [ReturnHome()], LIMITS, initial=(4.0, 0.0, 6.4)).ok


def _stretch(targets, gutter):
    """(first, last) east coordinate photographed of a gutter running east-west."""
    ends = [p[1] for t in targets.values() if t.gutter == gutter for p in (t.start, t.end)]
    return min(ends), max(ends)


def test_lower_eave_is_photographed_on_past_its_end_at_a_taller_wall():
    # The annex's south eave, measured 1 m short of the main house's wall
    # (east 5), as when misaligned scans spread the wall over the annex roof.
    # 附属建筑南侧屋檐测得比主楼墙（东 5 米）短 1 米，如同扫描没对齐时高墙盖到了低屋面上。
    annex = Eave('lower south gutter', (10.0, 6.0), (10.0, 10.0), 2.8, 180.0)
    main = Eave('upper south gutter', (10.0, -5.0), (10.0, 5.0), 5.8, 180.0)
    assert runs_into_taller(annex, False, [l_house()])
    assert not runs_into_taller(annex, True, [l_house()])
    assert not runs_into_taller(main, False, [l_house()])
    assert not runs_into_taller(main, True, [l_house()])     # the annex is lower
    cfg = InspectionConfig()
    skills, targets, skipped = plan_inspection([annex, main], [l_house()], (4.0, 0.0, 6.4),
                                               (0.0, 0.0), LIMITS, cfg)
    assert not skipped
    lo, hi = _stretch(targets, 'lower south gutter')
    assert lo == pytest.approx(6.0 - cfg.wall_margin, abs=0.01)     # on up to the wall
    assert hi == pytest.approx(10.0 + cfg.end_margin, abs=0.01)
    lo, hi = _stretch(targets, 'upper south gutter')
    assert (lo, hi) == pytest.approx((-5.0 - cfg.end_margin, 5.0 + cfg.end_margin), abs=0.01)
    assert check_plan(skills + [ReturnHome()], LIMITS, initial=(4.0, 0.0, 6.4)).ok


def test_a_rising_roof_is_not_a_taller_wall():
    # A steep roof (45 deg) climbing from the eave, each cell as high as its
    # top edge (max-pooled), as past an end measured short; that is no wall.
    # 从屋檐升起的陡屋面（45 度，每格取其高边的高度，与最大值降采样一样），不是墙。
    cells = {(i, j): 2.8 + 0.5 * (i - 19) for i in range(20, 30) for j in range(0, 20)}
    roof = HeightField((0.0, 0.0), 0.5, cells, 'roof')
    eave = Eave('south gutter', (10.0, 2.0), (10.0, 8.0), 2.8, 180.0)
    assert not runs_into_taller(eave, False, [roof])
    assert not runs_into_taller(eave, True, [roof])
