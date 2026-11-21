# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Survey and inspection plans must pass the pre-flight check for any house layout."""

import math

import pytest
from wukong_flight.skills import (check_plan, Decide, GoTo, Limits, ReturnHome, Scan,
                                  TakePhoto)
from wukong_mission.inspection import inspection_skills, InspectionConfig, Ring, survey_skills
from wukong_mission.world import RoughHouse
from wukong_perception.geometry import compass, Eave, HouseEstimate


def estimate(center, bearing, length, width, eave_h=2.7, ridge_h=5.3):
    """Build a HouseEstimate as fit_house would return it for a perfect rectangle."""
    b = math.radians(bearing)
    a = (math.cos(b), math.sin(b))
    c = (-math.sin(b), math.cos(b))
    eaves = []
    for sc in (1, -1):
        p = [(center[0] + su * length / 2 * a[0] + sc * width / 2 * c[0],
              center[1] + su * length / 2 * a[1] + sc * width / 2 * c[1]) for su in (-1, 1)]
        out = (bearing + (90 if sc > 0 else 270)) % 360
        eaves.append(Eave(compass(out) + ' gutter', p[0], p[1], eave_h, out))
    return HouseEstimate(center, length, width, bearing % 180, ridge_h, eaves)


# (truth centre, ridge bearing, outline length x width, rough map) per scenario.
HOUSES = {
    'basic': ((15.0, 0.0), 90, 11.0, 9.3, RoughHouse('house', (15.5, -0.2), 9.0, 7.0)),
    'rotated': ((16.0, 6.0), 60, 14.0, 8.3, RoughHouse('house', (15.5, 5.5), 10.5, 7.0)),
    'clean': ((14.0, -3.0), 0, 12.0, 9.3, RoughHouse('house', (14.0, -2.5), 9.0, 7.0)),
}
LIMITS = Limits(max_altitude=15, max_distance=45, margin=2.0)


@pytest.mark.parametrize('name', sorted(HOUSES))
def test_survey_then_inspection_pass_the_check(name):
    center, bearing, length, width, rough = HOUSES[name]
    limits = Limits(max_altitude=15, max_distance=45, margin=2.0, no_fly=(rough.zone(),))
    home = (0.0, 0.0)
    survey = survey_skills(rough, limits, home)
    assert check_plan(survey + [ReturnHome()], limits, home).ok
    assert sum(isinstance(s, Scan) for s in survey) == 8
    assert isinstance(survey[-1], Decide)

    last = next(s for s in reversed(survey) if isinstance(s, GoTo))
    est = estimate(center, bearing, length, width)
    skills, new_limits, targets = inspection_skills(est, (last.north, last.east), home, limits)
    here = (last.north, last.east, last.altitude)
    check = check_plan(skills + [ReturnHome()], new_limits, home, initial=here)
    assert check.ok, check.errors

    photos = [s.label for s in skills if isinstance(s, TakePhoto)]
    assert len(photos) == len(targets) == 2 * math.ceil(length / 3.0 - 1e-6)
    # Every photo is taken facing the middle of its segment, from outside.
    for i, s in enumerate(skills):
        if isinstance(s, TakePhoto):
            view = skills[i - 1]
            t = targets[s.label]
            mid = ((t.start[0] + t.end[0]) / 2, (t.start[1] + t.end[1]) / 2)
            assert view.look_at == pytest.approx(mid, abs=0.01)
            assert math.dist((view.north, view.east), mid) == pytest.approx(4.5, abs=0.01)


def test_view_angle_fits_the_camera():
    cfg = InspectionConfig()
    down = math.degrees(math.atan2(cfg.above_gutter, cfg.standoff))
    assert down < 21.0 - 3.0       # RGB half vertical FOV is ~21 deg; keep a margin


def test_only_named_gutters():
    est = estimate((15.0, 0.0), 90, 11.0, 9.3)
    skills, _, targets = inspection_skills(est, (3.0, 0.0), (0.0, 0.0), LIMITS,
                                           gutters=['south gutter'])
    assert {t.gutter for t in targets.values()} == {'south gutter'}


def test_ring_goes_round_the_short_way():
    ring = Ring(estimate((15.0, 0.0), 90, 11.0, 9.3), 4.5)
    south = (15.0 - 9.3 / 2 - 4.5, 0.0)
    north = (15.0 + 9.3 / 2 + 4.5, 0.0)
    east_end_south = (south[0], 4.0)
    assert len(ring.path(south, north)) == 2          # around one end of the house
    assert ring.path(south, east_end_south) == []     # same side: straight


def test_matching_gutters_by_direction():
    from wukong_mission.gutter_task import matching_gutters
    est = estimate((16.0, 6.0), 60, 14.0, 8.3)       # eaves face 330 (NW) and 150 (SE)
    names = sorted(e.name for e in est.eaves)
    assert names == ['northwest gutter', 'southeast gutter']
    assert matching_gutters(est, []) is None
    assert matching_gutters(est, ['north']) == ['northwest gutter']   # 30 deg off
    assert matching_gutters(est, ['south']) == ['southeast gutter']
    assert sorted(matching_gutters(est, ['east', 'west'])) == names
