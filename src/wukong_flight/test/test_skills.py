# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pre-flight plan check."""

from wukong_flight.skills import (Box, check_plan, GoTo, Hover, Land, Limits, ReturnHome,
                                  Takeoff, TakePhoto)

# The house in sim/models/wukong_house, 15 m north of home: footprint
# north 11..19, east -5..5, ridge at 5.3 m.
HOUSE = Box(north_min=11, north_max=19, east_min=-5, east_max=5, top=5.3, name='house')
LIMITS = Limits(no_fly=(HOUSE,))


def test_good_plan_passes():
    plan = [Takeoff(5), GoTo(6, 0, 4.5, look_at=(11, 0)), TakePhoto('south gutter'),
            ReturnHome()]
    assert check_plan(plan, LIMITS).ok


def test_structure_rules():
    assert not check_plan([], LIMITS).ok
    errors = check_plan([GoTo(1, 0, 3), Takeoff(5), Hover(1)], LIMITS).errors
    text = ' '.join(errors)
    assert 'first skill must be takeoff' in text
    assert 'last skill must be land' in text
    assert 'takeoff is only allowed as the first skill' in text
    assert not check_plan([Takeoff(5), Land(), Hover(1), Land()], LIMITS).ok


def test_altitude_and_distance_limits():
    assert not check_plan([Takeoff(30), Land()], LIMITS).ok
    assert not check_plan([Takeoff(5), GoTo(0, 0, 0.5), Land()], LIMITS).ok
    assert not check_plan([Takeoff(5), GoTo(60, 0, 5), Land()], LIMITS).ok
    assert not check_plan([Takeoff(5), Hover(600), Land()], LIMITS).ok


def test_path_through_the_house_is_rejected():
    # Straight across the house at roof height.
    errors = check_plan([Takeoff(5), GoTo(25, 0, 5), Land()], LIMITS).errors
    assert any('no-fly box house' in e for e in errors)


def test_margin_is_enforced():
    # 1 m south of the wall: inside the 2 m margin.
    assert not check_plan([Takeoff(5), GoTo(10, 0, 5), Land()], LIMITS).ok
    # 3 m south of the wall: fine.
    assert check_plan([Takeoff(5), GoTo(8, 0, 5), Land()], LIMITS).ok


def test_flying_over_with_clearance_is_allowed():
    # Ridge 5.3 m + 2 m margin = 7.3 m; cross at 9 m.
    assert check_plan([Takeoff(9), GoTo(25, 0, 9), ReturnHome()], LIMITS).ok


def test_landing_onto_the_roof_is_rejected():
    errors = check_plan([Takeoff(9), GoTo(15, 0, 9), Land()], LIMITS).errors
    assert any('descends into no-fly box' in e for e in errors)


def test_empty_photo_label_is_rejected():
    assert not check_plan([Takeoff(5), TakePhoto('  '), Land()], LIMITS).ok


def test_start_position_is_used_for_home_and_distance():
    # Taking off at (40, 0): going to (85, 0) is 45 m away, within the 50 m limit.
    assert check_plan([Takeoff(5), GoTo(85, 0, 5), ReturnHome()], Limits(),
                      start=(40, 0)).ok
    # Taking off right next to the house: the climb itself is inside the margin.
    assert not check_plan([Takeoff(5), Land()], LIMITS, start=(10, 0)).ok
