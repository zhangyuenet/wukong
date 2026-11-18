# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mission state machine, using a crude simulated vehicle."""

from dataclasses import replace

import pytest

from wukong_flight.mission import (Command, Mission, MissionConfig, Phase,
                                   VehicleState, waypoints_from_flat)

DT = 0.05


class FakeVehicle:
    """Just enough PX4 behaviour: arm/offboard on command, jump towards setpoints."""

    def __init__(self) -> None:
        self.s = VehicleState(t=0.0, connected=True, position_valid=True,
                              preflight_ok=True, heading=0.3)
        self.commands = []
        self.offboard_allowed = True

    def apply(self, out) -> None:
        s = self.s
        for c in out.commands:
            self.commands.append(c)
            if c == Command.ENGAGE_OFFBOARD and self.offboard_allowed:
                s = replace(s, offboard=True)
            elif c == Command.ARM:
                s = replace(s, armed=True)
            elif c == Command.DISARM:
                s = replace(s, armed=False)
            elif c == Command.LAND:
                s = replace(s, offboard=False, landing=True)
        if out.setpoint is not None and s.offboard and s.armed:
            # move 1 m per tick towards the setpoint
            sp = out.setpoint
            d = ((sp.x - s.x) ** 2 + (sp.y - s.y) ** 2 + (sp.z - s.z) ** 2) ** 0.5
            k = min(1.0, 1.0 / d) if d > 0 else 0.0
            s = replace(s, x=s.x + k * (sp.x - s.x), y=s.y + k * (sp.y - s.y),
                        z=s.z + k * (sp.z - s.z), landed=False, speed=k * d / DT)
        else:
            s = replace(s, speed=0.0)
        if s.landing:
            s = replace(s, z=min(0.0, s.z + 1.0))
            if s.z >= 0.0:
                s = replace(s, landed=True, armed=False, landing=False)
        self.s = replace(s, t=s.t + DT)


def run(mission, vehicle, max_ticks=5000, on_tick=None):
    phases = [mission.phase]
    for i in range(max_ticks):
        if on_tick:
            on_tick(i, mission, vehicle)
        out = mission.step(vehicle.s)
        if not phases or phases[-1] != mission.phase:
            phases.append(mission.phase)
        vehicle.apply(out)
        if mission.finished:
            break
    return phases


def test_full_mission():
    m = Mission(MissionConfig(takeoff_altitude=5, waypoints=((10, 0, 5), (10, 10, 8)),
                              hold_time=0.5))
    v = FakeVehicle()
    phases = run(m, v)
    assert m.phase == Phase.DONE and m.failure is None
    assert phases == [Phase.WAIT_READY, Phase.PRESTREAM, Phase.ARMING, Phase.TAKEOFF,
                      Phase.GOTO, Phase.HOLD, Phase.GOTO, Phase.HOLD, Phase.LAND, Phase.DONE]
    assert Command.LAND in v.commands


def test_targets_are_relative_to_home_and_keep_heading():
    m = Mission(MissionConfig(waypoints=((3, 4, 6),)))
    v = FakeVehicle()
    v.s = replace(v.s, x=100.0, y=-50.0, z=-0.2)
    seen = []

    def watch(i, mission, vehicle):
        if mission.phase == Phase.GOTO:
            seen.append(mission.target())
    run(m, v, on_tick=watch)
    sp = seen[0]
    assert (sp.x, sp.y) == pytest.approx((103.0, -46.0))
    assert sp.z == pytest.approx(-6.2)     # 6 m above home, NED z is down
    assert sp.yaw == pytest.approx(0.3)


def test_streams_setpoints_before_requesting_offboard():
    m = Mission(MissionConfig())
    v = FakeVehicle()
    outs = []
    for _ in range(100):
        out = m.step(v.s)
        outs.append(out)
        if Command.ENGAGE_OFFBOARD in out.commands:
            break
        v.apply(out)
    streamed = [o for o in outs[:-1] if o.setpoint is not None]
    assert len(streamed) * DT >= 1.0 - 1e-9


def test_waits_while_airborne_or_armed():
    m = Mission(MissionConfig())
    v = FakeVehicle()
    v.s = replace(v.s, armed=True, landed=False)
    for _ in range(20):
        v.apply(m.step(v.s))
    assert m.phase == Phase.WAIT_READY


def test_second_mission_starts_from_offboard_mode():
    # Regression: after a mission PX4 stays in offboard mode and reports
    # pre_flight_checks_pass=false until setpoints arrive again.
    first = Mission(MissionConfig(waypoints=((5, 0, 5),), hold_time=0.1))
    v = FakeVehicle()
    run(first, v)
    assert first.phase == Phase.DONE
    v.s = replace(v.s, offboard=True, preflight_ok=False)
    second = Mission(MissionConfig(waypoints=((5, 0, 5),), hold_time=0.1))
    run(second, v)
    assert second.phase == Phase.DONE


def test_not_ready_times_out():
    m = Mission(MissionConfig(ready_timeout=1.0))
    v = FakeVehicle()
    v.s = replace(v.s, connected=False, position_valid=False)
    run(m, v)
    assert m.phase == Phase.ABORTED


def test_arm_timeout_disarms():
    m = Mission(MissionConfig(arm_timeout=1.0))
    v = FakeVehicle()
    v.offboard_allowed = False        # PX4 refuses offboard
    run(m, v)
    assert m.phase == Phase.ABORTED
    assert v.commands[-1] == Command.DISARM


def test_pilot_takeover_aborts_without_commands():
    m = Mission(MissionConfig(waypoints=((50, 0, 5),)))
    v = FakeVehicle()

    def takeover(i, mission, vehicle):
        if mission.phase == Phase.GOTO:
            vehicle.s = replace(vehicle.s, offboard=False)  # switched to position mode
    run(m, v, on_tick=takeover)
    assert m.phase == Phase.ABORTED
    assert Command.LAND not in v.commands


def test_lost_position_lands():
    m = Mission(MissionConfig(waypoints=((50, 0, 5),)))
    v = FakeVehicle()

    def lose(i, mission, vehicle):
        if mission.phase == Phase.GOTO:
            vehicle.s = replace(vehicle.s, position_valid=False)
    phases = run(m, v, max_ticks=400, on_tick=lose)
    assert Phase.LAND in phases
    assert Command.LAND in v.commands


def test_land_request_during_flight_brakes_first():
    m = Mission(MissionConfig(waypoints=((50, 0, 5),)))
    v = FakeVehicle()
    brake_targets = []

    def interrupt(i, mission, vehicle):
        if mission.phase == Phase.GOTO and vehicle.s.x > 20:
            mission.request_land()
        if mission.phase == Phase.BRAKE:
            brake_targets.append(mission.target())
    phases = run(m, v, on_tick=interrupt)
    assert m.phase == Phase.DONE
    assert phases[-3:] == [Phase.BRAKE, Phase.LAND, Phase.DONE]
    assert Phase.HOLD not in phases
    assert 20 < brake_targets[0].x < 50     # stops where it was, not at the waypoint
    assert v.s.x == pytest.approx(brake_targets[0].x)


def test_brake_times_out_into_land():
    m = Mission(MissionConfig(waypoints=((50, 0, 5),), brake_timeout=0.5))
    v = FakeVehicle()

    def interrupt(i, mission, vehicle):
        if mission.phase == Phase.GOTO:
            mission.request_land()
        if mission.phase == Phase.BRAKE:
            vehicle.s = replace(vehicle.s, speed=5.0)   # never slows down
    phases = run(m, v, on_tick=interrupt)
    assert phases[-3:] == [Phase.BRAKE, Phase.LAND, Phase.DONE]


def test_land_request_before_takeoff_cancels():
    m = Mission(MissionConfig())
    m.request_land()
    run(m, FakeVehicle())
    assert m.phase == Phase.ABORTED


def test_leg_timeout_lands():
    m = Mission(MissionConfig(waypoints=((1000, 0, 5),), leg_timeout=2.0))
    phases = run(m, FakeVehicle())
    assert Phase.LAND in phases and Phase.HOLD not in phases


def test_waypoints_from_flat():
    assert waypoints_from_flat([1, 2, 3, 4, 5, 6]) == ((1, 2, 3), (4, 5, 6))
    with pytest.raises(ValueError):
        waypoints_from_flat([1, 2])


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        MissionConfig(takeoff_altitude=-1)
    with pytest.raises(ValueError):
        MissionConfig(waypoints=((0, 0, 0),))
