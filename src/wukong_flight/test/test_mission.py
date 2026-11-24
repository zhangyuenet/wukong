# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mission state machine, using a crude simulated vehicle."""

from dataclasses import replace
import math

import pytest

from wukong_flight.mission import (Command, Mission, MissionConfig, Phase, PlanRejected,
                                   VehicleState, waypoint_plan, waypoints_from_flat)
from wukong_flight.skills import (Decide, GoTo, Hover, Land, ReturnHome, Scan, Takeoff,
                                  TakePhoto)

DT = 0.05


class FakeVehicle:
    """Just enough PX4 behaviour: arm/offboard on command, chase setpoints at 20 m/s."""

    def __init__(self) -> None:
        self.s = VehicleState(t=0.0, connected=True, position_valid=True,
                              preflight_ok=True, heading=0.3)
        self.commands = []
        self.setpoints = []
        self.offboard_allowed = True
        self.land_allowed = True       # False: PX4 refuses the land mode
        self.drop_land = 0             # this many LAND commands get lost on the way
        self.descent = 1.0             # m per tick in land mode
        self.auto_disarm = True        # PX4 disarms on touchdown (COM_DISARM_LAND)

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
            elif c == Command.LAND and self.drop_land > 0:
                self.drop_land -= 1
            elif c == Command.LAND and self.land_allowed:
                s = replace(s, offboard=False, landing=True)
        if out.setpoint is not None:
            self.setpoints.append(out.setpoint)
        if out.setpoint is not None and s.offboard and s.armed:
            sp = out.setpoint
            d = math.dist((s.x, s.y, s.z), (sp.x, sp.y, sp.z))
            k = min(1.0, 1.0 / d) if d > 0 else 0.0
            s = replace(s, x=s.x + k * (sp.x - s.x), y=s.y + k * (sp.y - s.y),
                        z=s.z + k * (sp.z - s.z), heading=sp.yaw, landed=False,
                        speed=k * d / DT)
        else:
            s = replace(s, speed=0.0)
        if s.landing and s.armed and not s.landed:
            s = replace(s, z=min(0.0, s.z + self.descent))
            if s.z >= 0.0:
                s = replace(s, landed=True)
                if self.auto_disarm:
                    s = replace(s, armed=False, landing=False)
        self.s = replace(s, t=s.t + DT)


def run(mission, vehicle, max_ticks=20000, on_tick=None, on_photo=None):
    phases = [mission.phase]
    for i in range(max_ticks):
        if on_tick:
            on_tick(i, mission, vehicle)
        out = mission.step(vehicle.s)
        if phases[-1] != mission.phase:
            phases.append(mission.phase)
        if out.photo is not None and on_photo is not None:
            on_photo(out.photo, mission, vehicle)
        vehicle.apply(out)
        if mission.finished:
            break
    return phases


def square():
    return waypoint_plan(5, ((10, 0, 5), (10, 10, 8)), 0.5)


def test_full_mission():
    m = Mission(square())
    v = FakeVehicle()
    phases = run(m, v)
    assert m.phase == Phase.DONE and m.failure is None
    assert phases == [Phase.WAIT_READY, Phase.PRESTREAM, Phase.ARMING, Phase.RUNNING,
                      Phase.LAND, Phase.DONE]
    assert Command.LAND in v.commands
    assert m.index == len(m.plan) - 1


def test_bad_plan_is_rejected_before_anything_flies():
    with pytest.raises(PlanRejected) as e:
        Mission([GoTo(1, 1, 5), Land()])
    assert 'first skill must be takeoff' in str(e.value)


def test_targets_are_relative_to_home_and_keep_heading():
    m = Mission([Takeoff(5), GoTo(3, 4, 6), Land()])
    v = FakeVehicle()
    v.s = replace(v.s, x=100.0, y=-50.0, z=-0.2)
    run(m, v)
    assert m.phase == Phase.DONE
    sp = v.setpoints[-1]                       # the last setpoint: at the waypoint
    assert (sp.x, sp.y) == pytest.approx((103.0, -46.0))
    assert sp.z == pytest.approx(-6.2)          # 6 m above home, NED z is down
    assert sp.yaw == pytest.approx(0.3)         # no look_at: heading unchanged


def test_look_at_turns_towards_the_target():
    # From (0, 0) look at a point due east: yaw = +90 deg in NED.
    m = Mission([Takeoff(5), GoTo(0, 0, 5, look_at=(0, 10)), Land()])
    v = FakeVehicle()
    run(m, v)
    assert v.setpoints[-1].yaw == pytest.approx(math.pi / 2)


def test_setpoint_moves_at_max_speed():
    m = Mission([Takeoff(5), GoTo(30, 0, 5), Land()], MissionConfig(max_speed=2.0))
    v = FakeVehicle()
    run(m, v)
    steps = [math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
             for a, b in zip(v.setpoints, v.setpoints[1:])]
    assert max(steps) <= 2.0 * DT + 1e-9


def test_streams_setpoints_before_requesting_offboard():
    m = Mission(square())
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


def test_photo_waits_for_settle_and_continues_after_capture():
    m = Mission([Takeoff(5), GoTo(5, 0, 5, look_at=(15, 0)), TakePhoto('south gutter'),
                 ReturnHome()])
    v = FakeVehicle()
    taken = []

    def shoot(label, mission, vehicle):
        taken.append((label, vehicle.s.speed))
        mission.photo_done(True)
    run(m, v, on_photo=shoot)
    assert m.phase == Phase.DONE
    assert m.photos == [('south gutter', True)]
    assert taken[0][1] < MissionConfig().photo_settle_speed     # vehicle was still


def _photo_tick(hover):
    """
    First tick at which the photo is asked for, or None.

    hover(i) -> (metres above the setpoint, speed) is how the vehicle hangs
    about the viewpoint at tick i. 第 i 步时飞机在视点上方多高、速度多少。
    """
    m = Mission([Takeoff(5), GoTo(5, 0, 5, look_at=(15, 0)), TakePhoto('x'), ReturnHome()])
    v = FakeVehicle()
    for i in range(400):
        out = m.step(v.s)
        if out.photo:
            return i
        v.apply(out)
        if out.setpoint is not None and m.index >= 2:
            up, speed = hover(i)
            v.s = replace(v.s, z=out.setpoint.z - up, speed=speed)
    return None


def test_photo_waits_until_close_to_the_viewpoint():
    # Still braking after a descent: within the 0.5 m arrival tolerance, slow,
    # but 0.3 m high. 下降后还在刹车：在到达范围内、速度也低，但还高 0.3 米。
    assert _photo_tick(lambda i: (0.3, 0.05)) is None
    assert _photo_tick(lambda i: (0.3 if i < 200 else 0.1, 0.05)) is not None


def test_photo_waits_until_still_for_a_moment():
    cfg = MissionConfig()
    assert _photo_tick(lambda i: (0.0, 0.25)) is None            # slow, but not still
    tick = _photo_tick(lambda i: (0.0, 0.25 if i < 200 else 0.05))
    assert tick is not None and (tick - 200) * DT >= cfg.photo_settle_time - DT
    # A moment of movement starts the wait again. 中途动一下，重新计时。
    again = _photo_tick(lambda i: (0.0, 0.25 if i < 200 or i == 205 else 0.05))
    assert again is not None and again > tick


def _first_photo_tick(steady, heading_error, until=None):
    """Tick at which the photo is asked for, flying with a heading error (cleared at until)."""
    m = Mission([Takeoff(5), GoTo(5, 0, 5, look_at=(15, 0)), TakePhoto('x', steady=steady),
                 ReturnHome()])
    v = FakeVehicle()
    for i in range(3000):
        out = m.step(v.s)
        if out.photo:
            return i, out
        v.apply(out)
        err = heading_error if until is None or i < until else 0.0
        v.s = replace(v.s, heading=v.s.heading + err)
    return None, None


def test_steady_photo_waits_like_a_scan():
    # 3 degrees off: good enough for an ordinary photo, not for a steady one.
    # 偏 3 度：普通照片可以拍，需要稳定的照片要等。
    normal, _ = _first_photo_tick(False, math.radians(3))
    assert normal is not None
    assert _first_photo_tick(True, math.radians(3))[0] is None
    # Once on heading it still waits scan_settle_time before asking. 对准后仍需静止一段时间。
    steady, _ = _first_photo_tick(True, math.radians(3), until=normal + 10)
    cfg = MissionConfig()
    assert steady is not None and (steady - normal - 10) * DT >= cfg.scan_settle_time - DT


def test_photo_timeout_does_not_stop_the_mission():
    m = Mission([Takeoff(5), TakePhoto('nothing answers'), Land()],
                MissionConfig(photo_timeout=0.5))
    run(m, FakeVehicle())                         # nobody calls photo_done
    assert m.phase == Phase.DONE
    assert m.photos == [('nothing answers', False)]


def test_return_home_flies_back_then_lands():
    m = Mission([Takeoff(5), GoTo(20, 0, 5), ReturnHome()])
    v = FakeVehicle()
    run(m, v)
    assert m.phase == Phase.DONE
    assert (v.s.x, v.s.y) == pytest.approx((0.0, 0.0), abs=0.5)


def test_hover_longer_than_leg_timeout_is_not_cut_short():
    m = Mission([Takeoff(5), Hover(40), Land()], MissionConfig(leg_timeout=5))
    phases = run(m, FakeVehicle())
    assert Phase.BRAKE not in phases and m.phase == Phase.DONE


def test_waits_while_airborne_or_armed():
    m = Mission(square())
    v = FakeVehicle()
    v.s = replace(v.s, armed=True, landed=False)
    for _ in range(20):
        v.apply(m.step(v.s))
    assert m.phase == Phase.WAIT_READY


def test_second_mission_starts_from_offboard_mode():
    # Regression: after a mission PX4 stays in offboard mode and reports
    # pre_flight_checks_pass=false until setpoints arrive again.
    plan = [Takeoff(5), GoTo(5, 0, 5), Land()]
    first = Mission(plan)
    v = FakeVehicle()
    run(first, v)
    assert first.phase == Phase.DONE
    v.s = replace(v.s, offboard=True, preflight_ok=False)
    second = Mission(plan)
    run(second, v)
    assert second.phase == Phase.DONE


def test_not_ready_times_out():
    m = Mission(square(), MissionConfig(ready_timeout=1.0))
    v = FakeVehicle()
    v.s = replace(v.s, connected=False, position_valid=False)
    run(m, v)
    assert m.phase == Phase.ABORTED


def test_arm_timeout_disarms():
    m = Mission(square(), MissionConfig(arm_timeout=1.0))
    v = FakeVehicle()
    v.offboard_allowed = False        # PX4 refuses offboard
    run(m, v)
    assert m.phase == Phase.ABORTED
    assert v.commands[-1] == Command.DISARM


def test_pilot_takeover_aborts_without_commands():
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()])
    v = FakeVehicle()

    def takeover(i, mission, vehicle):
        if mission.index == 1 and mission.phase == Phase.RUNNING:
            vehicle.s = replace(vehicle.s, offboard=False)  # switched to position mode
    run(m, v, on_tick=takeover)
    assert m.phase == Phase.ABORTED
    assert Command.LAND not in v.commands


def _until(mission, vehicle, cond, max_ticks=5000):
    """Fly until cond(mission, vehicle) holds (checked before each step)."""
    for _ in range(max_ticks):
        if cond(mission, vehicle):
            return
        vehicle.apply(mission.step(vehicle.s))
    raise AssertionError('condition never met')


def _keep_stepping(mission, vehicle, seconds):
    """Step for this long whether or not the mission has finished; return all outputs."""
    outs = []
    for _ in range(int(round(seconds / DT))):
        out = mission.step(vehicle.s)
        outs.append((vehicle.s.t, out))
        vehicle.apply(out)
    return outs


def _commands(outs):
    return [c for _, o in outs for c in o.commands]


def test_pilot_takeover_while_landing_stops_commands():
    # Regression: after PX4 had started landing, a switch to position mode was
    # answered with LAND every second, still 600 s later.
    # 回归测试：PX4 开始降落后遥控切到位置模式，节点每秒重发降落，600 秒后仍在发。
    m = Mission([Takeoff(5), GoTo(10, 0, 5), Land()])
    v = FakeVehicle()
    v.descent = 0.05
    _until(m, v, lambda m, v: m.phase == Phase.LAND and v.s.landing)
    v.apply(m.step(v.s))                          # the mission sees PX4 landing
    v.s = replace(v.s, landing=False)             # pilot: position mode
    outs = _keep_stepping(m, v, 600.0)
    assert m.phase == Phase.ABORTED and 'pilot' in m.failure
    assert _commands(outs) == []
    assert all(o.setpoint is None for _, o in outs)


def test_pilot_takeover_before_px4_starts_landing():
    m = Mission([Takeoff(5), GoTo(10, 0, 5), Land()])
    v = FakeVehicle()
    v.land_allowed = False                        # PX4 has not switched yet
    _until(m, v, lambda m, v: m.phase == Phase.LAND)
    v.s = replace(v.s, offboard=False)            # pilot: position mode
    outs = _keep_stepping(m, v, 60.0)
    assert m.phase == Phase.ABORTED and 'pilot' in m.failure
    assert _commands(outs) == []


def test_px4_failsafe_mode_during_landing_is_not_overridden():
    # Position lost: PX4 switches to its own descend mode; asking for land
    # mode (which needs a position) would fight its failsafe.
    # 失去定位时 PX4 切到自己的下降模式；再请求需要位置的降落模式就是和失效保护对着干。
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()])
    v = FakeVehicle()
    v.land_allowed = False
    _until(m, v, lambda m, v: m.index == 1 and v.s.x > 10)
    v.s = replace(v.s, position_valid=False)
    _until(m, v, lambda m, v: m.phase == Phase.LAND)    # after contact_grace
    v.s = replace(v.s, offboard=False)            # PX4 failsafe: descend mode
    outs = _keep_stepping(m, v, 30.0)
    assert m.phase == Phase.ABORTED and Command.LAND not in _commands(outs)


def test_land_command_lost_is_repeated_while_holding():
    m = Mission([Takeoff(5), GoTo(10, 0, 5), Land()])
    v = FakeVehicle()
    v.drop_land = 2
    run(m, v)
    assert m.phase == Phase.DONE and m.failure is None
    assert v.commands.count(Command.LAND) == 3
    assert (v.s.x, v.s.y) == pytest.approx((10.0, 0.0), abs=0.5)   # held, then landed


def test_px4_refusing_to_land_is_given_up_after_confirm_timeout():
    m = Mission([Takeoff(5), GoTo(10, 0, 5), Land()], MissionConfig(land_confirm_timeout=3.0))
    v = FakeVehicle()
    v.land_allowed = False
    _until(m, v, lambda m, v: m.phase == Phase.LAND)
    t0 = v.s.t
    outs = _keep_stepping(m, v, 10.0)
    assert m.phase == Phase.ABORTED and 'did not start landing' in m.failure
    # Holds the position and keeps asking until the timeout, then nothing at all.
    # 超时前保持位置并重发降落指令，超时后什么都不发。
    assert all(o.setpoint is not None for t, o in outs if t - t0 < 2.5)
    assert len(_commands([x for x in outs if x[0] - t0 < 2.5])) >= 2
    assert all(o.setpoint is None and not o.commands for t, o in outs if t - t0 > 3.5)


def test_silence_after_px4_started_landing_sends_nothing():
    m = Mission([Takeoff(5), GoTo(10, 0, 5), Land()])
    v = FakeVehicle()
    v.descent = 0.05
    _until(m, v, lambda m, v: m.phase == Phase.LAND and v.s.landing)
    v.apply(m.step(v.s))                          # the mission sees PX4 landing
    sent = []
    for _ in range(int(5.0 / DT)):                # 5 s without messages
        sent += m.step(VehicleState(t=v.s.t, connected=False, position_valid=False)).commands
        v.s = replace(v.s, t=v.s.t + DT)
    assert sent == [] and m.phase == Phase.LAND
    run(m, v)                                     # messages again: still landing
    assert m.phase == Phase.DONE and m.failure is None


def test_landing_that_never_ends_is_left_to_px4():
    m = Mission([Takeoff(5), GoTo(10, 0, 5), Land()], MissionConfig(land_timeout=5.0))
    v = FakeVehicle()
    v.descent = 0.0                               # stuck in land mode
    run(m, v)
    assert m.phase == Phase.ABORTED and 'still landing' in m.failure
    assert v.commands.count(Command.LAND) == 1


def test_mode_change_after_touchdown_counts_as_landed():
    m = Mission([Takeoff(5), Land()])
    v = FakeVehicle()
    v.auto_disarm = False
    _until(m, v, lambda m, v: m.phase == Phase.LAND and v.s.landing and v.s.landed)
    v.s = replace(v.s, landing=False)             # pilot: position mode, to disarm by hand
    outs = _keep_stepping(m, v, 5.0)
    assert m.phase == Phase.DONE and m.failure is None
    assert _commands(outs) == []


def test_disarm_in_the_air_while_landing_stops():
    m = Mission([Takeoff(5), Land()])
    v = FakeVehicle()
    v.descent = 0.05
    _until(m, v, lambda m, v: m.phase == Phase.LAND and v.s.landing)
    v.s = replace(v.s, armed=False, landing=False)    # kill switch
    outs = _keep_stepping(m, v, 5.0)
    assert m.phase == Phase.ABORTED and 'disarmed' in m.failure
    assert _commands(outs) == []


def test_lost_position_lands():
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()])
    v = FakeVehicle()

    def lose(i, mission, vehicle):
        if mission.index == 1:
            vehicle.s = replace(vehicle.s, position_valid=False)
    phases = run(m, v, max_ticks=600, on_tick=lose)
    assert Phase.LAND in phases and Phase.BRAKE not in phases
    assert Command.LAND in v.commands


def test_land_request_during_flight_brakes_first():
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()])
    v = FakeVehicle()
    brake_points = []

    def interrupt(i, mission, vehicle):
        if mission.index == 1 and vehicle.s.x > 20 and mission.phase == Phase.RUNNING:
            mission.request_land()
        if mission.phase == Phase.BRAKE and not brake_points:
            brake_points.append(vehicle.s.x)
    phases = run(m, v, on_tick=interrupt)
    assert m.phase == Phase.DONE
    assert phases[-3:] == [Phase.BRAKE, Phase.LAND, Phase.DONE]
    assert 20 < brake_points[0] < 45                  # stopped where it was
    assert v.s.x == pytest.approx(brake_points[0], abs=1.0)


def test_brake_times_out_into_land():
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()], MissionConfig(brake_timeout=0.5))
    v = FakeVehicle()

    def interrupt(i, mission, vehicle):
        if mission.index == 1:
            mission.request_land()
        if mission.phase == Phase.BRAKE:
            vehicle.s = replace(vehicle.s, speed=5.0)   # never slows down
    phases = run(m, v, on_tick=interrupt)
    assert phases[-3:] == [Phase.BRAKE, Phase.LAND, Phase.DONE]


def test_land_request_before_takeoff_cancels():
    m = Mission(square())
    m.request_land()
    run(m, FakeVehicle())
    assert m.phase == Phase.ABORTED


def test_leg_timeout_brakes_and_lands():
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()],
                MissionConfig(leg_timeout=0.5, max_speed=1.0))
    v = FakeVehicle()

    def stall(i, mission, vehicle):        # the vehicle stops making progress
        if mission.index == 1 and mission.phase == Phase.RUNNING:
            vehicle.s = replace(vehicle.s, x=0.0)
    phases = run(m, v, on_tick=stall)
    assert Phase.BRAKE in phases and m.phase == Phase.DONE


def test_waypoints_from_flat():
    assert waypoints_from_flat([1, 2, 3, 4, 5, 6]) == ((1, 2, 3), (4, 5, 6))
    with pytest.raises(ValueError):
        waypoints_from_flat([1, 2])


def test_local_frame_uses_map_coordinates():
    # Vehicle starts at local (6, 0); the plan names absolute map positions.
    m = Mission([Takeoff(5), GoTo(6, 3, 5), ReturnHome()], frame='local', start=(6.0, 0.0))
    v = FakeVehicle()
    v.s = replace(v.s, x=6.0, y=0.0)
    run(m, v)
    assert m.phase == Phase.DONE
    assert any(sp.y == pytest.approx(3.0) and sp.x == pytest.approx(6.0) for sp in v.setpoints)
    assert (v.s.x, v.s.y) == pytest.approx((6.0, 0.0), abs=0.5)    # home = takeoff point


def test_local_frame_aborts_if_vehicle_moved_since_check():
    m = Mission([Takeoff(5), Land()], frame='local', start=(0.0, 0.0))
    v = FakeVehicle()
    v.s = replace(v.s, x=8.0)
    run(m, v)
    assert m.phase == Phase.ABORTED and 'not at the start' in m.failure


# -- Scan and Decide (M2) ------------------------------------------------------

def test_scan_is_requested_when_still_and_logged():
    m = Mission([Takeoff(5), Scan('corner 1'), Land()])
    asked = []
    v = FakeVehicle()
    for _ in range(5000):
        out = m.step(v.s)
        if out.scan:
            asked.append(out.scan)
            m.scan_done(True)
        v.apply(out)
        if m.finished:
            break
    assert asked == ['corner 1'] and m.scans == [('corner 1', True)]
    assert m.phase == Phase.DONE


def _run_with_decision(plan, answer, limits=None, config=MissionConfig()):
    """Fly plan; when Decide asks, answer with answer(mission) -> (skills, limits)."""
    from wukong_flight.skills import Limits
    m = Mission(plan, config, limits or Limits())
    v = FakeVehicle()
    for _ in range(20000):
        out = m.step(v.s)
        if out.decide and answer is not None:
            skills, new_limits = answer(m)
            m.decide_done(skills, new_limits)
        v.apply(out)
        if m.finished:
            break
    return m, v


def test_decide_inserts_checked_skills_before_the_rest():
    m, v = _run_with_decision(
        [Takeoff(5), Decide('after survey'), ReturnHome()],
        lambda m: ([GoTo(8, 0, 5), TakePhoto('x')], None))
    assert m.phase == Phase.DONE
    assert [type(s).__name__ for s in m.plan] == [
        'Takeoff', 'Decide', 'GoTo', 'TakePhoto', 'ReturnHome']
    assert m.decisions == [('after survey', True, [])]
    assert any(sp.x == pytest.approx(8.0) for sp in v.setpoints)


def test_decide_rejects_skills_that_fail_the_check_and_goes_on():
    from wukong_flight.skills import Box, Limits
    wall = Box(5, 7, -5, 5, 10, 'wall')
    m, v = _run_with_decision(
        [Takeoff(5), Decide('d'), ReturnHome()],
        lambda m: ([GoTo(12, 0, 5)], None),          # straight through the wall
        limits=Limits(no_fly=(wall,)))
    assert m.phase == Phase.DONE
    label, ok, errors = m.decisions[0]
    assert not ok and any('no-fly box wall' in e for e in errors)
    assert max(sp.x for sp in v.setpoints) < 1.0       # never went north


def test_decide_can_replace_limits_with_measured_ones():
    from wukong_flight.skills import Box, Cylinder, Limits
    rough = Cylinder((15, 0), 9, 8, 'rough house')      # too big to get close
    precise = Box(11, 19, -5, 5, 5.3, 'house')
    viewpoint = [GoTo(6.5, 0, 4.5, look_at=(11, 0)), TakePhoto('gutter')]
    # With the rough zone the viewpoint (8.5 m from its centre) is rejected...
    m, _ = _run_with_decision([Takeoff(4.5), Decide('d'), ReturnHome()],
                              lambda m: (viewpoint, None), limits=Limits(no_fly=(rough,)))
    assert not m.decisions[0][1]
    # ...with the measured box it is fine.
    m, _ = _run_with_decision([Takeoff(4.5), Decide('d'), ReturnHome()],
                              lambda m: (viewpoint, Limits(no_fly=(precise,))),
                              limits=Limits(no_fly=(rough,)))
    assert m.decisions[0][1] and m.limits.no_fly == (precise,)
    assert m.phase == Phase.DONE


def test_decide_timeout_goes_on_with_the_plan():
    m, _ = _run_with_decision([Takeoff(5), Decide('slow'), ReturnHome()], None,
                              config=MissionConfig(decide_timeout=0.5))
    assert m.phase == Phase.DONE
    assert m.decisions == [('slow', False, ['timed out'])]


def test_decide_with_nothing_to_add():
    m, _ = _run_with_decision([Takeoff(5), Decide('d'), ReturnHome()], lambda m: (None, None))
    assert m.phase == Phase.DONE and m.decisions[0][1] is False


def test_decide_with_own_way_home_replaces_the_rest():
    from wukong_flight.skills import Box, Limits
    wall = Box(5, 7, -5, 5, 10, 'wall')            # between home and the far side
    lim = Limits(no_fly=(wall,))
    # Go round the wall to the far side and back the same way: the decision
    # brings its own route home, so the straight ReturnHome is replaced.
    around = [GoTo(0, 8, 5), GoTo(12, 8, 5), GoTo(12, 0, 5), TakePhoto('far side'),
              GoTo(12, 8, 5), GoTo(0, 8, 5), GoTo(0, 0, 5), ReturnHome()]
    m, v = _run_with_decision([Takeoff(5), Decide('d'), ReturnHome()],
                              lambda m: (around, None), limits=lim)
    assert m.decisions[0][1] and m.phase == Phase.DONE
    assert any(sp.x == pytest.approx(12.0) for sp in v.setpoints)
    # The same skills without the way back would end on the far side, and the
    # straight ReturnHome from there crosses the wall: rejected.
    m, _ = _run_with_decision([Takeoff(5), Decide('d'), ReturnHome()],
                              lambda m: (around[:4], None), limits=lim)
    assert not m.decisions[0][1]


def test_losing_contact_in_flight_is_not_taken_for_a_disarm():
    # Px4Link reports connected=False with armed/offboard False when messages stop.
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()])
    v = FakeVehicle()

    def silence(i, mission, vehicle):
        if mission.index == 1 and mission.phase == Phase.RUNNING:
            vehicle.s = VehicleState(t=vehicle.s.t, connected=False, position_valid=False)
    phases = run(m, v, max_ticks=200, on_tick=silence)
    # Handled as lost contact (the landing path), not as a disarm (giving up),
    # and the defaults of a silent link (disarmed, landed) are not taken for a landing.
    assert Phase.ABORTED not in phases and Phase.DONE not in phases
    assert m.phase == Phase.LAND and 'lost vehicle state' in m.failure
    assert v.commands.count(Command.LAND) >= 5      # keeps asking PX4 to land


def test_landing_completes_when_contact_returns():
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()])
    v = FakeVehicle()
    silent = []

    def silence(i, mission, vehicle):
        if mission.index == 1 and mission.phase == Phase.RUNNING:
            silent.append(vehicle.s)
            vehicle.s = VehicleState(t=vehicle.s.t, connected=False, position_valid=False)
        elif mission.phase == Phase.LAND and silent and len(silent) < 50:
            silent.append(None)
        elif silent and not vehicle.s.connected:          # messages arrive again
            real = next(x for x in silent if x is not None)
            vehicle.s = replace(real, t=vehicle.s.t, landing=True, offboard=False)
    run(m, v, on_tick=silence)
    assert m.phase == Phase.DONE and m.failure      # landed, but the plan was cut short


def test_silent_landing_gives_up_after_lost_timeout():
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()], MissionConfig(lost_timeout=3.0))
    v = FakeVehicle()

    def silence(i, mission, vehicle):
        if mission.index == 1:
            vehicle.s = VehicleState(t=vehicle.s.t, connected=False, position_valid=False)
    run(m, v, max_ticks=400, on_tick=silence)
    assert m.phase == Phase.ABORTED and 'PX4 failsafe' in m.failure


def test_short_gap_in_messages_holds_and_flies_on():
    m = Mission([Takeoff(5), GoTo(45, 0, 5), Land()])
    v = FakeVehicle()
    gap = []

    def hiccup(i, mission, vehicle):
        if mission.index == 1 and vehicle.s.x > 10 and len(gap) < 5:   # 0.5 s of silence
            gap.append(vehicle.s)
            vehicle.s = replace(vehicle.s, connected=False, position_valid=False)
        elif len(gap) == 5 and not vehicle.s.connected:
            vehicle.s = replace(vehicle.s, connected=True, position_valid=True)
    phases = run(m, v, on_tick=hiccup)
    assert len(gap) == 5
    assert m.phase == Phase.DONE and m.failure is None and Phase.BRAKE not in phases
    assert v.s.x == pytest.approx(45, abs=1.0)


def test_hold_position_is_the_target_not_the_hovering_vehicle():
    # In-flight decisions are checked from the target, so they are planned from it.
    m = Mission([Takeoff(5), GoTo(10, 0, 5), Hover(1), Land()])
    v = FakeVehicle()
    seen = []

    def drift(i, mission, vehicle):
        if mission.index == 2 and not seen:
            vehicle.s = replace(vehicle.s, x=vehicle.s.x + 0.4)    # within tolerance
            seen.append((mission.plan_position(vehicle.s), mission.hold_position(vehicle.s)))
    run(m, v, on_tick=drift)
    actual, hold = seen[0]
    assert hold == pytest.approx((10.0, 0.0, 5.0))
    assert actual[0] == pytest.approx(10.4, abs=0.05)
