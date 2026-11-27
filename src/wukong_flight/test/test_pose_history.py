# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pose history: the pose at a camera frame's stamp."""

import math

import pytest
from wukong_flight.pose_history import (COMPANION_CLOCK, euler_from_quaternion, FrameShift,
                                        PoseHistory, PX4_CLOCK)

# PX4 time = sent stamp + offset; the sent stamps are on the companion's clock.
OFFSET = -1_795_747_827_684_719            # us, as seen in SITL
T0 = 1_795_747_929_000_000                 # us, a sent stamp


def _yaw_q(yaw: float, pitch: float = 0.0):
    """Quaternion (w, x, y, z) of yaw then pitch (ZYX)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def _history(n=50, step_us=10_000, speed=2.0, yaw0=0.0, yaw_rate=0.0, t0=T0, offset=OFFSET):
    """Flying north at speed m/s and turning at yaw_rate rad/s, a sample every step_us."""
    h = PoseHistory()
    h.set_offset(offset, t0 - 1_000_000)
    for k in range(n):
        t = t0 + k * step_us
        h.add_position(t, speed * k * step_us * 1e-6, 0.0, -5.0)
        h.add_attitude(t, _yaw_q(yaw0 + yaw_rate * k * step_us * 1e-6))
    return h


def _px4(h, sent_us):
    """Return the pose at a moment given on the sent clock, asked on PX4's clock (Gazebo)."""
    return h.pose_at(sent_us + h.offset_us, PX4_CLOCK)


def test_euler_from_quaternion():
    roll, pitch, yaw = euler_from_quaternion(_yaw_q(math.radians(30), math.radians(10)))
    assert roll == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(math.radians(10))
    assert yaw == pytest.approx(math.radians(30))


def test_a_gazebo_stamp_is_on_px4s_clock_and_is_interpolated():
    h = _history()
    # 125 ms after the first sample, stamped by Gazebo on the simulation clock.
    stamp = T0 + 125_000 + OFFSET
    x, y, z, yaw, roll, pitch = h.pose_at(stamp, PX4_CLOCK)
    assert x == pytest.approx(0.25)
    assert (y, z) == pytest.approx((0.0, -5.0))
    assert (yaw, roll, pitch) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_a_real_camera_stamp_is_on_the_companions_clock():
    h = _history()
    assert h.pose_at(T0 + 125_000, COMPANION_CLOCK)[0] == pytest.approx(0.25)
    # The wrong clock said: seconds from every sample. 时钟说错了：离所有样本都有几秒。
    assert h.pose_at(T0 + 125_000, PX4_CLOCK) is None
    assert 'seconds from every pose sample' in h.why_not
    with pytest.raises(ValueError):
        h.pose_at(T0 + 125_000, 'gazebo')


def test_the_clock_is_never_guessed_when_the_two_are_close():
    # PX4's clock only 200 ms from the companion's, flying 3 m/s: a Gazebo
    # stamp taken for a companion one would give a pose 0.6 m off.
    # PX4 时钟和机载电脑时钟只差 200 毫秒、以 3 m/s 飞行：把 Gazebo 的时间戳当成机载电脑
    # 时钟的，位置会差 0.6 米。
    h = _history(speed=3.0, offset=-200_000)
    stamp = T0 + 300_000 - 200_000          # 300 ms after the first sample, on PX4's clock
    assert h.pose_at(stamp, PX4_CLOCK)[0] == pytest.approx(0.9)
    assert h.pose_at(stamp, COMPANION_CLOCK)[0] == pytest.approx(0.3)   # what it says it is


def test_no_pose_on_px4s_clock_before_the_first_offset():
    h = PoseHistory()
    h.add_position(T0, 0.0, 0.0, -5.0)
    h.add_attitude(T0, _yaw_q(0.0))
    assert h.pose_at(T0 + OFFSET, PX4_CLOCK) is None
    assert 'offset' in h.why_not


def test_heading_is_interpolated_the_short_way_across_south():
    h = _history(yaw0=math.radians(175), yaw_rate=math.radians(100))   # through 180 deg
    yaw = _px4(h, T0 + 105_000)[3]
    assert math.degrees(yaw) % 360 == pytest.approx(185.5)


def test_no_pose_outside_the_samples_or_across_a_gap():
    h = _history(n=10)
    assert _px4(h, T0 - 1_000) is None
    assert _px4(h, T0 + 95_000) is None      # after the last sample
    gappy = _history(n=10, step_us=200_000)              # 5 Hz: too far apart
    assert _px4(gappy, T0 + 300_000) is None


def test_no_pose_just_after_the_offset_changes():
    h = _history()
    # An update while converging, sent at T0 + 200 ms: from then on the samples
    # are sent with the new offset. 收敛过程中的一次更新：此后的样本按新偏移发出。
    h.set_offset(OFFSET + 30_000, T0 + 200_000)
    assert _px4(h, T0 + 220_000) is None                # less than 40 ms after
    assert 'offset changed' in h.why_not
    assert _px4(h, T0 + 250_000) is not None
    h.set_offset(OFFSET + 30_500, T0 + 300_000)           # under 1 ms: no change
    assert _px4(h, T0 + 310_000) is not None
    assert _px4(h, T0 + 125_000) is None                # before the change


def test_a_jump_of_the_sent_clock_starts_afresh():
    h = _history(n=20)
    # PX4 reset its offset estimate: the sent stamps jump 9.8 s ahead.
    later = T0 + 200_000 + 9_800_000
    for k in range(10):
        h.add_position(later + k * 10_000, 10.0, 0.0, -5.0)
        h.add_attitude(later + k * 10_000, _yaw_q(0.0))
    assert _px4(h, T0 + 100_000) is None      # before the jump: gone
    assert _px4(h, later + 45_000)[0] == pytest.approx(10.0)
    # And back again: stamps earlier than the last one also start afresh.
    h.add_position(T0 + 500_000, 3.0, 0.0, -5.0)
    h.add_position(T0 + 510_000, 3.0, 0.0, -5.0)
    h.add_attitude(T0 + 500_000, _yaw_q(0.0))
    h.add_attitude(T0 + 510_000, _yaw_q(0.0))
    assert _px4(h, T0 + 505_000)[0] == pytest.approx(3.0)


def test_no_pose_is_interpolated_across_an_estimator_reset():
    # EKF2 moved its estimate 4 m between two samples 20 ms apart; the vehicle
    # did not move. EKF2 在相隔 20 毫秒的两个样本之间把估计移动了 4 米，飞机并没有动。
    h = PoseHistory()
    h.set_offset(0, T0 - 1_000_000)
    for k, (x, reset) in enumerate([(0.0, 0), (0.0, 0), (4.0, 1), (4.0, 1)]):
        h.add_position(T0 + k * 20_000, x, 0.0, -5.0, reset=reset)
        h.add_attitude(T0 + k * 20_000, _yaw_q(0.0))
    assert h.pose_at(T0 + 30_000, COMPANION_CLOCK) is None     # not 2 m: never was there
    assert 'estimator reset' in h.why_not
    assert h.pose_at(T0 + 50_000, COMPANION_CLOCK)[0] == pytest.approx(4.0)
    # A heading reset likewise. 航向重置同样处理。
    for k, (yaw, reset) in enumerate([(0.0, 0), (0.0, 0), (1.0, 7), (1.0, 7)]):
        h.add_attitude(T0 + 100_000 + k * 20_000, _yaw_q(yaw), reset=reset)
        h.add_position(T0 + 100_000 + k * 20_000, 4.0, 0.0, -5.0, reset=1)
    assert h.pose_at(T0 + 130_000, COMPANION_CLOCK) is None


def test_the_frame_follows_resets_and_knows_when_it_cannot():
    f = FrameShift()
    assert not f.update(5, (9.0, 9.0), 3, 9.0)             # before: does not matter
    assert f.to_frame(10.0, 20.0, -5.0) == (10.0, 20.0, -5.0)
    assert f.update(6, (1.5, -0.5), 3, 9.0)                # an xy reset
    assert f.to_frame(11.5, 19.5, -5.0) == pytest.approx((10.0, 20.0, -5.0))
    assert not f.update(6, (1.5, -0.5), 3, 9.0)            # the same one again
    assert f.update(6, (1.5, -0.5), 4, 0.25)               # a z reset
    assert f.to_px4(10.0, 20.0, -5.0) == pytest.approx((11.5, 19.5, -4.75))
    assert f.missed == 0 and f.resets == 2
    f = FrameShift()
    f.update(255, (0.0, 0.0), 0, 0.0)
    f.update(0, (1.0, 0.0), 0, 0.0)                        # the uint8 counter wraps
    assert f.shift == pytest.approx((1.0, 0.0, 0.0)) and f.missed == 0
    f.update(2, (1.0, 0.0), 0, 0.0)                        # one reset not seen
    assert f.missed == 1
