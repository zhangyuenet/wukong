# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
From one sentence to a landed drone and a report.

    ros2 run wukong_mission run_task "Check the gutters on the house in front of you"
    ros2 run wukong_mission run_task --offline "anything"     # no network, no API key

1. Read where the drone is.
2. Plan: Claude turns the sentence into skills; the plan must pass the pre-flight
   check (else: no takeoff).
3. Fly the plan. Each photo is judged in a worker thread while the drone flies on.
4. After landing, wait for the judgements and write report.md.
从一句话到降落和报告：读取位置 → 规划（不通过检查就不起飞）→ 飞行（照片在后台判断）
→ 落地后汇总报告。
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import functools
import os
import sys
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from wukong_flight.mission import MissionConfig, PlanRejected
from wukong_flight.mission_node import MissionNode, spin_mission
from wukong_flight.px4_link import Px4Link
from wukong_flight.skills import describe

from . import llm
from .judge import ClaudeJudge, FakeJudge, Verdict
from .planner import ClaudePlanner, Plan, PlanningError, ScriptedPlanner
from .report import summary_line, write_report
from .world import load_world


def _default_world() -> str:
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('wukong_mission'), 'config',
                        'sim_house.yaml')


def _default_out() -> str:
    root = os.environ.get('WUKONG_ROOT', os.getcwd())
    return os.path.join(root, 'log', 'missions', time.strftime('%Y%m%d-%H%M%S'))


def _parse_args(argv):
    p = argparse.ArgumentParser(prog='run_task', description=__doc__.split('\n\n')[0])
    p.add_argument('instruction', nargs='+', help='what to do, in plain language')
    p.add_argument('--offline', action='store_true',
                   help='scripted planner and fake judge: no network, no API key')
    p.add_argument('--world', default=None, help='map file (default: the sim house)')
    p.add_argument('--model', default=llm.DEFAULT_MODEL)
    p.add_argument('--effort', default='medium', choices=['low', 'medium', 'high'])
    p.add_argument('--out', default=None, help='output directory for photos and report')
    p.add_argument('--judge-timeout', type=float, default=120.0,
                   help='seconds to wait for photo judgements after landing')
    args = p.parse_args(argv)
    args.instruction = ' '.join(args.instruction)
    return args


def _wait_for_vehicle(timeout: float = 15.0) -> Optional[Tuple[float, float]]:
    """Return the current (north, east) in PX4's local frame, or None if PX4 is not there."""
    node = Node('run_task_probe')
    link = Px4Link(node)
    end = time.monotonic() + timeout
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            s = link.state()
            if s.connected and s.position_valid:
                return s.x, s.y
        return None
    finally:
        node.destroy_node()


def _collect(futures: List[Tuple[str, str, Future]], timeout: float) -> List[Verdict]:
    end = time.monotonic() + timeout
    verdicts = []
    for label, path, fut in futures:
        try:
            verdicts.append(fut.result(timeout=max(0.0, end - time.monotonic())))
        except Exception as e:   # noqa: B902 - timeout or a bug in a worker: report, go on
            note = 'timed out' if not fut.done() else f'failed: {e}'
            verdicts.append(Verdict(label, path, 'pending', 'low', f'judgement {note}'))
    return verdicts


def main(argv=None) -> int:
    args = _parse_args(remove_ros_args(sys.argv if argv is None else argv)[1:])
    out_dir = args.out or _default_out()
    world = load_world(args.world or _default_world())
    say = functools.partial(print, flush=True)   # keep order with ROS logs

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        # 1. Where are we? 读取飞机位置。
        start = _wait_for_vehicle()
        if start is None:
            say('[run_task] no vehicle position from PX4 (is ./scripts/sim.sh running?)')
            return 2
        say(f'[run_task] drone at north {start[0]:.1f}, east {start[1]:.1f}')

        # 2. Plan, on the ground. 在地面上完成规划。
        try:
            if args.offline:
                planner, judge = ScriptedPlanner(world), FakeJudge()
            else:
                planner = ClaudePlanner(world, args.model, args.effort)
                judge = ClaudeJudge(args.model, args.effort)
            say(f'[run_task] planning: "{args.instruction}"')
            t0 = time.monotonic()
            plan: Plan = planner.plan(args.instruction, start)
        except (llm.LlmError, PlanningError) as e:
            say(f'[run_task] no takeoff: {e}')
            write_report(out_dir, args.instruction, None, False, str(e), [])
            return 2
        except KeyboardInterrupt:
            say('[run_task] cancelled during planning, no takeoff')
            return 2
        say(f'[run_task] plan ready in {time.monotonic() - t0:.1f} s '
            f'(attempt {plan.attempts}): {plan.understood}')
        for i, s in enumerate(plan.skills, 1):
            say(f'           {i:2d}. {describe(s)}')

        # 3. Fly; judge photos in the background. 飞行，照片在后台判断。
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='judge')
        futures: List[Tuple[str, str, Future]] = []

        def on_photo(label: str, path: str) -> None:
            futures.append((label, path, pool.submit(
                judge.judge, args.instruction, label, plan.questions.get(label, ''), path)))

        try:
            node = MissionNode(plan.skills, MissionConfig(), world.limits, out_dir, on_photo,
                               name='run_task', frame='local', start=start)
        except PlanRejected as e:   # cannot happen after planning, but never fly unchecked
            say(f'[run_task] no takeoff: {e}')
            return 2
        flight_ok = spin_mission(node)
        note = node.mission.failure or ''
        node.destroy_node()

        # 4. Report. 汇总报告。
        if futures:
            say(f'[run_task] waiting for {len(futures)} photo judgement(s)...')
        try:
            verdicts = _collect(futures, args.judge_timeout)
        except KeyboardInterrupt:
            verdicts = _collect(futures, 0.0)
        pool.shutdown(wait=False, cancel_futures=True)
        md = write_report(out_dir, args.instruction, plan, flight_ok, note, verdicts)
        say(f'[run_task] {summary_line(verdicts)}')
        for v in verdicts:
            say(f'           - {v.label}: {v.needs_attention} ({v.confidence}) {v.findings}')
        say(f'[run_task] report: {md}')
        return 0 if flight_ok else 1
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
