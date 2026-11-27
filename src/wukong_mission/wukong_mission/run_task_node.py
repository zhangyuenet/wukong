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
import dataclasses
import functools
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from wukong_flight.mission import MissionConfig, PlanRejected
from wukong_flight.mission_node import MissionNode, spin_mission
from wukong_flight.px4_link import Px4Link
from wukong_flight.skills import describe

from . import llm
from .classify import Classification, ClaudeClassifier, TruthClassifier
from .house_task import HouseTask
from .judge import ClaudeJudge, FakeJudge, Verdict
from .planner import (ClaudePlanner, GutterPlanner, OfflineGutterPlanner, Plan, PlanningError,
                      ScriptedPlanner)
from .report import summary_line, write_report
from .world import load_world


def _default_world() -> str:
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('wukong_mission'), 'config',
                        'sim_house.yaml')


def _running_world() -> Optional[str]:
    """Name of the Gazebo world being simulated, or None (no Gazebo, or not found)."""
    import subprocess
    try:
        out = subprocess.run(['gz', 'topic', '-l'], capture_output=True, text=True,
                             timeout=8).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        parts = line.strip().split('/')
        if len(parts) == 4 and parts[1] == 'world' and parts[3] == 'clock':
            return parts[2]
    return None


def _choose_map(args, running: Optional[str], say) -> Tuple[Optional[str], str]:
    """
    Pick the map that belongs to the simulated world; refuse a mismatch.

    Returns (map path, problem). Without --world/--scenario the running world
    decides: wukong_<scenario> uses sim/generated/<scenario>/map.yaml, anything
    else the default sim house map.
    地图必须与正在仿真的世界一致；没有指定时按正在运行的世界自动选择。
    """
    root = os.environ.get('WUKONG_ROOT', '.')

    def scenario_map(name):
        return os.path.join(root, 'sim', 'generated', name, 'map.yaml')
    if args.scenario:
        path, expect = scenario_map(args.scenario), f'wukong_{args.scenario}'
    elif args.world:
        path, expect = args.world, None
    elif running and running.startswith('wukong_') and os.path.isfile(
            scenario_map(running[len('wukong_'):])):
        path, expect = scenario_map(running[len('wukong_'):]), running
        say(f'[run_task] simulated world is {running}: using {path}')
    else:
        path, expect = _default_world(), 'wukong_house'
    if not os.path.isfile(path):
        return None, f'map not found: {path}'
    if running and expect and running != expect:
        return None, (f'the map is for world {expect}, but the simulation runs {running}. '
                      f'Restart the simulation with that world, or drop --scenario to use '
                      f'the map of the running one.')
    return path, ''


def _default_out() -> str:
    root = os.environ.get('WUKONG_ROOT', os.getcwd())
    return os.path.join(root, 'log', 'missions', time.strftime('%Y%m%d-%H%M%S'))


def _parse_args(argv):
    p = argparse.ArgumentParser(prog='run_task', description=__doc__.split('\n\n')[0])
    p.add_argument('instruction', nargs='+', help='what to do, in plain language')
    p.add_argument('--offline', action='store_true',
                   help='scripted planner and fake judge: no network, no API key')
    p.add_argument('--world', default=None, help='map file (default: the sim house)')
    p.add_argument('--scenario', default=None,
                   help='use the rough map of sim scenario NAME (sim/generated/NAME/map.yaml)')
    p.add_argument('--model', default=llm.DEFAULT_MODEL)
    p.add_argument('--effort', default='medium', choices=['low', 'medium', 'high'])
    p.add_argument('--out', default=None, help='output directory for photos and report')
    p.add_argument('--judge-timeout', type=float, default=120.0,
                   help='seconds to wait for photo judgements after landing')
    p.add_argument('--seen-free', action='store_true',
                   help='fly the survey and photos only through space seen free (M4 class A)')
    args = p.parse_args(argv)
    args.instruction = ' '.join(args.instruction)
    return args


def _wait_for_vehicle(timeout: float = 15.0) -> Optional[Tuple[float, float, float]]:
    """Return (north, east, heading) in PX4's local frame, or None if PX4 is not there."""
    node = Node('run_task_probe')
    link = Px4Link(node)
    end = time.monotonic() + timeout
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            s = link.state()
            if s.connected and s.position_valid:
                return s.x, s.y, s.heading
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


def _run_gutters(args, world, start, facing, out_dir, say) -> int:
    """Find and survey the house in flight, then photograph and judge every gutter segment."""
    try:
        if args.offline:
            planner, judge = OfflineGutterPlanner(world), FakeJudge()
        else:
            planner = GutterPlanner(world, args.model)
            judge = ClaudeJudge(args.model, args.effort)
        say(f'[run_task] understanding: "{args.instruction}"')
        request = planner.request(args.instruction)
        # One flow for every map: the house is found in flight, near its rough
        # position if the map gives one. 所有地图同一个流程：在飞行中找到房屋，
        # 地图给了大概位置时就在那附近找。
        task = HouseTask(world, request, start, out_dir, judge, args.instruction, log=say,
                         facing=facing)
        if args.seen_free:
            task.survey = dataclasses.replace(task.survey, seen_free_only=True)
            say('[run_task] flying only through space seen free')
        if not args.offline:       # judge the kind of house from a photo 根据照片判断房屋类型
            task.classifier = ClaudeClassifier(args.model)
        else:                      # sim only: the scenario's truth stands in for Claude
            task.classifier = TruthClassifier.beside(args.map_path)
            if task.classifier:
                say('[run_task] offline: house type from the simulation truth')
    except (llm.LlmError, PlanningError, ValueError) as e:
        say(f'[run_task] no takeoff: {e}')
        write_report(out_dir, args.instruction, None, False, str(e), [])
        return 2
    except KeyboardInterrupt:
        say('[run_task] cancelled during planning, no takeoff')
        return 2
    sides = ', '.join(request.sides) or 'all'
    say(f'[run_task] task: {request.understood} (house "{request.house}", gutters: {sides})')
    skills = task.plan()
    say('[run_task] first steps; the rest is planned in flight from what the drone sees:')
    for i, s in enumerate(skills, 1):
        say(f'           {i:2d}. {describe(s)}')
    try:
        node = MissionNode(skills, MissionConfig(), world.limits, out_dir, name='run_task',
                           frame='local', start=start)
    except PlanRejected as e:
        say(f'[run_task] no takeoff: {e}')
        return 2
    task.attach(node)
    flight_ok = spin_mission(node)
    mission = node.mission
    _save_flight_frames(task.depth, out_dir, say)
    note = mission.failure or ''
    for label, accepted, errors in mission.decisions:
        if not accepted:
            note = (note + '; ' if note else '') + f'{label}: ' + '; '.join(errors[:3])
    node.destroy_node()

    if task.pending():
        say(f'[run_task] waiting for {task.pending()} photo judgement(s)...')
    try:
        verdicts = task.collect(args.judge_timeout)
    except KeyboardInterrupt:
        verdicts = task.collect(0.0)
    # Segments the drone could not inspect safely belong in the report too.
    # 无法安全检查的段也写进报告。
    for sk in getattr(task, 'skipped', []):
        if sk.segment:
            verdicts.append(Verdict(sk.label, '', 'skipped', 'low', f'not inspected: {sk.reason}',
                                    extra={'gutter': sk.gutter, 'segment': sk.segment,
                                           'segments': sk.segment, 'from': list(sk.start),
                                           'to': list(sk.end)}))
    # Planned photos never taken (the flight ended early) were not inspected either.
    # 计划了但没拍到的段（飞行提前结束）同样写成“未检查”。
    seen = {v.label for v in verdicts}
    for label, t in getattr(task, 'targets', {}).items():
        if label not in seen:
            verdicts.append(Verdict(label, '', 'skipped', 'low',
                                    'not inspected: the flight ended before this photo',
                                    extra={'gutter': t.gutter, 'segment': t.segment,
                                           'segments': t.segments, 'from': list(t.start),
                                           'to': list(t.end), 'height': t.height}))
    plan = Plan(understood_with_house(request.understood, task), list(mission.plan))
    md = write_report(out_dir, args.instruction, plan, flight_ok, note, verdicts)
    say(f'[run_task] {summary_line(verdicts)}')
    say(f'[run_task] report: {md}')
    return 0 if flight_ok and mission.decisions and mission.decisions[-1][1] else 1


def _save_flight_frames(depth, out_dir: str, say) -> None:
    """Log the depth frames taken in flight and save their poses, to check against the truth."""
    poses = getattr(depth, 'flight_poses', None)
    if poses is None:
        return
    say(f'[run_task] depth frames in flight: {depth.describe_flight()}')
    if poses:
        # poses: (stamp s, north, east, altitude, heading, roll, pitch) per frame;
        # points: some of its hit points, frame: which frame each is from.
        # 每帧的（时间戳, 北, 东, 高度, 航向, 横滚, 俯仰），以及部分击中点和所属的帧。
        pts = depth.flight_points
        np.savez_compressed(
            os.path.join(out_dir, 'flight_frames.npz'), poses=np.array(poses),
            points=np.vstack(pts) if pts else np.zeros((0, 3), np.float32),
            frame=np.concatenate([np.full(len(p), k, np.int32) for k, p in enumerate(pts)]))


def understood_with_house(understood: str, task: HouseTask) -> str:
    """
    Return what the drone took the task to be, with what kind of house it judged it.

    Nothing to add if it never judged it (task.house is then None, or, in
    the M2 flow, was a RoughHouse, which once crashed the report).
    没有判断过房屋类型时不补充（task.house 为 None；M2 流程里曾是 RoughHouse，
    写报告时因此崩溃过）。
    """
    house = getattr(task, 'house', None)
    if isinstance(house, Classification):
        return f'{understood} House: {house.describe()}. {house.notes}'
    return understood


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(remove_ros_args(sys.argv if argv is None else argv)[1:])
    out_dir = args.out or _default_out()
    os.makedirs(out_dir, exist_ok=True)
    say = functools.partial(print, flush=True)   # keep order with ROS logs
    map_path, problem = _choose_map(args, _running_world(), say)
    if problem:
        say(f'[run_task] {problem}')
        return 2
    args.map_path = map_path
    world = load_world(map_path)
    if not world.houses and not world.rough_houses and not world.find_house:
        say('[run_task] this map names no house and no search radius: nothing to inspect. '
            'No takeoff.')
        return 2

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        # 1. Where are we? 读取飞机位置。
        found = _wait_for_vehicle()
        if found is None:
            say('[run_task] no vehicle position from PX4 (is ./scripts/sim.sh running?)')
            return 2
        start, facing = found[:2], found[2]
        say(f'[run_task] drone at north {start[0]:.1f}, east {start[1]:.1f}, '
            f'facing {math.degrees(facing) % 360:.0f} deg')
        if world.rough_houses or world.find_house:
            return _run_gutters(args, world, start, facing, out_dir, say)

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

        def on_photo(label: str, path: str, pose) -> None:
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
