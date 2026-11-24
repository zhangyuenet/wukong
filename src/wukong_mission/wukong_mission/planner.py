# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
From one sentence to a checked plan of skills.

The large model only proposes. Its answer is converted to wukong_flight skills
and must pass check_plan(); if it does not, the errors go back to the model
once. A plan that still fails is not flown. Planning happens on the ground,
before arming, so a slow or unreachable cloud can only delay the takeoff.
大模型只负责提出计划。计划必须通过 check_plan()，不通过就把错误反馈给模型重试一次，
仍不通过就不起飞。规划在起飞前完成，云端慢或失联只会推迟起飞。
"""

from dataclasses import dataclass, field
import json
import math
from typing import List, Literal, Optional, Sequence, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field
from wukong_flight.skills import (check_plan, GoTo, Hover, Land, ReturnHome, Skill, Takeoff,
                                  TakePhoto)

from . import llm
from .world import WorldMap

if TYPE_CHECKING:
    import anthropic

    from .gutter_task import GutterRequest


# -- what the model fills in ---------------------------------------------------

class PlanStep(BaseModel):
    """One skill. Only the fields that skill uses are filled; the rest are null."""

    skill: Literal['takeoff', 'goto', 'hover', 'take_photo', 'return_home', 'land']
    north: Optional[float] = Field(None, description='goto: target north, metres')
    east: Optional[float] = Field(None, description='goto: target east, metres')
    altitude: Optional[float] = Field(
        None, description='takeoff, goto: metres above the takeoff point')
    look_at_north: Optional[float] = Field(
        None, description='goto: face this point on arrival (north), or null')
    look_at_east: Optional[float] = Field(
        None, description='goto: face this point on arrival (east), or null')
    seconds: Optional[float] = Field(None, description='hover: duration')
    label: Optional[str] = Field(None, description='take_photo: short name, e.g. "south gutter"')
    question: Optional[str] = Field(
        None, description='take_photo: what the photo should be checked for')


class TaskPlan(BaseModel):
    understood: str = Field(description='The task, restated in one sentence')
    feasible: bool = Field(description='False if the skills and map cannot do this task')
    reason_if_not: Optional[str] = None
    steps: List[PlanStep]


# -- what the rest of Wukong gets ------------------------------------------------

@dataclass
class Plan:
    understood: str
    skills: List[Skill]
    questions: dict = field(default_factory=dict)   # photo label -> what to check
    attempts: int = 1


class PlanningError(RuntimeError):
    """No acceptable plan. The vehicle must not take off."""


def to_skills(steps: Sequence[PlanStep]) -> Tuple[List[Skill], dict, List[str]]:
    """Convert model steps to skills. Returns (skills, photo questions, errors)."""
    skills: List[Skill] = []
    questions = {}
    errors: List[str] = []

    def need(i, step, *names):
        missing = [n for n in names if getattr(step, n) is None]
        if missing:
            errors.append(f'step {i} ({step.skill}): missing {", ".join(missing)}')
        return not missing

    for i, st in enumerate(steps, 1):
        if st.skill == 'takeoff' and need(i, st, 'altitude'):
            skills.append(Takeoff(st.altitude))
        elif st.skill == 'goto' and need(i, st, 'north', 'east', 'altitude'):
            look = None
            if (st.look_at_north is None) != (st.look_at_east is None):
                errors.append(f'step {i} (goto): give both look_at_north and look_at_east')
            elif st.look_at_north is not None:
                look = (st.look_at_north, st.look_at_east)
            skills.append(GoTo(st.north, st.east, st.altitude, look))
        elif st.skill == 'hover' and need(i, st, 'seconds'):
            skills.append(Hover(st.seconds))
        elif st.skill == 'take_photo' and need(i, st, 'label'):
            if st.label in questions:
                errors.append(f'step {i} (take_photo): label "{st.label}" is used twice')
            questions[st.label] = st.question or ''
            skills.append(TakePhoto(st.label))
        elif st.skill == 'return_home':
            skills.append(ReturnHome())
        elif st.skill == 'land':
            skills.append(Land())
    for s in skills:
        if any(isinstance(v, float) and not math.isfinite(v) for v in vars(s).values()):
            errors.append(f'non-finite number in {s}')
    return skills, questions, errors


SYSTEM_PROMPT = """\
You plan flights for Wukong, a small quadcopter that inspects houses. You turn \
the user's request into a list of skills. The plan is checked by software \
before takeoff and executed exactly as written; nobody will fix it in flight.

Skills:
- takeoff(altitude): climb straight up from the takeoff point. Always first.
- goto(north, east, altitude, look_at_north, look_at_east): fly in a straight \
line to the point; on arrival turn to face the look_at point (or keep the \
current heading if look_at is null).
- hover(seconds)
- take_photo(label, question): wait until still, then take one photo with the \
forward camera. The label names what is in the photo; the question says what \
to check in it. Photos are judged by a vision model after they are taken.
- return_home: fly back above the takeoff point at the current altitude and land.
- land: land where the vehicle is.
The plan must end with return_home (preferred) or land.

Taking good photos: the camera is level and cannot tilt. To see into a gutter, \
fly somewhat above it, so the camera looks slightly down on it, and far enough \
back that the whole gutter fits in the horizontal field of view. Face the \
middle of the gutter. Choose the viewpoint on the side the gutter is on, \
outside the safety margin.

Moving between viewpoints: paths are straight lines and must keep the safety \
margin from every house, including above the ridge. To get to the other side \
of a house, go around a corner with intermediate goto steps, or climb above \
the ridge plus the margin first.

If the request cannot be done with these skills and this map, set feasible to \
false, explain why in reason_if_not, and give an empty step list.
"""


class ClaudePlanner:
    """Plans with Claude (structured output), retrying once with the check's errors."""

    def __init__(self, world: WorldMap, model: str = llm.DEFAULT_MODEL,
                 effort: str = 'medium', client: Optional['anthropic.Anthropic'] = None) -> None:
        self.world = world
        self.model = model
        self.effort = effort
        self.client = client if client is not None else llm.make_client()

    def plan(self, instruction: str, start: Tuple[float, float]) -> Plan:
        context = (f'{self.world.describe()}\n\nThe drone is on the ground at '
                   f'({start[0]:.1f}, {start[1]:.1f}); that is the takeoff point.\n\n'
                   f'User request: {instruction}')
        feedback = ''
        for attempt in (1, 2):
            text = context + feedback
            answer: TaskPlan = llm.parse(self.client, self.model, SYSTEM_PROMPT,
                                         [{'type': 'text', 'text': text}], TaskPlan,
                                         effort=self.effort)
            if not answer.feasible:
                raise PlanningError(f'the model says this cannot be done: '
                                    f'{answer.reason_if_not or "no reason given"}')
            skills, questions, errors = to_skills(answer.steps)
            if not errors:
                errors = check_plan(skills, self.world.limits, start).errors
            if not errors:
                return Plan(answer.understood, skills, questions, attempt)
            feedback = ('\n\nYour previous plan was rejected by the pre-flight check:\n'
                        + json.dumps([s.model_dump(exclude_none=True) for s in answer.steps])
                        + '\nErrors:\n- ' + '\n- '.join(errors)
                        + '\nWrite a corrected plan.')
        raise PlanningError('no plan passed the pre-flight check: ' + '; '.join(errors))


class ScriptedPlanner:
    """
    Offline stand-in: photograph every gutter on the map, then return home.

    It ignores the wording of the request. Used for tests and for flying the
    whole pipeline without network or API costs.
    离线替身：不理解指令，按地图给每条水槽拍一张照片后返航。用于测试和无网络运行。
    """

    def __init__(self, world: WorldMap, standoff: float = 5.5, above_gutter: float = 2.0):
        self.world = world
        self.standoff = standoff
        self.above = above_gutter

    def plan(self, instruction: str, start: Tuple[float, float]) -> Plan:
        skills: List[Skill] = []
        questions = {}
        alt = 0.0
        for house in self.world.houses:
            box = house.box()
            clear = box.top + self.world.limits.margin + 1.0   # transit height
            for g in house.gutters:
                mid_n, mid_e = g.midpoint
                # Stand off on the outside of the gutter, away from the house centre.
                side = 1.0 if mid_n >= house.center[0] else -1.0
                view_n = mid_n + side * self.standoff
                alt = g.height + self.above
                if not skills:
                    skills.append(Takeoff(alt))
                else:
                    # Climb over the roof, then descend to the next viewpoint.
                    skills.append(GoTo(*_last_xy(skills, start), clear))
                    skills.append(GoTo(view_n, mid_e, clear))
                skills.append(GoTo(view_n, mid_e, alt, look_at=(mid_n, mid_e)))
                skills.append(TakePhoto(g.name))
                questions[g.name] = f'Is the {g.name} blocked by leaves or debris?'
        if not skills:
            skills.append(Takeoff(3.0))
        else:
            skills.append(GoTo(*_last_xy(skills, start), clear))
        skills.append(ReturnHome())
        return Plan(f'(offline) photograph every gutter on the map: {instruction}',
                    skills, questions)


def _last_xy(skills, start):
    for s in reversed(skills):
        if isinstance(s, GoTo):
            return s.north, s.east
    return start


# -- M2: gutters of a roughly known house ---------------------------------------

Side = Literal['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west',
               'northwest']


class GutterTaskSpec(BaseModel):
    understood: str = Field(description='The task, restated in one sentence')
    feasible: bool = Field(description='False if this is not a gutter inspection of a '
                                       'house on the map')
    reason_if_not: Optional[str] = None
    house: str = Field(description='Name of the house on the map, or "nearest" when the '
                                   'map has no house positions')
    sides: List[Side] = Field(
        description='Only inspect gutters facing these directions; empty list = all gutters')


GUTTER_SYSTEM_PROMPT = """\
You turn a homeowner's request into a gutter inspection task for Wukong, a small \
quadcopter. The drone knows only roughly where each house is. In flight it scans \
the house with a depth camera, finds the eaves, and photographs every gutter \
segment; a vision model then judges each segment. You do not plan coordinates: \
pick the house and, if the user asked for particular sides, the compass \
directions those gutters face. Leave sides empty for "all gutters" or when the \
user gives no direction. Words like "front" or "back" have no direction on this \
map: leave sides empty and mention it in understood.

If the request is not about inspecting gutters of a house on the map, set \
feasible to false and explain why in reason_if_not.
"""


class GutterPlanner:
    """Claude reads the request; geometry is done in flight by code."""

    def __init__(self, world: WorldMap, model: str = llm.DEFAULT_MODEL,
                 effort: str = 'low', client: Optional['anthropic.Anthropic'] = None) -> None:
        self.world = world
        self.model, self.effort = model, effort
        self.client = client if client is not None else llm.make_client()

    def request(self, instruction: str) -> 'GutterRequest':
        from .gutter_task import GutterRequest
        text = f'{self.world.describe()}\n\nUser request: {instruction}'
        spec: GutterTaskSpec = llm.parse(self.client, self.model, GUTTER_SYSTEM_PROMPT,
                                         [{'type': 'text', 'text': text}], GutterTaskSpec,
                                         effort=self.effort, max_tokens=4000)
        if not spec.feasible:
            raise PlanningError(f'the model says this cannot be done: '
                                f'{spec.reason_if_not or "no reason given"}')
        if self.world.find_house:       # M3: the house is whatever is nearest, found in flight
            return GutterRequest(spec.understood, 'nearest', list(spec.sides))
        names = [h.name for h in self.world.rough_houses]
        if spec.house not in names:
            raise PlanningError(f'the model picked "{spec.house}", which is not on the map '
                                f'({", ".join(names)})')
        return GutterRequest(spec.understood, spec.house, list(spec.sides))


class OfflineGutterPlanner:
    """Offline stand-in: all gutters of the first house on the map."""

    def __init__(self, world: WorldMap) -> None:
        self.world = world

    def request(self, instruction: str) -> 'GutterRequest':
        from .gutter_task import GutterRequest
        if self.world.find_house:
            return GutterRequest(f'(offline) inspect all gutters of the nearest building: '
                                 f'{instruction}', 'nearest', [])
        if not self.world.rough_houses:
            raise PlanningError('no house on the map')
        h = self.world.rough_houses[0]
        return GutterRequest(f'(offline) inspect all gutters of {h.name}: {instruction}',
                             h.name, [])
