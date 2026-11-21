# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Planner tests with a fake Claude client: no network, no API key."""

import os
from types import SimpleNamespace

import pytest
from wukong_flight.skills import check_plan, GoTo, ReturnHome, Takeoff, TakePhoto

from wukong_mission import llm
from wukong_mission.planner import (ClaudePlanner, PlanningError, PlanStep, ScriptedPlanner,
                                    TaskPlan, to_skills)
from wukong_mission.world import load_world

WORLD = load_world(os.path.join(os.path.dirname(__file__), '..', 'config', 'sim_house.yaml'))


class FakeClient:
    """Mimics client.beta.messages.parse: returns queued answers, records requests."""

    def __init__(self, *answers, stop_reason='end_turn'):
        self.answers = list(answers)
        self.requests = []
        self.stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kw):
        self.requests.append(kw)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None,
                               parsed_output=self.answers.pop(0) if self.answers else None)


def step(skill, **kw):
    return PlanStep(skill=skill, **kw)


GOOD = TaskPlan(understood='photograph the south gutter', feasible=True, steps=[
    step('takeoff', altitude=4.5),
    step('goto', north=5.0, east=0.0, altitude=4.5, look_at_north=10.45, look_at_east=0.0),
    step('take_photo', label='south gutter', question='Is it blocked?'),
    step('return_home'),
])
# Flies straight through the house to the north side.
BAD = TaskPlan(understood='x', feasible=True, steps=[
    step('takeoff', altitude=4.5),
    step('goto', north=25.0, east=0.0, altitude=4.5),
    step('land'),
])


def test_world_map_matches_the_sim_house():
    (house,) = WORLD.houses
    box = house.box()
    assert (box.north_min, box.north_max, box.east_min, box.east_max) == (10.5, 19.5, -5.5, 5.5)
    assert 'south gutter' in WORLD.describe()


def test_good_plan_is_accepted_first_time():
    client = FakeClient(GOOD)
    plan = ClaudePlanner(WORLD, client=client).plan('check the gutter', (0.0, 0.0))
    assert plan.attempts == 1
    assert isinstance(plan.skills[0], Takeoff) and isinstance(plan.skills[-1], ReturnHome)
    assert plan.skills[1] == GoTo(5.0, 0.0, 4.5, (10.45, 0.0))
    assert plan.questions == {'south gutter': 'Is it blocked?'}
    req = client.requests[0]
    assert req['model'] == llm.DEFAULT_MODEL and req['fallbacks'] == 'default'
    assert 'check the gutter' in req['messages'][0]['content'][0]['text']


def test_rejected_plan_is_sent_back_once_with_the_errors():
    client = FakeClient(BAD, GOOD)
    plan = ClaudePlanner(WORLD, client=client).plan('check', (0.0, 0.0))
    assert plan.attempts == 2
    retry_text = client.requests[1]['messages'][0]['content'][0]['text']
    assert 'rejected by the pre-flight check' in retry_text and 'no-fly box house' in retry_text


def test_two_bad_plans_mean_no_flight():
    with pytest.raises(PlanningError):
        ClaudePlanner(WORLD, client=FakeClient(BAD, BAD)).plan('check', (0.0, 0.0))


def test_infeasible_task_means_no_flight():
    nope = TaskPlan(understood='paint the house', feasible=False,
                    reason_if_not='no painting skill', steps=[])
    with pytest.raises(PlanningError, match='no painting skill'):
        ClaudePlanner(WORLD, client=FakeClient(nope)).plan('paint it', (0.0, 0.0))


def test_refusal_is_an_llm_error():
    with pytest.raises(llm.LlmError, match='declined'):
        ClaudePlanner(WORLD, client=FakeClient(GOOD, stop_reason='refusal')).plan(
            'check', (0.0, 0.0))


def test_to_skills_reports_missing_fields():
    _, _, errors = to_skills([step('goto', north=1.0), step('take_photo')])
    assert any('missing east, altitude' in e for e in errors)
    assert any('missing label' in e for e in errors)
    _, _, errors = to_skills([step('goto', north=1, east=1, altitude=3, look_at_north=2)])
    assert any('both look_at' in e for e in errors)


def test_scripted_plan_photographs_both_gutters_and_passes_the_check():
    plan = ScriptedPlanner(WORLD).plan('whatever', (0.0, 0.0))
    photos = [s.label for s in plan.skills if isinstance(s, TakePhoto)]
    assert photos == ['south gutter', 'north gutter']
    assert check_plan(plan.skills, WORLD.limits, (0.0, 0.0)).ok
    # Each photo is taken facing the gutter it names.
    for i, s in enumerate(plan.skills):
        if isinstance(s, TakePhoto):
            before = plan.skills[i - 1]
            assert isinstance(before, GoTo) and before.look_at is not None


def test_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    env = tmp_path / '.env'
    env.write_text('# comment\nANTHROPIC_API_KEY=from-file\nOTHER_TEST_VAR="quoted"\n')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'from-env')
    monkeypatch.delenv('OTHER_TEST_VAR', raising=False)
    llm.load_dotenv(str(env))
    assert os.environ['ANTHROPIC_API_KEY'] == 'from-env'
    assert os.environ['OTHER_TEST_VAR'] == 'quoted'
