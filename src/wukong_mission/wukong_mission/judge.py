# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Judge one inspection photo with a vision model.

Runs in a worker thread while the vehicle keeps flying: a slow or failed
judgement never touches the flight. It only changes the report.
在工作线程里运行，飞机照常飞行。判断变慢或失败只影响报告，不影响飞行。
"""

import base64
from dataclasses import dataclass, field
import io
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field

from . import llm


class PhotoVerdict(BaseModel):
    needs_attention: Literal['yes', 'no', 'unclear'] = Field(
        description='yes if what was asked about needs action (e.g. the gutter needs '
                    'cleaning), no if it clearly does not, unclear if the photo cannot tell')
    confidence: Literal['low', 'medium', 'high']
    findings: str = Field(description='What is visible that supports the answer, '
                                      'in one or two sentences')
    where: str = Field(description='Where in the photo the problem is (e.g. "left third, '
                                   'along the gutter"), or "" if none')


class HomeBoundary(BaseModel):
    found: bool = Field(description='true if the picture shows where one home ends and the '
                                    'next begins')
    at: Optional[float] = Field(
        None, description='where that boundary is, as a fraction of the picture width from '
                          'its left edge (0 = left edge, 1 = right edge); null if not found')
    confidence: Literal['low', 'medium', 'high']
    notes: str = Field(description='What in the picture marks the boundary, in one sentence')


# Only the wall counts: in Gazebo a gap between two clumps of debris of different
# colours on the gutter was once taken for a boundary. 只看墙面：Gazebo 中曾把水槽上
# 两团不同颜色杂物之间的空隙当成了两户分界。
BOUNDARY_QUESTION = (
    'This picture shows part of a row of joined homes (townhouses): a stretch of their '
    'roof edge with the gutter, and below it the top of the wall. {home} Is there a place '
    'in this picture where one home ends and the next begins? Look only at the wall below '
    'the gutter: its colour or material changes there, or a vertical divider, trim or '
    'downspout runs down the wall between two homes. Anything lying on the gutter or the '
    'roof (leaves, debris, clumps of any colour) is NOT a boundary, and the roof may run on '
    'unbroken. If there is more than one boundary, give the one nearest the home being '
    'inspected. Answer found=false if the wall shows no such change, or if too little of '
    'the wall is visible to tell.')


def home_hint(position: float) -> str:
    """Tell the model where the inspected home is, given its position across the picture."""
    if position < 0.1:
        return 'The home being inspected is at the left edge of this picture or beyond it.'
    if position > 0.9:
        return 'The home being inspected is at the right edge of this picture or beyond it.'
    return (f'The home being inspected is at about {position:.0%} of the way across this '
            f'picture from the left.')


@dataclass
class Verdict:
    label: str
    path: str
    needs_attention: str          # yes / no / unclear / error
    confidence: str
    findings: str
    where: str = ''
    extra: dict = field(default_factory=dict)   # e.g. which gutter segment it shows


SYSTEM_PROMPT = """\
You review inspection photos taken by a small drone for a homeowner. Answer \
only from what is visible in the photo. If the photo does not show the thing \
asked about clearly enough, say unclear rather than guessing. The photo may \
show parts of the drone itself (propellers at the top edge); ignore them.
"""

MAX_SIDE = 1568   # px; larger images are downscaled by the API anyway


def _jpeg_b64(path: str) -> str:
    from PIL import Image
    with Image.open(path) as img:
        img = img.convert('RGB')
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode('ascii')


class ClaudeJudge:

    def __init__(self, model: str = llm.DEFAULT_MODEL, effort: str = 'medium',
                 client=None) -> None:
        self.model = model
        self.effort = effort
        self.client = client if client is not None else llm.make_client()

    def judge(self, instruction: str, label: str, question: str, path: str) -> Verdict:
        text = (f'The homeowner asked: "{instruction}"\n'
                f'This photo shows: {label}.\n'
                f'Question: {question or "Does this need attention?"}')
        content = [
            {'type': 'image',
             'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': _jpeg_b64(path)}},
            {'type': 'text', 'text': text},
        ]
        try:
            v: PhotoVerdict = llm.parse(self.client, self.model, SYSTEM_PROMPT, content,
                                        PhotoVerdict, effort=self.effort, max_tokens=4000)
        except llm.LlmError as e:
            return Verdict(label, path, 'error', 'low', f'no judgement: {e}')
        return Verdict(label, path, v.needs_attention, v.confidence, v.findings, v.where)

    def find_boundary(self, path: str, home_position: float) -> Tuple[Optional[float], str]:
        """
        Where one home of a row ends in this picture: (fraction across, notes).

        None when no boundary is visible, the answer is unsure, or the call failed.
        找出照片里两户的分界（占画面宽度的比例）；看不到、不确定或调用失败时为 None。
        """
        content = [
            {'type': 'image',
             'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': _jpeg_b64(path)}},
            {'type': 'text', 'text': BOUNDARY_QUESTION.format(home=home_hint(home_position))},
        ]
        try:
            b: HomeBoundary = llm.parse(self.client, self.model, SYSTEM_PROMPT, content,
                                        HomeBoundary, effort=self.effort, max_tokens=4000)
        except llm.LlmError as e:
            return None, f'no answer: {e}'
        if not b.found or b.at is None or b.confidence == 'low':
            return None, b.notes
        return min(1.0, max(0.0, float(b.at))), b.notes


class FakeJudge:
    """Offline stand-in: never pretends to have looked at the photo."""

    def judge(self, instruction: str, label: str, question: str, path: str) -> Verdict:
        return Verdict(label, path, 'unclear', 'low', '(offline) photo not judged')

    def find_boundary(self, path: str, home_position: float) -> Tuple[Optional[float], str]:
        return None, '(offline) photo not looked at'
