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
from typing import Literal

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


class FakeJudge:
    """Offline stand-in: never pretends to have looked at the photo."""

    def judge(self, instruction: str, label: str, question: str, path: str) -> Verdict:
        return Verdict(label, path, 'unclear', 'low', '(offline) photo not judged')
