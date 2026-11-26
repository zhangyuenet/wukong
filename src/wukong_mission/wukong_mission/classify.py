# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
What kind of house is this? One photo, a ruler drawn on it, and a vision model.

Geometry finds the nearest building but cannot tell a townhouse row from one
long detached house, nor where one unit ends. A photo can. The drone draws a
ruler along the building's front (a tick every metre, numbered) and a line where
it is looking, and asks which kind of house it is and, for a row, which metres
of the ruler belong to the unit in front of it.
几何能找到最近的建筑，但分不清联排和一栋很长的独立屋，也看不出户界。照片可以。
在照片上沿建筑正面画一把每米一格的刻度尺，再画出无人机正对的位置，让视觉模型判断。

The answer only chooses which part of the building to inspect. Safety (no-fly
zones, paths) never depends on it. 判断结果只决定检查哪一部分，不影响安全。
"""

from dataclasses import dataclass
import math
from typing import Literal, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field
from wukong_perception.geometry import project, RGB_X500_DEPTH

from . import llm

if TYPE_CHECKING:
    import anthropic
    from wukong_perception.heightmap import Rectangle

Point = Tuple[float, float]


class HouseKind(BaseModel):
    kind: Literal['detached', 'townhouse_row', 'not_a_house'] = Field(
        description='detached: one house on its own (a garage or porch attached to it still '
                    'counts as detached). townhouse_row: several homes joined side by side '
                    'sharing walls. not_a_house: anything else.')
    storeys: int = Field(description='Storeys above ground of the house in front of the '
                                     'camera (1, 2 or 3)')
    garage: Literal['attached', 'none', 'not_visible']
    units: Optional[int] = Field(None, description='townhouse_row only: how many homes in the '
                                                   'row are visible')
    unit_from_m: Optional[float] = Field(
        None, description='townhouse_row only: ruler reading (metres) where the home under '
                          'the red line begins')
    unit_to_m: Optional[float] = Field(
        None, description='townhouse_row only: ruler reading (metres) where that home ends')
    end_unit: Optional[bool] = Field(
        None, description='townhouse_row only: true if that home is at an end of the row')
    confidence: Literal['low', 'medium', 'high']
    notes: str = Field(description='What in the photo supports the answer, in one sentence')


@dataclass
class Classification:
    kind: str
    storeys: int
    garage: str
    unit: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]   # (north, east) ends
    end_unit: Optional[bool]
    confidence: str
    notes: str
    source: str                       # 'site' (confirmed), 'model', or 'default'

    def describe(self) -> str:
        if self.kind == 'townhouse_row':
            where = 'an end unit' if self.end_unit else 'a middle unit'
            return (f'townhouse row, {where}, {self.storeys} storey(s), garage: {self.garage} '
                    f'({self.source}, {self.confidence} confidence)')
        return (f'{self.kind}, {self.storeys} storey(s), garage: {self.garage} '
                f'({self.source}, {self.confidence} confidence)')


SYSTEM_PROMPT = """\
You look at a photo a small drone took of the house it is about to inspect, \
from in front of it and a little above. A ruler has been drawn along the front \
of the building: yellow ticks every metre, numbered every two metres. The red \
vertical line marks where the drone is looking: the home that line falls on is \
the one to inspect. Answer from what is visible only.

For a townhouse row, find where that home begins and ends along the ruler: \
the boundaries between homes are usually where walls change colour or \
material, at downspouts, at a step in the roof, or halfway between two \
front doors. Read the boundaries off the ruler as precisely as you can.
"""


def annotate(photo: str, front: Tuple[Tuple[float, float], Tuple[float, float]],
             height: float, pose: Sequence[float], out: str) -> Optional[str]:
    """
    Draw the ruler along the building front and the line of sight; return out.

    front: the front's two ends (north, east); the ruler reads 0 at front[0].
    height: where to draw it (m, the vehicle's altitude frame).
    返回带刻度尺的照片路径；刻度尺在 front[0] 处读数为 0。
    """
    from PIL import Image, ImageDraw
    a, b = np.array(front[0], float), np.array(front[1], float)
    length = float(np.linalg.norm(b - a))
    if length < 1.0:
        return None
    with Image.open(photo) as img:
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    ticks = []
    for m in range(int(length) + 1):
        p = a + (b - a) * (m / length)
        u, v, d = project(np.array([[p[0], p[1], height]]), RGB_X500_DEPTH, pose)[0]
        if d > 0.5 and -50 <= u <= w + 50:
            ticks.append((m, u, v))
    if len(ticks) < 2:
        return None
    draw.line([(u, v) for _, u, v in ticks], fill=(255, 220, 0), width=4)
    for m, u, v in ticks:
        big = m % 2 == 0
        draw.line([(u, v - (22 if big else 12)), (u, v + (22 if big else 12))],
                  fill=(255, 220, 0), width=4 if big else 2)
        if big:
            draw.rectangle([u - 16, v + 26, u + 16, v + 56], fill=(0, 0, 0))
            draw.text((u - 10, v + 30), str(m), fill=(255, 220, 0))
    draw.line([(w / 2, 0), (w / 2, h)], fill=(255, 0, 0), width=3)
    img.save(out, quality=92)
    return out


class _ClassifyHouse:
    """Shared by the real classifier: annotate, ask, convert."""

    def classify_house(self, photo: str, pose: Sequence[float], front: Tuple[Point, Point],
                       height: float, out_dir: str) -> 'Classification':
        import os
        out = os.path.join(out_dir, 'house_overview_ruler.jpg')
        annotated = annotate(photo, front, height, pose, out)
        if annotated is None:
            raise ValueError('the building front is not in the photo')
        return from_answer(self.classify(annotated), front)


class ClaudeClassifier(_ClassifyHouse):

    def __init__(self, model: str = llm.DEFAULT_MODEL, effort: str = 'medium',
                 client: Optional['anthropic.Anthropic'] = None) -> None:
        self.model, self.effort = model, effort
        self.client = client if client is not None else llm.make_client()

    def classify(self, annotated: str) -> HouseKind:
        from .judge import _jpeg_b64
        content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                'data': _jpeg_b64(annotated)}},
                   {'type': 'text', 'text': 'What kind of house is the drone looking at?'}]
        return llm.parse(self.client, self.model, SYSTEM_PROMPT, content, HouseKind,
                         effort=self.effort, max_tokens=4000)


def from_answer(ans: HouseKind, front: Tuple[Point, Point],
                source: str = 'model') -> Classification:
    """Turn ruler readings into the unit's ends on the ground (north, east)."""
    unit = None
    if (ans.kind == 'townhouse_row' and ans.unit_from_m is not None
            and ans.unit_to_m is not None):
        a, b = np.array(front[0], float), np.array(front[1], float)
        length = float(np.linalg.norm(b - a))
        lo, hi = sorted((ans.unit_from_m, ans.unit_to_m))
        if hi - lo >= 2.0 and length > 0:
            pa = a + (b - a) * (max(lo, -2.0) / length)
            pb = a + (b - a) * (min(hi, length + 2.0) / length)
            unit = ((round(float(pa[0]), 2), round(float(pa[1]), 2)),
                    (round(float(pb[0]), 2), round(float(pb[1]), 2)))
    return Classification(ans.kind, ans.storeys, ans.garage, unit, ans.end_unit,
                          ans.confidence, ans.notes, source)


def front_of(rect: 'Rectangle', viewer: Point) -> Tuple[Point, Point]:
    """Return the rectangle's side facing the viewer, ends ordered left to right as seen."""
    hu, hv = rect.length / 2, rect.width / 2
    sides = [((hu, -hv), (hu, hv)), ((-hu, hv), (-hu, -hv)),
             ((-hu, -hv), (hu, -hv)), ((hu, hv), (-hu, hv))]
    best = min(sides, key=lambda s: math.dist(rect.to_ne(*[(s[0][k] + s[1][k]) / 2
                                                           for k in range(2)]), viewer))
    p, q = rect.to_ne(*best[0]), rect.to_ne(*best[1])
    # Left to right as seen from the viewer: cross product sign. 从观察者看从左到右。
    d = (q[0] - p[0], q[1] - p[1])
    to_p = (p[0] - viewer[0], p[1] - viewer[1])
    if to_p[0] * d[1] - to_p[1] * d[0] < 0:
        p, q = q, p
    return p, q


class TruthClassifier:
    """
    Sim only: answers what a perfect look at the photo would, from the scenario's truth.

    Stands in for Claude offline, as FakeJudge does for the photo judgements.
    仅用于仿真：离线时用场景真值代替 Claude 判断房屋类型，与 FakeJudge 的作用相同。
    """

    def __init__(self, truth: dict) -> None:
        self.truth = truth

    @classmethod
    def beside(cls, map_path: str) -> Optional['TruthClassifier']:
        """Load the truth generated next to a sim map (truth.json); None if there is none."""
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(map_path)), 'truth.json')
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            truth = json.load(f)
        return cls(truth) if 'buildings' in truth or 'house' in truth else None

    def classify_house(self, photo: str, pose: Sequence[float], front: Tuple[Point, Point],
                       height: float, out_dir: str) -> 'Classification':
        for b in self.truth.get('buildings', []):
            if not b['target']:
                continue
            if b['kind'] == 'row':
                u = next(u for u in b['units'] if u['target'])
                end = u['unit'] in (1, len(b['units']))
                return Classification('townhouse_row', 2, 'none', (tuple(u['from']),
                                      tuple(u['to'])), end, 'high', 'from the truth', 'truth')
            return Classification('detached', 2, 'attached', None, None, 'high',
                                  'from the truth', 'truth')
        # An M2 scenario: one single-storey house. M2 场景：一栋单层房屋。
        return Classification('detached', 1, 'none', None, None, 'high', 'from the truth',
                              'truth')
