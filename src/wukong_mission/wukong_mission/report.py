# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""Write the mission report (Markdown for people, JSON for programs)."""

from dataclasses import asdict
import json
import os
from typing import List, Optional

from wukong_flight.skills import describe

from .judge import Verdict
from .planner import Plan

_ANSWER = {'yes': 'needs attention', 'no': 'OK', 'unclear': 'unclear',
           'error': 'not judged', 'pending': 'not judged (timed out)',
           'skipped': 'not inspected'}


def _segment_summary(verdicts: List[Verdict]) -> str:
    """Summarise per-segment results, e.g. 'Needs cleaning: south gutter segments 2, 4.'."""
    bad: dict = {}
    unknown: dict = {}
    for v in verdicts:
        g, i = v.extra['gutter'], v.extra['segment']
        if v.needs_attention == 'yes':
            bad.setdefault(g, []).append(i)
        elif v.needs_attention != 'no':
            unknown.setdefault(g, []).append(i)

    def fmt(d):
        return '; '.join(f'{g} segment{"s" if len(ix) > 1 else ""} '
                         + ', '.join(map(str, sorted(ix))) for g, ix in sorted(d.items()))
    if not bad and sum(map(len, unknown.values())) == len(verdicts):
        return f'No segment could be judged: {fmt(unknown)}.'
    parts = [f'Needs cleaning: {fmt(bad)}.' if bad else 'No gutter segment needs cleaning.']
    if unknown:
        parts.append(f'Could not tell: {fmt(unknown)}.')
    return ' '.join(parts)


def summary_line(verdicts: List[Verdict]) -> str:
    bad = [v.label for v in verdicts if v.needs_attention == 'yes']
    unknown = [v.label for v in verdicts if v.needs_attention not in ('yes', 'no')]
    if not verdicts:
        return 'No photos were taken.'
    if all('gutter' in v.extra for v in verdicts):
        return _segment_summary(verdicts)
    if len(unknown) == len(verdicts):
        return 'No photo could be judged: ' + ', '.join(unknown) + '.'
    parts = [f'Needs attention: {", ".join(bad)}.' if bad else 'Nothing needs attention.']
    if unknown:
        parts.append(f'Could not tell: {", ".join(unknown)}.')
    return ' '.join(parts)


def write_report(out_dir: str, instruction: str, plan: Optional[Plan], flight_ok: bool,
                 flight_note: str, verdicts: List[Verdict]) -> str:
    """Write report.md and report.json into out_dir; return the Markdown path."""
    os.makedirs(out_dir, exist_ok=True)
    lines = ['# Wukong mission report', '',
             f'**Request:** {instruction}', '']
    if plan is not None:
        lines += [f'**Understood as:** {plan.understood}', '',
                  f'**Flight:** {"completed, landed" if flight_ok else "did not complete"}'
                  + (f' ({flight_note})' if flight_note else ''), '',
                  f'**Result:** {summary_line(verdicts)}', '',
                  '## Photos', '']
    segs = [v for v in verdicts if 'gutter' in v.extra]
    if segs:
        lines += ['| Gutter | Segment | Result | Confidence | Findings |',
                  '|---|---|---|---|---|']
        for v in sorted(segs, key=lambda v: (v.extra['gutter'], v.extra['segment'])):
            x = v.extra
            answer = _ANSWER.get(v.needs_attention, v.needs_attention)
            findings = v.findings.replace('|', '/')
            lines.append(f'| {x["gutter"]} | {x["segment"]} of {x["segments"]} | {answer} | '
                         f'{v.confidence} | {findings} |')
        lines.append('')
    for v in verdicts:
        lines += [f'### {v.label}: {_ANSWER.get(v.needs_attention, v.needs_attention)}'
                  f' ({v.confidence} confidence)', '',
                  v.findings + (f' Where: {v.where}.' if v.where else ''), '']
        if v.path:                # segments not inspected have no photo
            lines += [f'![{v.label}]({os.path.relpath(v.path, out_dir)})', '']
    if plan is not None:
        lines += ['## Plan', '']
        lines += [f'{i}. {describe(s)}' for i, s in enumerate(plan.skills, 1)]
        lines.append('')
    md = os.path.join(out_dir, 'report.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    with open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump({'instruction': instruction,
                   'understood': plan.understood if plan else None,
                   'plan': [describe(s) for s in plan.skills] if plan else [],
                   'flight_ok': flight_ok, 'flight_note': flight_note,
                   'summary': summary_line(verdicts),
                   'photos': [asdict(v) for v in verdicts]}, f, ensure_ascii=False, indent=2)
    return md
