#!/usr/bin/env python3
# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Check that every public function has type annotations (docs/CONVENTIONS.md).

    python3 scripts/check_annotations.py            # run by scripts/check.sh

Public: module-level functions and methods of classes whose names do not
start with an underscore (__init__ counts; its return may be left out).
Every parameter but self/cls needs an annotation, and so does the return.
Tests and setup.py are not checked. Nothing checks the annotations
themselves (no type checker runs); this only keeps them from going missing.
公共函数（模块级函数和公共类的方法，不以下划线开头，含 __init__）的每个参数和返回值
都要有类型标注；测试和 setup.py 不检查。不检查标注是否正确，只防止缺失。
"""

import ast
import glob
import os
import sys
from typing import Iterator, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = ('src/*/*/*.py', 'sim/*.py', 'scripts/*.py')


def checked_files(root: str = ROOT) -> List[str]:
    """Return the source files the rule applies to, relative to root."""
    out = []
    for pattern in PATTERNS:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            rel = os.path.relpath(path, root)
            name = os.path.basename(rel)
            if name.startswith('test_') or name == 'setup.py' or '/test/' in rel:
                continue
            out.append(rel)
    return out


def _public(tree: ast.Module) -> Iterator[tuple]:
    """Yield (function, class or None) for every public function and method."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, None
        elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield m, node


def missing(source: str) -> List[str]:
    """Return 'line name: what is missing' for each public function lacking annotations."""
    out = []
    for fn, cls in _public(ast.parse(source)):
        if fn.name.startswith('_') and fn.name != '__init__':
            continue
        a = fn.args
        params = a.posonlyargs + a.args + a.kwonlyargs
        if cls is not None and params and params[0].arg in ('self', 'cls') and not any(
                isinstance(d, ast.Name) and d.id == 'staticmethod' for d in fn.decorator_list):
            params = params[1:]
        bare = [p.arg for p in params if p.annotation is None]
        for star, v in (('*', a.vararg), ('**', a.kwarg)):
            if v is not None and v.annotation is None:
                bare.append(star + v.arg)
        lacks = []
        if bare:
            lacks.append('parameters ' + ', '.join(bare))
        if fn.returns is None and fn.name != '__init__':
            lacks.append('return')
        if lacks:
            name = f'{cls.name}.{fn.name}' if cls is not None else fn.name
            out.append(f'{fn.lineno} {name}: no annotation for {" and ".join(lacks)}')
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """Print every gap; return 1 if there is any."""
    gaps = 0
    for rel in checked_files():
        with open(os.path.join(ROOT, rel), encoding='utf-8') as f:
            for line in missing(f.read()):
                print(f'{rel}:{line}')
                gaps += 1
    if gaps:
        print(f'{gaps} public function(s) without full type annotations')
    return 1 if gaps else 0


if __name__ == '__main__':
    sys.exit(main())
