"""Parse strategy specs like ``nesterov(lr=0.7, momentum=0.9)`` used in configuration (DESIGN §9)."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """``"name(a=1, b='x')"`` → ``("name", {"a": 1, "b": "x"})``. A bare ``"name"`` has no kwargs."""
    spec = spec.strip()
    try:
        node = ast.parse(spec, mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"invalid strategy spec: {spec!r}") from e
    if isinstance(node, ast.Name):
        return node.id, {}
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
        raise ValueError(f"strategy spec must look like name(key=value, ...): {spec!r}")
    kwargs = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError(f"**kwargs not allowed in spec: {spec!r}")
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    return node.func.id, kwargs


def build(spec: str, registry: dict[str, Callable[..., Any]]) -> Any:
    name, kwargs = parse_spec(spec)
    try:
        factory = registry[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; known: {sorted(registry)}") from None
    return factory(**kwargs)
