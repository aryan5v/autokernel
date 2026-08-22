"""Static execution-context policy for generated candidate kernels.

MotionKernel candidates own computation, but the embedding runtime owns CUDA
streams and CUDA graphs. Letting a candidate capture a graph around its local
microbenchmark is unsafe in a repeated model stack: it can bake live parameter
addresses into the graph, poison an outer capture, or turn every new block into
another expensive capture. The fixed harness rejects those APIs before it
imports candidate code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = [
    "CandidatePolicyViolation",
    "find_execution_context_violations",
]


_FORBIDDEN_APIS = frozenset(
    {
        "torch.cuda.CUDAGraph",
        "torch.cuda.ExternalStream",
        "torch.cuda.Stream",
        "torch.cuda.graph",
        "torch.cuda.graph_pool_handle",
        "torch.cuda.graphs.CUDAGraph",
        "torch.cuda.graphs.graph",
        "torch.cuda.make_graphed_callables",
        "torch.cuda.set_stream",
        "torch.cuda.stream",
    }
)


@dataclass(frozen=True)
class CandidatePolicyViolation:
    """One forbidden execution-context API reference in candidate source."""

    api: str
    line: int
    column: int


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".", 1)[0]
                aliases[local] = item.name if item.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                if item.name == "*":
                    continue
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _resolve(name: str, aliases: dict[str, str]) -> str:
    head, separator, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{tail}" if separator else resolved


def find_execution_context_violations(
    source: str,
) -> tuple[CandidatePolicyViolation, ...]:
    """Return forbidden CUDA graph/stream API references in Python source.

    The check is syntax-aware, so comments and docstrings do not trigger it,
    and it resolves ordinary import aliases such as ``import torch as t`` or
    ``from torch.cuda import CUDAGraph as Graph``.
    """

    tree = ast.parse(source)
    aliases = _aliases(tree)
    found: set[tuple[str, int, int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        name = _dotted_name(node)
        if not name:
            continue
        resolved = _resolve(name, aliases)
        if resolved in _FORBIDDEN_APIS:
            found.add(
                (
                    resolved,
                    int(getattr(node, "lineno", 0)),
                    int(getattr(node, "col_offset", 0)),
                )
            )
    return tuple(
        CandidatePolicyViolation(api=api, line=line, column=column)
        for api, line, column in sorted(
            found, key=lambda item: (item[1], item[2], item[0])
        )
    )
