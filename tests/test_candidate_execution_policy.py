"""Candidate code may compute on CUDA, but may not own its execution context."""

from __future__ import annotations

import pytest

from autokernel.optimize.search import BuiltinSearchError, _dispatch_stress
from autokernel.verification import find_execution_context_violations


@pytest.mark.parametrize(
    ("source", "api"),
    [
        ("import torch\ng = torch.cuda.CUDAGraph()\n", "torch.cuda.CUDAGraph"),
        ("import torch as t\nwith t.cuda.graph(None): pass\n", "torch.cuda.graph"),
        (
            "from torch.cuda import CUDAGraph as Graph\ng = Graph()\n",
            "torch.cuda.CUDAGraph",
        ),
        ("import torch\ns = torch.cuda.Stream()\n", "torch.cuda.Stream"),
        (
            "from torch.cuda import stream as use_stream\nuse_stream(None)\n",
            "torch.cuda.stream",
        ),
    ],
)
def test_forbidden_execution_context_apis_are_detected(
    source: str, api: str
) -> None:
    violations = find_execution_context_violations(source)
    assert api in {item.api for item in violations}


def test_comments_docstrings_and_normal_kernel_launches_are_allowed() -> None:
    source = '''\
"""Do not call torch.cuda.CUDAGraph or torch.cuda.Stream here."""
import torch

def kernel_fn(x):
    # torch.cuda.graph would be unsafe here.
    return torch.relu(x)
'''
    assert find_execution_context_violations(source) == ()


def test_search_requires_dispatch_stress_evidence() -> None:
    with pytest.raises(BuiltinSearchError, match="no dispatch-stress evidence"):
        _dispatch_stress({"performance": {"primary": {}}})


def test_search_rejects_a_fresh_identity_regression() -> None:
    payload = {
        "performance": {
            "dispatch_stress": {
                "status": "FAIL",
                "correctness": "PASS",
                "fresh_input_speedup": 0.8,
                "steady_input_speedup": 1.2,
                "reason": "fresh identities are slower",
            }
        }
    }
    with pytest.raises(BuiltinSearchError, match="fresh identities are slower"):
        _dispatch_stress(payload)
