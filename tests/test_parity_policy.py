"""Parity-policy propagation and strict output verification.

Every test here corresponds to a concrete way the R4 LTX run accepted a
candidate it should have rejected. Run ``ltx-v1-overnight-20260801-r4-sol``
packaged four VAE kernels built on ``rcp.approx.ftz.f32`` / ``tanh.approx.f32``
against a workload declaring ``parity.policy: byte_equal``; full-generation
parity then failed, hours of GPU time later. These are CPU-only tests.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from autokernel.specs.types import SpecValidationError, Tolerance
from autokernel.verification.outputs import (
    compare_output_trees,
    compare_tensor_leaf,
)
from autokernel.verification.policy import (
    ParityPolicy,
    ToleranceResolutionError,
    detect_approximate_math,
    resolve_leaf_tolerance,
)

BF16 = {"bfloat16": Tolerance(atol=2e-2, rtol=2e-2)}


# -- policy semantics ---------------------------------------------------


def test_byte_equal_is_exact_and_forbids_relaxation() -> None:
    policy = ParityPolicy(policy="byte_equal")
    assert policy.exact
    assert not policy.approximate_math_allowed
    assert not policy.relaxation_allowed
    # The stability stage asks for relax=10.0; an exact policy refuses it.
    assert policy.effective_relax(10.0) == 1.0


def test_tolerance_policy_permits_relaxation_and_approximate_math() -> None:
    policy = ParityPolicy(policy="tolerance")
    assert not policy.exact
    assert policy.approximate_math_allowed
    assert policy.effective_relax(10.0) == 10.0


def test_missing_parity_block_defaults_to_the_strictest_policy() -> None:
    class _Workload:
        parity = None

    assert ParityPolicy.from_workload(_Workload()).exact


def test_workload_approximate_math_choice_is_forwarded() -> None:
    class _Parity:
        policy = "tolerance"
        atol = 1e-3
        rtol = 1e-3
        allow_approximate_math = False

    class _Workload:
        parity = _Parity()

    policy = ParityPolicy.from_workload(_Workload())
    assert not policy.approximate_math_allowed


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(SpecValidationError):
        ParityPolicy(policy="approximately_fine")


# -- tolerance resolution ----------------------------------------------


def test_exact_policy_pins_tolerance_to_zero() -> None:
    tol = resolve_leaf_tolerance("bfloat16", BF16, ParityPolicy("byte_equal"))
    assert tol == Tolerance(atol=0.0, rtol=0.0)


def test_undeclared_dtype_raises_instead_of_defaulting_to_1e_2() -> None:
    """R4's root cause: manifests carried ``tolerances: null``.

    The verifier silently substituted DEFAULT_TOLERANCE (1e-2/1e-2) for every
    bfloat16 leaf, which is why approximate kernels reported ``match=True``.
    """
    with pytest.raises(ToleranceResolutionError, match="requires an explicit"):
        resolve_leaf_tolerance(
            "bfloat16",
            {},  # exactly what spec_from_manifest produced in R4
            ParityPolicy("tolerance"),
            default=Tolerance(atol=1e-2, rtol=1e-2),
        )


def test_declared_dtype_is_used_when_present() -> None:
    tol = resolve_leaf_tolerance("bfloat16", BF16, ParityPolicy("tolerance"))
    assert tol == BF16["bfloat16"]


def test_workload_override_applies_to_every_dtype() -> None:
    policy = ParityPolicy("tolerance", atol=1e-5, rtol=1e-6)
    tol = resolve_leaf_tolerance("float64", {}, policy)
    assert tol == Tolerance(atol=1e-5, rtol=1e-6)


# -- exact comparison ---------------------------------------------------


def test_exact_policy_rejects_a_one_ulp_difference() -> None:
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    candidate = expected.clone()
    candidate[1] = torch.nextafter(candidate[1], torch.tensor(3.0))
    record = compare_tensor_leaf("output", candidate, expected, Tolerance(0.0, 0.0), exact=True)
    assert not record.match
    assert "not bitwise equal" in record.reason
    assert record.max_abs_error > 0.0


def test_exact_policy_accepts_identical_tensors() -> None:
    expected = torch.randn(64, dtype=torch.float32)
    record = compare_tensor_leaf(
        "output", expected.clone(), expected, Tolerance(0.0, 0.0), exact=True
    )
    assert record.match
    assert record.max_abs_error == 0.0


def test_exact_policy_rejects_approximate_reciprocal_error() -> None:
    """Stand-in for ``rcp.approx.ftz.f32``: right to ~1e-7, not bit-exact."""
    expected = torch.linspace(1.0, 10.0, 512, dtype=torch.float32)
    candidate = expected * (1.0 + 1e-7)
    strict = compare_output_trees(
        candidate, expected, BF16, policy=ParityPolicy("byte_equal")
    )
    assert not strict.match
    loose = compare_output_trees(
        candidate, expected, {"float32": Tolerance(1e-2, 1e-2)},
        policy=ParityPolicy("tolerance"),
    )
    assert loose.match, "the same kernel passes a tolerance workload"


# -- the large-value / relative-tolerance trap --------------------------

def test_relative_tolerance_hides_a_huge_absolute_error() -> None:
    """Reproduces the exact R4 number: max_abs_error=32768.0 reported as a pass.

    With reference values near 3.3e6 an rtol of 1e-2 licenses ~3.3e4 of error.
    Nothing about ``allclose`` is wrong; the metric is simply not a safety
    property on its own.
    """
    expected = torch.full((1024,), 3.3e6, dtype=torch.float32)
    candidate = expected.clone()
    candidate[0] += 32768.0

    permissive = compare_tensor_leaf(
        "output", candidate, expected, Tolerance(atol=1e-1, rtol=1e-1)
    )
    assert permissive.match, "this is the R4 behaviour being fixed"
    assert permissive.max_abs_error == pytest.approx(32768.0)
    assert permissive.pct_within_tol == 100.0

    capped = compare_tensor_leaf(
        "output",
        candidate,
        expected,
        Tolerance(atol=1e-1, rtol=1e-1),
        max_absolute_error=1.0,
    )
    assert not capped.match
    assert "absolute ceiling" in capped.reason
    assert "scaled by large reference values" in capped.reason


def test_absolute_cap_propagates_through_the_tree_comparator() -> None:
    expected = torch.full((256,), 6.5e5, dtype=torch.float32)
    candidate = expected.clone()
    candidate[3] += 65536.0
    policy = ParityPolicy("tolerance", atol=1e-1, rtol=1e-1, max_absolute_error=1.0)
    assert not compare_output_trees(candidate, expected, {}, policy=policy).match


# -- non-finite handling ------------------------------------------------


def test_matching_infinities_no_longer_report_nan_error() -> None:
    """``inf - inf`` is NaN, which used to poison ``max_abs_error``."""
    expected = torch.tensor([1.0, float("inf"), 3.0], dtype=torch.float32)
    record = compare_tensor_leaf(
        "output", expected.clone(), expected, Tolerance(1e-3, 1e-3)
    )
    assert record.match
    assert record.max_abs_error == record.max_abs_error, "must not be NaN"
    assert record.max_abs_error == 0.0
    assert record.has_inf


def test_flipped_infinity_sign_is_rejected() -> None:
    expected = torch.tensor([float("inf"), 1.0], dtype=torch.float32)
    candidate = torch.tensor([float("-inf"), 1.0], dtype=torch.float32)
    record = compare_tensor_leaf("output", candidate, expected, Tolerance(1e-3, 1e-3))
    assert not record.match
    assert "infinity signs" in record.reason


def test_candidate_nan_where_reference_is_finite_is_rejected() -> None:
    expected = torch.ones(8, dtype=torch.float32)
    candidate = expected.clone()
    candidate[2] = float("nan")
    record = compare_tensor_leaf("output", candidate, expected, Tolerance(1e9, 1e9))
    assert not record.match, "an enormous tolerance must not launder a NaN"
    assert "NaN positions differ" in record.reason
    assert record.has_nan


def test_reference_nan_must_be_reproduced() -> None:
    expected = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
    candidate = torch.tensor([0.0, 1.0], dtype=torch.float32)
    record = compare_tensor_leaf("output", candidate, expected, Tolerance(1e-3, 1e-3))
    assert not record.match
    assert "NaN positions differ" in record.reason


def test_matching_nan_positions_are_accepted_under_tolerance() -> None:
    expected = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    record = compare_tensor_leaf(
        "output", expected.clone(), expected, Tolerance(1e-3, 1e-3)
    )
    assert record.match


# -- contract: dtype, shape, layout -------------------------------------


def test_exact_policy_rejects_a_layout_change() -> None:
    """Right values, wrong strides: ``allclose`` passes, consumers break."""
    expected = torch.randn(8, 16, dtype=torch.float32)
    candidate = expected.t().contiguous().t()
    assert torch.equal(candidate, expected)
    assert candidate.stride() != expected.stride()

    assert compare_output_trees(candidate, expected, {}, policy=ParityPolicy("tolerance", atol=0.0, rtol=0.0)).match
    strict = compare_output_trees(candidate, expected, {}, policy=ParityPolicy("byte_equal"))
    assert not strict.match
    assert "layout mismatch" in strict.reason


def test_dtype_mismatch_is_rejected_under_every_policy() -> None:
    expected = torch.ones(4, dtype=torch.float32)
    candidate = torch.ones(4, dtype=torch.bfloat16)
    for policy in (ParityPolicy("byte_equal"), ParityPolicy("tolerance", atol=1.0, rtol=1.0)):
        comparison = compare_output_trees(candidate, expected, BF16, policy=policy)
        assert not comparison.match
        assert "dtype mismatch" in comparison.reason


# -- stability-stage relaxation ----------------------------------------


def test_stability_relaxation_is_disabled_under_byte_equal() -> None:
    """R4 applied relax=10.0 to adversarial inputs, widening 1e-2 to 1e-1."""
    expected = torch.full((128,), 1000.0, dtype=torch.float32)
    candidate = expected.clone()
    candidate[0] += 5.0

    relaxed = compare_output_trees(
        candidate, expected, {"float32": Tolerance(1e-2, 1e-2)},
        relax=10.0, policy=ParityPolicy("tolerance"),
    )
    assert relaxed.match

    strict = compare_output_trees(
        candidate, expected, {"float32": Tolerance(1e-2, 1e-2)},
        relax=10.0, policy=ParityPolicy("byte_equal"),
    )
    assert not strict.match


# -- approximate-math screening ----------------------------------------


@pytest.mark.parametrize(
    "source, expected_marker",
    [
        ("asm=\"rcp.approx.ftz.f32 $0, $1;\"", "rcp.approx"),
        ("tanh.approx.f32 $0, $1;", "tanh.approx"),
        ("y = __expf(x);", "__expf"),
        ("torch.backends.cuda.matmul.allow_tf32 = True", "allow_tf32"),
    ],
)
def test_approximate_math_markers_are_detected(source: str, expected_marker: str) -> None:
    assert expected_marker in detect_approximate_math(source)


def test_exact_arithmetic_is_not_flagged() -> None:
    assert detect_approximate_math("value = a * b + c\ntl.store(ptr, value)") == ()


def test_r4_vae_kernel_source_is_flagged() -> None:
    """The literal intrinsics from mk-a81e140d62ff170c-43850138-sm100."""
    source = """
    @triton.jit
    def _approx_reciprocal(value):
        return tl.inline_asm_elementwise(asm="rcp.approx.ftz.f32 $0, $1;", ...)
    @triton.jit
    def _approx_tanh(value):
        return tl.inline_asm_elementwise(asm="tanh.approx.f32 $0, $1;", ...)
    """
    markers = detect_approximate_math(source)
    assert "rcp.approx" in markers and "tanh.approx" in markers
    assert not ParityPolicy("byte_equal").approximate_math_allowed


# -- stage propagation: workload -> bench.py ----------------------------


def test_ltx_workload_policy_reaches_the_benchmark_command() -> None:
    """The end-to-end propagation R4 was missing.

    ``workloads/ltx_480p.yaml`` declares ``parity.policy: byte_equal``. Before
    this wiring the declaration reached only the final frame comparison, so the
    search and isolated-validation gates used an unrelated 1e-2 tolerance.
    """
    from pathlib import Path

    from autokernel.optimize.search import _benchmark_command, _parity_settings

    policy, ceiling = _parity_settings({"workload": "workloads/ltx_480p.yaml"})
    assert policy == "byte_equal"

    generated = {name: Path(f"/candidate/{name}") for name in ("spec", "corpus", "kernel", "manifest")}
    command = _benchmark_command(
        Path("/repo"),
        generated,
        Path("/out/result.json"),
        baseline="compile",
        quick=True,
        parity_policy=policy,
        max_absolute_error=ceiling,
    )
    assert "--parity-policy" in command
    assert command[command.index("--parity-policy") + 1] == "byte_equal"
    assert "--dispatch-stress" in command


def test_absent_workload_falls_back_to_the_strictest_policy() -> None:
    from autokernel.optimize.search import _parity_settings

    assert _parity_settings({})[0] == "byte_equal"
    assert _parity_settings({"workload": "/nonexistent/workload.yaml"})[0] == "byte_equal"


def test_absolute_error_ceiling_is_forwarded_when_configured() -> None:
    from pathlib import Path

    from autokernel.optimize.search import _benchmark_command, _parity_settings

    policy, ceiling = _parity_settings(
        {"workload": "workloads/ltx_480p.yaml", "max_absolute_error": 0.5}
    )
    assert ceiling == 0.5
    generated = {name: Path(f"/c/{name}") for name in ("spec", "corpus", "kernel", "manifest")}
    command = _benchmark_command(
        Path("/repo"), generated, Path("/out/r.json"),
        baseline="eager", quick=False,
        parity_policy=policy, max_absolute_error=ceiling,
    )
    assert command[command.index("--max-absolute-error") + 1] == "0.5"
