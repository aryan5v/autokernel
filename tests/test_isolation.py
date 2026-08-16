"""Per-artifact end-to-end isolation reporting.

Driven by the real dispatch decisions from
ltx-v1-overnight-20260801-r4-sol/stages/end_to_end_validate/dispatch.json, which
recorded 56 candidate calls across four artifacts in one scope with no way to
attribute the parity failure or the 0.8327x regression to any of them.
"""

from __future__ import annotations

import json

import pytest

from autokernel.optimize.isolation import (
    IsolationError,
    IsolationReport,
    TrialRecord,
    artifact_ids_in,
    dispatch_counts_for,
)

R4_ARTIFACTS = (
    "mk-a81e140d62ff170c-43850138-sm100",
    "mk-b6cb64f99049683b-c41e9922-sm100",
    "mk-baecc3825d4a8c18-b92f826b-sm100",
    "mk-bbfe15180d31bf50-6755dc62-sm100",
)


def _r4_dispatch() -> dict:
    """The four selected decisions r4 recorded, one shape variant each."""
    decisions = [
        {
            "active": True,
            "artifact_id": artifact_id,
            "calls": 14,
            "candidate_calls": 14,
            "runtime_fallbacks": 0,
            "reason": "artifact_selected",
            "scope": "vae.decoder.up_blocks.6.res_blocks",
            "shape_key": f"input_0:variant{index}:bfloat16",
        }
        for index, artifact_id in enumerate(R4_ARTIFACTS)
    ]
    decisions.append(
        {
            "active": False,
            "artifact_id": None,
            "calls": 1151,
            "candidate_calls": 0,
            "runtime_fallbacks": 0,
            "reason": "no_artifact_for_input_signature",
            "scope": "transformer.model.transformer_blocks",
            "shape_key": "kwarg_video_x:1x4680x4096:bfloat16",
        }
    )
    return {"decisions": decisions, "dispatch": {"reason_counts": {"artifact_selected": 4}}}


def test_combined_counts_reproduce_the_r4_total() -> None:
    counts = dispatch_counts_for(_r4_dispatch())
    assert counts["candidate_calls"] == 56
    assert counts["runtime_fallbacks"] == 0
    assert counts["scopes"] == ("vae.decoder.up_blocks.6.res_blocks",)


def test_counts_can_be_attributed_to_one_artifact() -> None:
    """What r4 could not answer: how much did this one artifact run?"""
    counts = dispatch_counts_for(_r4_dispatch(), ["mk-a81e140d62ff170c-43850138-sm100"])
    assert counts["candidate_calls"] == 14


def test_unselected_scopes_never_count_toward_an_artifact() -> None:
    counts = dispatch_counts_for(_r4_dispatch(), ["mk-bbfe15180d31bf50-6755dc62-sm100"])
    assert counts["calls"] == 14, "the 1151 unselected transformer calls are not ours"


def test_malformed_diagnostics_fail_loudly() -> None:
    with pytest.raises(IsolationError):
        dispatch_counts_for({"dispatch": {}})


# -- trial classification -----------------------------------------------


def _trial(**overrides) -> TrialRecord:
    base = dict(
        trial="mk-test",
        artifact_ids=("mk-test",),
        status="ok",
        dispatch_calls=14,
        candidate_calls=14,
        runtime_fallbacks=0,
        parity_passed=True,
        end_to_end_speedup=1.02,
    )
    base.update(overrides)
    return TrialRecord(**base)


def test_a_parity_failure_is_never_safe() -> None:
    assert not _trial(parity_passed=False).safe


def test_a_runtime_fallback_is_never_safe() -> None:
    assert not _trial(runtime_fallbacks=1).safe


def test_an_artifact_that_never_dispatched_is_not_worthwhile() -> None:
    assert not _trial(candidate_calls=0).worthwhile


def test_a_regression_is_not_worthwhile() -> None:
    """r4's combined result: 0.8327x."""
    assert not _trial(end_to_end_speedup=0.8327386235303763).worthwhile
    assert _trial(end_to_end_speedup=1.02).worthwhile


def test_report_separates_offenders_from_keepers() -> None:
    report = IsolationReport(native_median_wall_seconds=3.2818314481992275)
    report.trials = [
        _trial(trial="a", artifact_ids=("a",), parity_passed=False, end_to_end_speedup=1.05),
        _trial(trial="b", artifact_ids=("b",), end_to_end_speedup=1.03),
        _trial(trial="c", artifact_ids=("c",), end_to_end_speedup=0.97),
        _trial(trial="d", artifact_ids=("d",), candidate_calls=0, end_to_end_speedup=1.0),
    ]
    assert report.parity_offenders == ("a",)
    assert report.safe_and_worthwhile == ("b",)


def test_combined_trials_do_not_pollute_the_keeper_set() -> None:
    """Only single-artifact trials establish an individual verdict."""
    report = IsolationReport()
    report.trials = [
        _trial(trial="combined", artifact_ids=("a", "b"), end_to_end_speedup=1.05),
    ]
    assert report.safe_and_worthwhile == ()


def test_table_renders_every_trial() -> None:
    report = IsolationReport()
    report.trials = [_trial(trial="mk-a"), _trial(trial="mk-b", parity_passed=False)]
    table = report.table()
    assert "mk-a" in table and "mk-b" in table
    assert "FAIL" in table and "pass" in table


# -- artifact discovery -------------------------------------------------


def test_artifact_ids_are_read_from_bundle_manifests(tmp_path) -> None:
    for artifact_id in R4_ARTIFACTS:
        directory = tmp_path / artifact_id
        directory.mkdir()
        (directory / "artifact.json").write_text(
            json.dumps({"artifact_id": artifact_id, "schema_version": 1}),
            encoding="utf-8",
        )
    assert artifact_ids_in(tmp_path) == tuple(sorted(R4_ARTIFACTS))


def test_an_empty_artifact_root_is_reported(tmp_path) -> None:
    assert artifact_ids_in(tmp_path) == ()


# -- execution signals are per kind -------------------------------------
#
# A schedule transform never increments candidate_calls. Asserting the kernel
# counter for every kind reads "did not run" as "ran" for the kinds that do not
# touch it, which is how a timing whose intervention never happened gets read
# as a result (SLURM 1078).


def test_a_kernel_trial_still_proves_itself_with_dispatch_calls() -> None:
    assert _trial(target_kinds=("module",), candidate_calls=14).ran
    assert not _trial(target_kinds=("module",), candidate_calls=0).ran


def test_a_schedule_transform_is_not_judged_by_the_kernel_counter() -> None:
    # Dispatch calls say nothing about a loop transform either way.
    never_ran = _trial(
        target_kinds=("schedule_transform",),
        candidate_calls=0,
        execution_evidence={"steps_skipped": 0, "hook_invocations": 0},
    )
    assert not never_ran.ran
    assert "schedule_transform" in never_ran.execution_failures[0]

    ran = _trial(
        target_kinds=("schedule_transform",),
        candidate_calls=0,
        execution_evidence={"steps_skipped": 6, "hook_invocations": 40},
    )
    assert ran.ran
    assert ran.worthwhile


def test_a_transform_with_no_reported_evidence_is_not_assumed_to_have_run() -> None:
    """Absent evidence is not zero and is certainly not proof."""
    silent = _trial(target_kinds=("schedule_transform",), execution_evidence={})
    assert not silent.ran
    assert "no evidence reported" in silent.execution_failures[0]


def test_an_attention_trial_needs_the_backend_it_asked_for() -> None:
    substituted = _trial(
        target_kinds=("attention",),
        execution_evidence={
            "attention_backend": "SAGE_ATTN",
            "effective_backend": "FLASH_ATTN",
        },
    )
    assert not substituted.ran
    assert "FLASH_ATTN" in substituted.execution_failures[0]

    honest = _trial(
        target_kinds=("attention",),
        execution_evidence={
            "attention_backend": "SAGE_ATTN",
            "effective_backend": "SAGE_ATTN",
        },
    )
    assert honest.ran


def test_a_mixed_trial_must_satisfy_every_kind_present() -> None:
    trial = _trial(
        target_kinds=("module", "schedule_transform"),
        execution_evidence={"candidate_calls": 14},
    )
    assert not trial.ran
    assert len(trial.execution_failures) == 1
    assert trial.execution_failures[0].startswith("schedule_transform")
