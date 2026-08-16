"""CPU contract tests for the resumable V1 optimize control plane."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from autokernel.optimize import (
    PIPELINE_STAGES,
    OptimizeConfig,
    OptimizeError,
    run_optimize,
)
from autokernel.optimize.runner import _decide_terminal
from autokernel.optimize.stages import _load_stage_result
from conftest import make_fastvideo_checkout, make_workload


def _config(tmp_path: Path, repo_root: Path, **overrides) -> OptimizeConfig:
    checkout = make_fastvideo_checkout(tmp_path)
    workload = make_workload(tmp_path / "workload.json")
    values = {
        "fastvideo_checkout": checkout,
        "model": "test/model",
        "workload": workload,
        "output": tmp_path / "run",
        "budget_hours": 1.0,
        "repo_root": repo_root,
    }
    values.update(overrides)
    return OptimizeConfig(**values)


def _simulate(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    monkeypatch.setenv("MOTIONKERNEL_SIMULATE", "1")
    monkeypatch.setenv("MOTIONKERNEL_SIMULATE_OUTCOME", outcome)


def test_promoted_campaign_writes_receipt_state_report_and_preserves_candidate(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root)

    receipt = run_optimize(config)

    assert receipt["terminal"] == "promoted"
    assert receipt["completed_stages"] == list(PIPELINE_STAGES)
    assert receipt["candidates"][0]["fingerprint"] == "fp_toy_001"
    assert receipt["candidates"][0]["status"] == "promoted"
    assert (config.output / "receipt.json").is_file()
    assert (config.output / "morning_report.md").is_file()
    assert (config.output / "artifacts" / "manifest.json").is_file()
    report = (config.output / "morning_report.md").read_text(encoding="utf-8")
    assert "end-to-end" in report
    assert "Isolated operator speedup alone **never** promotes" in report


def test_workload_promotion_threshold_cannot_be_weakened_by_campaign_config(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    workload = make_workload(
        tmp_path / "strict-workload.json",
        performance={
            "min_end_to_end_speedup": 1.2,
            "max_peak_memory_regression": 0.05,
        },
    )
    config = _config(
        tmp_path,
        repo_root,
        workload=workload,
        min_e2e_speedup=1.01,
    )

    receipt = run_optimize(config)

    assert receipt["terminal"] == "no_worthwhile_candidate"
    assert "below threshold" in receipt["message"]


def test_no_worthwhile_candidate_stops_after_discovery(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "no_worthwhile_candidate")
    config = _config(tmp_path, repo_root)

    receipt = run_optimize(config)

    assert receipt["terminal"] == "no_worthwhile_candidate"
    assert receipt["completed_stages"] == ["baseline", "profile", "discover"]
    assert not (config.output / "stages" / "search").exists()


def test_isolated_speedup_cannot_promote_without_e2e_improvement():
    state = {"candidates": [{"fingerprint": "candidate"}]}
    terminal, message = _decide_terminal(
        state,
        {
            "isolated_validate": {"metrics": {"isolated_speedup": 20.0}},
            "end_to_end_validate": {
                "recommendation": "promoted",
                "metrics": {
                    "end_to_end_speedup": 1.0,
                    "classification": "neutral",
                },
            },
        },
        min_e2e_speedup=1.01,
    )
    assert terminal == "no_worthwhile_candidate"
    assert "isolated speedup=20.0 is not sufficient" in message


def test_isolated_no_worthwhile_candidate_is_not_reported_as_failure():
    terminal, message = _decide_terminal(
        {"candidates": [{"fingerprint": "candidate"}]},
        {
            "isolated_validate": {
                "recommendation": "no_worthwhile_candidate",
                "message": "all measured candidates were slower",
                "metrics": {"isolated_speedup": 0.9},
            }
        },
        min_e2e_speedup=1.01,
    )

    assert terminal == "no_worthwhile_candidate"
    assert message == "all measured candidates were slower"


@pytest.mark.parametrize("speedup", [None, "bad", float("nan"), float("inf")])
def test_non_finite_or_invalid_e2e_metrics_never_promote(speedup):
    terminal, message = _decide_terminal(
        {"candidates": [{"fingerprint": "candidate"}]},
        {
            "end_to_end_validate": {
                "recommendation": "promoted",
                "metrics": {
                    "end_to_end_speedup": speedup,
                    "classification": "improved",
                },
            }
        },
        min_e2e_speedup=1.01,
    )
    assert terminal == "no_worthwhile_candidate"
    assert "promotion blocked" in message


def test_resume_skips_durable_completed_stages(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root)
    run_optimize(config)
    baseline_result = config.output / "stages" / "baseline" / "result.json"
    original = baseline_result.read_bytes()

    # Model a process interruption after discovery: durable early-stage state
    # survives, while later stage records are absent and the campaign is live.
    state_path = config.output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state["terminal"] = None
    state["completed_stages"] = list(PIPELINE_STAGES[:3])
    state["stage_records"] = {
        name: state["stage_records"][name] for name in PIPELINE_STAGES[:3]
    }
    state["candidates"] = [
        {
            "name": "toy.elementwise",
            "fingerprint": "fp_toy_001",
            "status": "discovered",
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (config.output / "receipt.json").unlink()

    receipt = run_optimize(config)

    assert receipt["terminal"] == "promoted"
    assert receipt["completed_stages"] == list(PIPELINE_STAGES)
    assert baseline_result.read_bytes() == original


def test_resume_rejects_campaign_identity_drift(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root)
    run_optimize(config)

    # The run contract is compared during preflight, so drift is now reported
    # with a stable reason code before any stage can run.
    changed = _config(tmp_path, repo_root, model="different/model")
    with pytest.raises(OptimizeError, match="contract_mismatch_model"):
        run_optimize(changed)


def test_no_resume_replaces_a_previous_terminal_campaign(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    run_optimize(_config(tmp_path, repo_root))

    _simulate(monkeypatch, "no_worthwhile_candidate")
    receipt = run_optimize(_config(tmp_path, repo_root, resume=False))

    assert receipt["terminal"] == "no_worthwhile_candidate"
    persisted = json.loads(
        (tmp_path / "run" / "receipt.json").read_text(encoding="utf-8")
    )
    assert persisted["terminal"] == "no_worthwhile_candidate"


def test_per_candidate_timeout_is_terminal_and_receipted(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(
        tmp_path,
        repo_root,
        per_candidate_budget_seconds=0.05,
        stage_commands={
            "search": [sys.executable, "-c", "import time; time.sleep(2)"]
        },
    )

    receipt = run_optimize(config)

    assert receipt["terminal"] == "budget_exhausted"
    assert "per-candidate budget exhausted" in receipt["message"]
    assert receipt["failed_stages"]["search"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_campaign_budgets_are_rejected(
    tmp_path: Path, repo_root: Path, value: float
):
    config = _config(tmp_path, repo_root, budget_hours=value)
    with pytest.raises(OptimizeError, match="budget_hours must be finite and positive"):
        run_optimize(config)


def test_stage_command_placeholders_are_expanded(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    script = (
        "import json,os,pathlib; "
        "p=pathlib.Path(os.environ['MOTIONKERNEL_RUN_DIR'])/'stages'/'baseline'/'result.json'; "
        "p.write_text(json.dumps({'schema_version':1,'stage':'baseline','status':'ok',"
        "'message':os.environ['MOTIONKERNEL_MODEL']}))"
    )
    config = _config(
        tmp_path,
        repo_root,
        stage_commands={"baseline": [sys.executable, "-c", script, "{model}"]},
    )
    receipt = run_optimize(config)
    command = json.loads(
        (config.output / "commands" / "baseline.json").read_text(encoding="utf-8")
    )
    assert command["command"][-1] == "test/model"
    assert receipt["terminal"] == "promoted"


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            {"schema_version": 1, "stage": "profile", "status": "ok"},
            "identity mismatch",
        ),
        ({"schema_version": 1, "stage": "baseline"}, "invalid stage result status"),
        (
            {
                "schema_version": 1,
                "stage": "baseline",
                "status": "ok",
                "metrics": [],
            },
            "metrics must be an object",
        ),
    ],
)
def test_stage_result_contract_fails_closed(
    tmp_path: Path, payload: dict, match: str
):
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OptimizeError, match=match):
        _load_stage_result(result, expected_stage="baseline")


def test_stop_after_discover_terminates_as_discovery_complete(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root, stop_after_stage="discover")

    receipt = run_optimize(config)

    assert receipt["terminal"] == "discovery_complete"
    assert receipt["completed_stages"] == ["baseline", "profile", "discover"]
    # No verdict was reached: candidates keep their discovered status rather
    # than being rewritten to promoted/not_promoted.
    assert receipt["candidates"][0]["status"] == "discovered"
    assert "stopped after discover" in receipt["message"]


def test_stop_after_discover_is_idempotent_on_resume(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root, stop_after_stage="discover")
    first = run_optimize(config)
    assert first["terminal"] == "discovery_complete"

    second = run_optimize(config)

    assert second["terminal"] == "discovery_complete"
    assert second["completed_stages"] == ["baseline", "profile", "discover"]
    # A resume must not drift past the requested stop stage.
    assert not (config.output / "stages" / "specgen" / "result.json").exists()


def test_stop_after_discover_preserves_no_worthwhile_verdict(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "no_worthwhile_candidate")
    config = _config(tmp_path, repo_root, stop_after_stage="discover")

    receipt = run_optimize(config)

    assert receipt["terminal"] == "no_worthwhile_candidate"


def test_resume_with_later_stop_stage_continues_campaign(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    discovery_only = _config(tmp_path, repo_root, stop_after_stage="discover")
    assert run_optimize(discovery_only)["terminal"] == "discovery_complete"

    # stop_after_stage is an operational control, not run identity: resuming
    # without it continues the same campaign to a verdict.
    full = _config(tmp_path, repo_root)
    receipt = run_optimize(full)

    assert receipt["terminal"] == "promoted"
    assert receipt["completed_stages"] == list(PIPELINE_STAGES)


def test_stop_after_stage_rejects_unknown_stage(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root, stop_after_stage="not_a_stage")

    with pytest.raises(OptimizeError, match="stop_after_stage"):
        run_optimize(config)


# -- the benchmark object a search writes must actually validate ---------
#
# search.py once wrote whole-model impact fields into the bundle's benchmark
# object. Artifact schema 1 admits isolated harness evidence only, so
# _unknown_fields rejected every bundle: the search produced candidates that
# could never package, and the campaign died at the packaging stage with a
# message about an unknown field rather than anything about the kernel.


def test_the_benchmark_fields_search_writes_are_the_ones_the_schema_admits() -> None:
    import inspect

    from autokernel.artifact.types import _BENCHMARK_FIELDS
    from autokernel.optimize import search as search_module

    source = inspect.getsource(search_module.validate_candidates)
    start = source.index('"benchmark": {')
    end = source.index('"generation": {', start)
    written = set(re.findall(r'"([a-z_]+)":', source[start:end]))
    written.discard("benchmark")

    unknown = written - set(_BENCHMARK_FIELDS)
    assert not unknown, (
        f"validate_candidates writes benchmark fields the artifact schema "
        f"rejects: {sorted(unknown)}. Every bundle would fail validation."
    )


def test_whole_model_impact_survives_in_the_validation_receipt() -> None:
    """Removing those fields from the bundle must not lose the evidence."""
    import inspect

    from autokernel.optimize import search as search_module

    source = inspect.getsource(search_module.validate_candidates)
    start = source.index('"validation": {')
    receipt = source[start : source.index("}", source.index("parity_policy", start))]
    for field in (
        "region_share_of_e2e",
        "measured_e2e_improvement",
        "impact_basis",
        "projected_end_to_end_speedup",
    ):
        assert field in receipt, f"{field} lost when it left the benchmark object"


# -- search agents are a preset, not a vendor ----------------------------


def test_the_pi_preset_requires_a_model_because_it_serves_many_providers(
    tmp_path: Path,
) -> None:
    from autokernel.optimize.search import BuiltinSearchError, _agent_command

    prompt = tmp_path / "prompt.md"
    prompt.write_text("optimize this", encoding="utf-8")
    with pytest.raises(BuiltinSearchError, match="needs a model"):
        _agent_command(
            None,
            repo_root=tmp_path,
            run_dir=tmp_path,
            candidate_dir=tmp_path,
            prompt_path=prompt,
            last_message=tmp_path / "last.md",
            agent="pi",
        )


def test_an_unknown_agent_names_the_presets_it_knows(tmp_path: Path) -> None:
    from autokernel.optimize.search import BuiltinSearchError, agent_preset

    with pytest.raises(BuiltinSearchError, match="codex, pi"):
        agent_preset("nonexistent-agent")


def test_only_codex_claims_to_confine_its_own_writes() -> None:
    """pi ships no permission system, so the harness digest check is not optional."""
    from autokernel.optimize.search import AGENT_PRESETS

    assert AGENT_PRESETS["codex"]["sandboxed"] is True
    assert AGENT_PRESETS["pi"]["sandboxed"] is False


def test_the_harness_digest_notices_an_edited_verification_module(
    tmp_path: Path,
) -> None:
    from autokernel.optimize.search import _harness_digest

    repo = tmp_path / "repo"
    (repo / "autokernel" / "verification").mkdir(parents=True)
    (repo / "autokernel" / "specs").mkdir(parents=True)
    (repo / "bench.py").write_text("original\n", encoding="utf-8")
    policy = repo / "autokernel" / "verification" / "policy.py"
    policy.write_text("atol = 1e-6\n", encoding="utf-8")

    before = _harness_digest(repo)
    assert _harness_digest(repo) == before

    policy.write_text("atol = 1e-1\n", encoding="utf-8")
    assert _harness_digest(repo) != before
