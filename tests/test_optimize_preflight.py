"""CPU contract tests for fail-closed preflight and the immutable run contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import autokernel.optimize.preflight as preflight_module
from autokernel.optimize import (
    PREFLIGHT_SCHEMA_VERSION,
    RUN_CONTRACT_SCHEMA_VERSION,
    OptimizeConfig,
    OptimizeError,
    PreflightError,
    compare_run_contract,
    execute_preflight,
    run_optimize,
)
from autokernel.optimize.preflight import command_identity
from conftest import make_fastvideo_checkout, make_workload

SECRET = "sk-live-DO-NOT-PERSIST-0123456789"


def _config(tmp_path: Path, repo_root: Path, **overrides) -> OptimizeConfig:
    """Build a config, creating only the default inputs that were not supplied.

    Defaults are materialized lazily so a test that deliberately edits or
    replaces the workload or checkout is not silently overwritten by a later
    ``_config`` call.
    """
    values: dict = {
        "model": "test/model",
        "output": tmp_path / "run",
        "budget_hours": 1.0,
        "repo_root": repo_root,
    }
    values.update(overrides)
    if "fastvideo_checkout" not in values:
        values["fastvideo_checkout"] = make_fastvideo_checkout(tmp_path)
    if "workload" not in values:
        workload = tmp_path / "workload.json"
        if not workload.is_file():
            make_workload(workload)
        values["workload"] = workload
    return OptimizeConfig(**values)


def _simulate(monkeypatch: pytest.MonkeyPatch, outcome: str = "promoted") -> None:
    monkeypatch.setenv("MOTIONKERNEL_SIMULATE", "1")
    monkeypatch.setenv("MOTIONKERNEL_SIMULATE_OUTCOME", outcome)


def _preflight(config: OptimizeConfig, *, resuming: bool = False):
    return execute_preflight(
        config,
        contract_path=config.output / "run_contract.json",
        resuming=resuming,
    )


def _codes(report) -> set[str]:
    return set(report.reason_codes)


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


def test_preflight_passes_and_records_identity(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch)
    config = _config(tmp_path, repo_root)

    report, contract = _preflight(config)

    assert report.passed
    assert report.reason_codes == []
    payload = report.as_dict()
    assert payload["schema_version"] == PREFLIGHT_SCHEMA_VERSION
    assert payload["status"] == "pass"
    assert payload["model"] == "test/model"
    assert payload["workload"]["workload_id"] == "cpu-contract"
    assert payload["workload"]["sha256"]
    assert payload["policy"]["baseline"] == "eager"
    assert contract["schema_version"] == RUN_CONTRACT_SCHEMA_VERSION
    assert "budget_hours" not in contract["policy"]
    assert "resume" not in contract["policy"]


def test_preflight_warns_when_output_space_is_low(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch)
    monkeypatch.setattr(
        preflight_module.shutil,
        "disk_usage",
        lambda _path: preflight_module.shutil._ntuple_diskusage(
            20 * 1024**3,
            19 * 1024**3,
            1024**3,
        ),
    )

    report, _ = _preflight(_config(tmp_path, repo_root))

    assert report.passed
    assert "output_free_space_low" in {
        finding.code for finding in report.warnings
    }
    assert report.sections["output_storage"]["free_bytes"] == 1024**3


def test_preflight_rejects_when_output_space_is_insufficient(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch)
    monkeypatch.setattr(
        preflight_module.shutil,
        "disk_usage",
        lambda _path: preflight_module.shutil._ntuple_diskusage(
            20 * 1024**3,
            20 * 1024**3 - 256 * 1024**2,
            256 * 1024**2,
        ),
    )

    report, _ = _preflight(_config(tmp_path, repo_root))

    assert not report.passed
    assert _codes(report) == {"output_free_space_insufficient"}


def test_preflight_only_writes_report_without_campaign_state(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch)
    config = _config(tmp_path, repo_root)

    receipt = run_optimize(config, preflight_only=True)

    assert receipt["terminal"] == "preflight_passed"
    assert (config.output / "preflight.json").is_file()
    # No campaign was started.
    assert not (config.output / "state.json").exists()
    assert not (config.output / "config.json").exists()
    assert not (config.output / "receipt.json").exists()
    assert not (config.output / "run_contract.json").exists()
    assert not (config.output / "stages").exists()


def test_failed_preflight_only_creates_no_misleading_state(
    tmp_path: Path, repo_root: Path
):
    config = _config(tmp_path, repo_root, model="   ")

    with pytest.raises(PreflightError, match="model_empty"):
        run_optimize(config, preflight_only=True)

    report = json.loads((config.output / "preflight.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "model_empty" in report["reason_codes"]
    assert not (config.output / "state.json").exists()
    assert not (config.output / "run_contract.json").exists()
    assert not (config.output / "stages").exists()


def test_campaign_writes_run_contract(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch)
    config = _config(tmp_path, repo_root)

    run_optimize(config)

    contract = json.loads(
        (config.output / "run_contract.json").read_text(encoding="utf-8")
    )
    assert contract["model"] == "test/model"
    assert contract["workload"]["workload_id"] == "cpu-contract"
    assert contract["policy"]["min_e2e_speedup"] == pytest.approx(1.01)


# ---------------------------------------------------------------------------
# Missing checkouts, workloads, executables
# ---------------------------------------------------------------------------


def test_missing_fastvideo_checkout_fails_closed(tmp_path: Path, repo_root: Path):
    config = _config(tmp_path, repo_root, fastvideo_checkout=tmp_path / "absent")
    report, _ = _preflight(config)
    assert "fastvideo_checkout_missing" in _codes(report)


def test_incomplete_fastvideo_checkout_fails_closed(tmp_path: Path, repo_root: Path):
    checkout = make_fastvideo_checkout(tmp_path / "partial", complete=False)
    config = _config(tmp_path, repo_root, fastvideo_checkout=checkout)
    report, _ = _preflight(config)
    assert "fastvideo_structure_invalid" in _codes(report)
    detail = next(f for f in report.errors if f.code == "fastvideo_structure_invalid")
    assert "generation_launcher.py" in detail.detail["missing"][0]


def test_fastvideo_checkout_that_is_a_file_fails_closed(
    tmp_path: Path, repo_root: Path
):
    target = tmp_path / "not-a-dir"
    target.write_text("", encoding="utf-8")
    config = _config(tmp_path, repo_root, fastvideo_checkout=target)
    report, _ = _preflight(config)
    assert "fastvideo_checkout_not_a_directory" in _codes(report)


def test_missing_workload_fails_closed(tmp_path: Path, repo_root: Path):
    config = _config(tmp_path, repo_root, workload=tmp_path / "absent.json")
    report, _ = _preflight(config)
    assert "workload_missing" in _codes(report)


def test_malformed_workload_fails_the_real_schema(tmp_path: Path, repo_root: Path):
    workload = tmp_path / "bad.json"
    workload.write_text(
        json.dumps({"schema_version": 1, "workload_id": "x"}), encoding="utf-8"
    )
    config = _config(tmp_path, repo_root, workload=workload)
    report, _ = _preflight(config)
    assert "workload_invalid" in _codes(report)


def test_output_path_occupied_by_a_file_fails_closed(tmp_path: Path, repo_root: Path):
    occupied = tmp_path / "run"
    occupied.write_text("", encoding="utf-8")
    config = _config(tmp_path, repo_root, output=occupied)
    report, _ = _preflight(config)
    assert "output_not_a_directory" in _codes(report)


def test_missing_repo_entry_point_fails_closed(tmp_path: Path):
    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    config = _config(tmp_path, empty_repo)
    report, _ = _preflight(config)
    assert "repo_entry_point_missing" in _codes(report)


def test_missing_search_agent_fails_closed(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MOTIONKERNEL_SIMULATE", raising=False)
    monkeypatch.setattr(
        "autokernel.optimize.preflight.shutil.which",
        lambda name: None if name == "codex" else f"/usr/bin/{name}",
    )
    config = _config(tmp_path, repo_root)
    report, _ = _preflight(config)
    assert "search_agent_missing" in _codes(report)


def test_configured_search_agent_must_be_executable(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MOTIONKERNEL_SIMULATE", raising=False)
    config = _config(
        tmp_path,
        repo_root,
        search_agent_command=[str(tmp_path / "no-such-agent"), "run"],
    )
    report, _ = _preflight(config)
    assert "search_agent_missing" in _codes(report)


def test_search_agent_check_is_skipped_when_campaign_stops_before_search(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MOTIONKERNEL_SIMULATE", raising=False)
    monkeypatch.setattr(
        "autokernel.optimize.preflight.shutil.which",
        lambda name: None if name == "codex" else f"/usr/bin/{name}",
    )
    config = _config(tmp_path, repo_root, stop_after_stage="discover")
    report, _ = _preflight(config)
    assert report.passed
    assert report.sections["search_agent"]["checked"] is False
    assert report.sections["search_agent"]["skipped_reason"] == (
        "campaign stops after discover; search never runs"
    )


def test_search_agent_check_is_skipped_and_recorded_in_simulation(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch)
    monkeypatch.setattr(
        "autokernel.optimize.preflight.shutil.which",
        lambda name: None if name == "codex" else f"/usr/bin/{name}",
    )
    config = _config(tmp_path, repo_root)
    report, _ = _preflight(config)
    assert report.passed
    assert report.sections["search_agent"]["checked"] is False
    assert report.sections["search_agent"]["skipped_reason"] == "simulation"


# ---------------------------------------------------------------------------
# Stage commands
# ---------------------------------------------------------------------------


def test_unknown_stage_command_name_fails_closed(tmp_path: Path, repo_root: Path):
    config = _config(
        tmp_path, repo_root, stage_commands={"not_a_stage": [sys.executable, "-c", ""]}
    )
    report, _ = _preflight(config)
    assert "stage_command_unknown_stage" in _codes(report)


def test_empty_stage_command_fails_closed(tmp_path: Path, repo_root: Path):
    config = _config(tmp_path, repo_root, stage_commands={"baseline": []})
    report, _ = _preflight(config)
    assert "stage_command_empty" in _codes(report)


def test_unknown_stage_command_placeholder_fails_closed(
    tmp_path: Path, repo_root: Path
):
    config = _config(
        tmp_path,
        repo_root,
        stage_commands={"baseline": [sys.executable, "-c", "pass", "{fastvideo}"]},
    )
    report, _ = _preflight(config)
    assert "stage_command_placeholder_invalid" in _codes(report)


def test_known_placeholders_and_literal_braces_are_accepted(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch)
    config = _config(
        tmp_path,
        repo_root,
        stage_commands={
            "baseline": [sys.executable, "-c", "print({'a': 1})", "{model}", "{run_dir}"]
        },
    )
    report, _ = _preflight(config)
    assert report.passed


def test_missing_stage_command_executable_fails_closed(
    tmp_path: Path, repo_root: Path
):
    config = _config(
        tmp_path,
        repo_root,
        stage_commands={"baseline": [str(tmp_path / "nope"), "--go"]},
    )
    report, _ = _preflight(config)
    assert "stage_command_executable_missing" in _codes(report)


# ---------------------------------------------------------------------------
# Numeric policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_non_finite_budget_fails_closed(tmp_path: Path, repo_root: Path, value: float):
    report, _ = _preflight(_config(tmp_path, repo_root, budget_hours=value))
    assert "budget_invalid" in _codes(report)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -2.5])
def test_non_finite_promotion_threshold_fails_closed(
    tmp_path: Path, repo_root: Path, value: float
):
    report, _ = _preflight(_config(tmp_path, repo_root, min_e2e_speedup=value))
    assert "promotion_threshold_invalid" in _codes(report)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -30.0])
def test_non_finite_candidate_timeout_fails_closed(
    tmp_path: Path, repo_root: Path, value: float
):
    report, _ = _preflight(
        _config(tmp_path, repo_root, per_candidate_budget_seconds=value)
    )
    assert "candidate_timeout_invalid" in _codes(report)


def test_invalid_baseline_fails_closed(tmp_path: Path, repo_root: Path):
    report, _ = _preflight(_config(tmp_path, repo_root, baseline="inductor"))
    assert "baseline_invalid" in _codes(report)


@pytest.mark.parametrize("value", ["", ".", "..", "a/b"])
def test_invalid_artifact_dir_fails_closed(
    tmp_path: Path, repo_root: Path, value: str
):
    report, _ = _preflight(_config(tmp_path, repo_root, artifact_dir_name=value))
    assert "artifact_dir_invalid" in _codes(report)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_command_identity_never_exposes_arguments():
    identity = command_identity(["/usr/local/bin/codex", "--api-key", SECRET])
    assert identity["program"] == "codex"
    assert identity["argc"] == 3
    assert SECRET not in json.dumps(identity)


def test_no_secret_material_is_persisted(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch)
    config = _config(
        tmp_path,
        repo_root,
        stage_commands={"baseline": [sys.executable, "-c", "pass", f"--token={SECRET}"]},
        search_agent_command=[sys.executable, "-c", "pass", f"--api-key={SECRET}"],
    )

    run_optimize(config)

    for name in ("preflight.json", "run_contract.json"):
        text = (config.output / name).read_text(encoding="utf-8")
        assert SECRET not in text, f"{name} leaked secret material"
        # The digest still pins the command for contract comparison.
        assert "sha256" in text


# ---------------------------------------------------------------------------
# Run contract comparison
# ---------------------------------------------------------------------------


def _start_campaign(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch, **overrides
) -> OptimizeConfig:
    _simulate(monkeypatch)
    config = _config(tmp_path, repo_root, **overrides)
    run_optimize(config)
    return config


def _interrupt(config: OptimizeConfig) -> None:
    """Model a process interruption so the campaign is resumable, not terminal."""
    from autokernel.optimize.types import PIPELINE_STAGES

    state_path = config.output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state["terminal"] = None
    state["completed_stages"] = list(PIPELINE_STAGES[:3])
    state["stage_records"] = {
        name: state["stage_records"][name] for name in PIPELINE_STAGES[:3]
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (config.output / "receipt.json").unlink(missing_ok=True)


def test_unchanged_resume_succeeds(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(tmp_path, repo_root, monkeypatch)
    contract_before = (config.output / "run_contract.json").read_bytes()
    _interrupt(config)

    receipt = run_optimize(_config(tmp_path, repo_root))

    assert receipt["terminal"] == "promoted"
    # The contract is immutable: a resume never rewrites it.
    assert (config.output / "run_contract.json").read_bytes() == contract_before


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"model": "other/model"}, "contract_mismatch_model"),
        ({"baseline": "compile"}, "contract_mismatch_baseline"),
        ({"min_e2e_speedup": 1.5}, "contract_mismatch_promotion_threshold"),
        (
            {"per_candidate_budget_seconds": 60.0},
            "contract_mismatch_candidate_timeout",
        ),
        ({"artifact_dir_name": "other"}, "contract_mismatch_artifact_dir"),
    ],
)
def test_resume_rejects_changed_policy(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict,
    code: str,
):
    config = _start_campaign(tmp_path, repo_root, monkeypatch)
    _interrupt(config)

    with pytest.raises(PreflightError, match=code):
        run_optimize(_config(tmp_path, repo_root, **overrides))


def test_resume_accepts_changed_budget_and_renews_deadline(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(tmp_path, repo_root, monkeypatch)
    contract_before = (config.output / "run_contract.json").read_bytes()
    _interrupt(config)
    state_path = config.output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["budget_deadline_epoch"] = 0.0
    state_path.write_text(json.dumps(state), encoding="utf-8")

    receipt = run_optimize(_config(tmp_path, repo_root, budget_hours=2.0))

    assert receipt["terminal"] == "promoted"
    assert (config.output / "run_contract.json").read_bytes() == contract_before
    renewed = json.loads(state_path.read_text(encoding="utf-8"))
    assert renewed["budget_deadline_epoch"] > 0.0


def test_resume_rejects_changed_workload_content(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(tmp_path, repo_root, monkeypatch)
    _interrupt(config)
    # Same path, different content: exactly the silent drift the contract exists
    # to catch, and something a path comparison cannot see.
    make_workload(config.workload, description="edited overnight")

    with pytest.raises(PreflightError, match="contract_mismatch_workload_content"):
        run_optimize(_config(tmp_path, repo_root, workload=config.workload))


def test_resume_rejects_changed_fastvideo_checkout(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(tmp_path, repo_root, monkeypatch)
    _interrupt(config)
    other = make_fastvideo_checkout(tmp_path / "elsewhere")

    with pytest.raises(PreflightError, match="contract_mismatch_fastvideo_checkout"):
        run_optimize(_config(tmp_path, repo_root, fastvideo_checkout=other))


def test_resume_rejects_changed_stage_commands(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(
        tmp_path,
        repo_root,
        monkeypatch,
        stage_commands={"baseline": [sys.executable, "-c", "pass"]},
    )
    assert (config.output / "run_contract.json").is_file()

    # Preflight compares the contract before the terminal-resume shortcut, so
    # drift is caught whatever state the previous campaign ended in.
    with pytest.raises(PreflightError, match="contract_mismatch_stage_commands"):
        run_optimize(
            _config(
                tmp_path,
                repo_root,
                stage_commands={"baseline": [sys.executable, "-c", "pass  "]},
            )
        )


def test_resume_rejects_changed_search_agent_command(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(
        tmp_path,
        repo_root,
        monkeypatch,
        search_agent_command=[sys.executable, "-c", "pass"],
    )
    assert (config.output / "run_contract.json").is_file()

    with pytest.raises(
        PreflightError, match="contract_mismatch_search_agent_command"
    ):
        run_optimize(
            _config(
                tmp_path,
                repo_root,
                search_agent_command=[sys.executable, "-c", "print(1)"],
            )
        )


def test_changed_resume_fails_before_any_stage_runs(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(tmp_path, repo_root, monkeypatch)
    _interrupt(config)
    search_dir = config.output / "stages" / "search"
    import shutil as _shutil

    _shutil.rmtree(search_dir, ignore_errors=True)
    baseline_result = config.output / "stages" / "baseline" / "result.json"
    before = baseline_result.read_bytes()

    with pytest.raises(PreflightError, match="contract_mismatch_model"):
        run_optimize(_config(tmp_path, repo_root, model="drifted/model"))

    # No stage ran and no completed work was disturbed.
    assert not search_dir.exists()
    assert baseline_result.read_bytes() == before


def test_resume_without_a_contract_fails_closed(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(tmp_path, repo_root, monkeypatch)
    _interrupt(config)
    (config.output / "run_contract.json").unlink()

    with pytest.raises(PreflightError, match="contract_missing"):
        run_optimize(_config(tmp_path, repo_root))


def test_no_resume_supersedes_a_previous_contract(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _start_campaign(tmp_path, repo_root, monkeypatch)
    before = json.loads(
        (config.output / "run_contract.json").read_text(encoding="utf-8")
    )

    run_optimize(_config(tmp_path, repo_root, model="fresh/model", resume=False))

    after = json.loads(
        (config.output / "run_contract.json").read_text(encoding="utf-8")
    )
    assert before["model"] == "test/model"
    assert after["model"] == "fresh/model"


def test_checkout_identity_prefers_commit_over_path():
    stored = {"commit": "abc123", "path_digest": "one"}
    moved = {"commit": "abc123", "path_digest": "two"}
    rebased = {"commit": "def456", "path_digest": "one"}

    assert compare_run_contract(
        {"fastvideo": stored, "policy": {}, "commands": {}},
        {"fastvideo": moved, "policy": {}, "commands": {}},
    ) == []
    codes = [
        f.code
        for f in compare_run_contract(
            {"fastvideo": stored, "policy": {}, "commands": {}},
            {"fastvideo": rebased, "policy": {}, "commands": {}},
        )
    ]
    assert codes == ["contract_mismatch_fastvideo_checkout"]


def test_every_mismatch_code_is_distinct_and_stable():
    from autokernel.optimize.preflight import CONTRACT_MISMATCH_CODES

    codes = list(CONTRACT_MISMATCH_CODES.values())
    assert len(codes) == len(set(codes))
    assert all(code.startswith("contract_mismatch_") for code in codes)


def test_preflight_error_is_an_optimize_error(tmp_path: Path, repo_root: Path):
    with pytest.raises(OptimizeError):
        run_optimize(_config(tmp_path, repo_root, baseline="nope"))


# -- the harness is evidence identity ------------------------------------
#
# Six gap-7 attempts each edited the harness between runs, so no run measured
# the same system as the one before it and nothing accumulated. The contract
# recorded the MotionKernel commit from the start but never compared it.


def test_a_changed_harness_commit_fails_a_resume_closed():
    stored = {"commit": "aaa111", "path_digest": "one"}
    edited = {"commit": "bbb222", "path_digest": "one"}

    codes = [
        f.code
        for f in compare_run_contract(
            {"motionkernel": stored, "policy": {}, "commands": {}},
            {"motionkernel": edited, "policy": {}, "commands": {}},
        )
    ]
    assert codes == ["contract_mismatch_harness"]


def test_a_moved_harness_checkout_at_the_same_commit_still_resumes():
    stored = {"commit": "aaa111", "path_digest": "one"}
    moved = {"commit": "aaa111", "path_digest": "two"}

    assert (
        compare_run_contract(
            {"motionkernel": stored, "policy": {}, "commands": {}},
            {"motionkernel": moved, "policy": {}, "commands": {}},
        )
        == []
    )
