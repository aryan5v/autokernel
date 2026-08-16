"""Top-level resumable optimize campaign runner."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

from autokernel.workload import load_workload

from .preflight import PreflightError, execute_preflight, write_run_contract
from .report import write_morning_report
from .stages import run_stage
from .state import (
    OptimizeError,
    build_receipt,
    ensure_budget,
    initial_state,
    load_state,
    read_json,
    run_dir_layout,
    save_state,
    stage_is_complete,
    utc_now,
    write_json_atomic,
)
from .types import PIPELINE_STAGES, RECEIPT_SCHEMA_VERSION, TERMINAL_STATUSES, OptimizeConfig

_RESUME_IDENTITY_FIELDS = (
    "fastvideo_checkout",
    "model",
    "workload",
    "baseline",
    "min_e2e_speedup",
    "stage_commands",
    "artifact_dir_name",
    "per_candidate_budget_seconds",
    "search_agent_command",
    "repo_root",
)


def _run_preflight(
    config: OptimizeConfig,
    layout: dict[str, Path],
) -> dict[str, Any]:
    """Validate every precondition before any campaign state is mutated.

    The report is persisted whenever the run directory is usable so a failed
    unattended run still leaves a machine-readable diagnosis behind, but no
    campaign state is created until every check passes.
    """
    resuming = bool(config.resume and layout["state"].is_file())
    report, contract = execute_preflight(
        config,
        contract_path=layout["run_contract"],
        resuming=resuming,
    )
    try:
        write_json_atomic(layout["preflight"], report.as_dict())
    except OSError:
        # The output directory itself is unusable; the findings below say so.
        pass
    if not report.passed:
        raise PreflightError(report.failure_message())
    return contract


def _validate_resume_config(stored: dict[str, Any], config: OptimizeConfig) -> None:
    requested = config.as_dict()
    changed = [
        name
        for name in _RESUME_IDENTITY_FIELDS
        if stored.get(name) != requested.get(name)
    ]
    if changed:
        raise OptimizeError(
            "cannot resume with changed campaign configuration: "
            + ", ".join(changed)
            + "; use a new --output or --no-resume"
        )


def _finite_metric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None



def _variance_blocked(metrics: Mapping[str, Any]) -> str | None:
    """Promotion-blocking reason when the e2e measurement is variance-invalid.

    Only an explicit ``variance_valid: False`` blocks: legacy campaigns
    without a measured CV are untouched, while any campaign whose native arm
    exceeds the declared ceiling is recorded but unusable as promotion
    evidence.
    """
    if metrics.get("variance_valid") is False:
        return str(
            metrics.get("variance_reason")
            or "native-arm variance exceeds the declared ceiling"
        )
    return None


def _decide_terminal(
    state: dict[str, Any],
    stage_results: dict[str, dict[str, Any]],
    *,
    min_e2e_speedup: float,
) -> tuple[str, str]:
    """Return (terminal, message). Never promote from isolated speedup alone."""
    e2e = stage_results.get("end_to_end_validate") or {}
    discover = stage_results.get("discover") or {}
    isolated = stage_results.get("isolated_validate") or {}

    # Explicit stage recommendations win when present.
    for stage_name in (
        "finalize",
        "end_to_end_validate",
        "discover",
        "specgen",
        "search",
        "isolated_validate",
        "package",
    ):
        rec = (stage_results.get(stage_name) or {}).get("recommendation")
        if rec in {"promoted", "no_worthwhile_candidate", "failed"}:
            if rec == "promoted":
                # Hard gate: e2e metrics required.
                metrics = e2e.get("metrics") or {}
                raw_speedup = metrics.get("end_to_end_speedup")
                speedup = _finite_metric(raw_speedup)
                classification = metrics.get("classification")
                if speedup is None or speedup < min_e2e_speedup:
                    return (
                        "no_worthwhile_candidate",
                        (
                            f"promotion blocked: end-to-end speedup below threshold "
                            f"(isolated speedup={(isolated.get('metrics') or {}).get('isolated_speedup')!r} "
                            f"is not sufficient)"
                        ),
                    )
                if classification != "improved":
                    return (
                        "no_worthwhile_candidate",
                        f"promotion blocked: e2e classification={classification!r}",
                    )
                variance_reason = _variance_blocked(metrics)
                if variance_reason is not None:
                    return (
                        "no_worthwhile_candidate",
                        f"promotion blocked: {variance_reason}",
                    )
                return (
                    "promoted",
                    f"promoted with end-to-end speedup={speedup}",
                )
            if rec == "no_worthwhile_candidate":
                return (
                    "no_worthwhile_candidate",
                    str(
                        (stage_results.get(stage_name) or {}).get("message")
                        or "no worthwhile candidate"
                    ),
                )
            return ("failed", str((stage_results.get(stage_name) or {}).get("message") or "failed"))

    candidates = state.get("candidates") or discover.get("candidates") or []
    if not candidates:
        return (
            "no_worthwhile_candidate",
            "discovery produced no search-worthy candidates",
        )

    metrics = e2e.get("metrics") or {}
    speedup = _finite_metric(metrics.get("end_to_end_speedup"))
    classification = metrics.get("classification")
    if (
        speedup is not None
        and speedup >= min_e2e_speedup
        and classification == "improved"
    ):
        variance_reason = _variance_blocked(metrics)
        if variance_reason is not None:
            return (
                "no_worthwhile_candidate",
                f"promotion blocked: {variance_reason}",
            )
        return ("promoted", f"promoted with end-to-end speedup={speedup}")
    if classification in {"neutral", "regressed"} or (
        speedup is not None and speedup < min_e2e_speedup
    ):
        return (
            "no_worthwhile_candidate",
            (
                f"end-to-end result does not meet promotion threshold "
                f"(isolated speedup={(isolated.get('metrics') or {}).get('isolated_speedup')!r} "
                f"ignored)"
            ),
        )
    return ("failed", "campaign finished without a clear promotion decision")


def run_optimize(
    config: OptimizeConfig,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Execute or resume the full optimize pipeline.

    ``preflight_only`` validates every precondition, writes ``preflight.json``,
    and returns without running a stage or creating campaign state.
    """
    layout = run_dir_layout(config.output)
    contract = _run_preflight(config, layout)

    if preflight_only:
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "terminal": "preflight_passed",
            "status": "preflight_passed",
            "message": "preflight passed; no stages were run",
            "model": config.model,
            "workload": str(config.workload),
            "output": str(config.output),
            "preflight": str(layout["preflight"]),
            "pipeline_stages": list(PIPELINE_STAGES),
        }

    for key in ("stages", "logs", "candidates", "artifacts", "commands"):
        layout[key].mkdir(parents=True, exist_ok=True)

    if config.resume and layout["state"].is_file():
        if not layout["config"].is_file():
            raise OptimizeError(
                f"resume state exists but config is missing: {layout['config']}"
            )
        stored_config = read_json(layout["config"])
        if not isinstance(stored_config, dict):
            raise OptimizeError("stored campaign config must be a JSON object")
        _validate_resume_config(stored_config, config)
        state = load_state(layout["state"])
        status = state.get("status")
        if status in {"failed", "budget_exhausted"}:
            # --resume is specifically the recovery path for an interrupted or
            # repaired campaign. A previous failure receipt is not permanently
            # terminal: preserve every durable, successful stage and reopen the
            # first incomplete one, rather than replaying the old receipt and
            # discarding hours of GPU work because one stage died.
            failed_stages = dict(state.get("failed_stages") or {})
            state["failed_stages"] = {
                stage: message
                for stage, message in failed_stages.items()
                if stage_is_complete(state, stage)
            }
            state["status"] = "running"
            state["terminal"] = None
            state.setdefault("messages", []).append(
                {"at": utc_now(), "text": f"resumed after terminal status {status}"}
            )
            save_state(layout["state"], state)
        if state.get("status") in TERMINAL_STATUSES:
            # A discovery-only stop is terminal for the *same* requested stop
            # stage, so a nightly rerun is idempotent. Resuming with a later
            # (or no) stop stage continues the campaign past it instead.
            continues = (
                state.get("status") == "discovery_complete"
                and stored_config.get("stop_after_stage") != config.stop_after_stage
            )
            if not continues:
                # Already terminal — rewrite morning report and return receipt.
                if layout["receipt"].is_file():
                    receipt = read_json(layout["receipt"])
                else:
                    receipt = build_receipt(
                        state,
                        config,
                        terminal=str(state.get("terminal") or state.get("status")),
                        message="campaign already terminal on resume",
                    )
                    write_json_atomic(layout["receipt"], receipt)
                write_morning_report(layout["morning_report"], receipt=receipt)
                return receipt
    else:
        # A deliberately fresh run must not expose an old terminal receipt
        # while new stages are still running.
        layout["receipt"].unlink(missing_ok=True)
        layout["morning_report"].unlink(missing_ok=True)
        # A deliberately fresh campaign supersedes any previous contract; a
        # resumed one never rewrites it.
        layout["run_contract"].unlink(missing_ok=True)
        write_run_contract(layout["run_contract"], contract)
        write_json_atomic(layout["config"], config.as_dict())
        state = initial_state(config)
        # Wall-clock deadline for the campaign budget.
        state["budget_deadline_epoch"] = time.time() + config.budget_hours * 3600.0
        save_state(layout["state"], state)

    # A resume receives a fresh wall-clock allowance. Duration is an
    # operational control, so extending it does not change run identity.
    if config.resume:
        state["budget_deadline_epoch"] = time.time() + config.budget_hours * 3600.0
        save_state(layout["state"], state)

    stage_results: dict[str, dict[str, Any]] = {}
    # Reload completed stage results for decision logic.
    for stage in PIPELINE_STAGES:
        result_path = layout["stages"] / stage / "result.json"
        if stage_is_complete(state, stage) and result_path.is_file():
            stage_results[stage] = read_json(result_path)

    if config.stop_after_stage is not None and config.stop_after_stage not in PIPELINE_STAGES:
        raise OptimizeError(
            f"stop_after_stage must be one of {list(PIPELINE_STAGES)}; "
            f"got {config.stop_after_stage!r}"
        )

    stopped_early = False
    try:
        for stage in PIPELINE_STAGES:
            ensure_budget(state)
            if config.resume and stage_is_complete(state, stage):
                # Already done. The stop check below still applies, so a
                # resumed discovery-only campaign terminates at the same
                # stage instead of drifting past it.
                pass
            else:
                payload = run_stage(
                    stage,
                    config=config,
                    state=state,
                    state_path=layout["state"],
                )
                stage_results[stage] = payload
                # Early exit if discover declares nothing worth searching.
                if stage == "discover" and payload.get("recommendation") == "no_worthwhile_candidate":
                    break
                if stage == "specgen" and payload.get("recommendation") == "no_worthwhile_candidate":
                    break
                if stage in {"search", "isolated_validate"} and payload.get(
                    "recommendation"
                ) == "no_worthwhile_candidate":
                    break
            if stage == config.stop_after_stage:
                stopped_early = True
                break
    except OptimizeError as exc:
        if "budget exhausted" in str(exc).lower():
            terminal = "budget_exhausted"
        else:
            terminal = "failed"
        state["status"] = terminal
        state["terminal"] = terminal
        save_state(layout["state"], state)
        receipt = build_receipt(state, config, terminal=terminal, message=str(exc))
        write_json_atomic(layout["receipt"], receipt)
        write_morning_report(layout["morning_report"], receipt=receipt)
        return receipt

    workload = load_workload(config.workload)
    workload_threshold = (
        workload.performance.min_end_to_end_speedup
        if workload.performance is not None
        else config.min_e2e_speedup
    )
    discover_recommendation = (stage_results.get("discover") or {}).get("recommendation")
    if stopped_early and discover_recommendation != "no_worthwhile_candidate":
        # Stopped on request, not on a verdict. This is the discovery-only
        # nightly path: the receipt records how far the run got and stays
        # terminal so a rerun is idempotent instead of drifting onward.
        terminal = "discovery_complete"
        completed = [name for name in PIPELINE_STAGES if stage_is_complete(state, name)]
        candidate_count = len(state.get("candidates") or [])
        message = (
            f"stopped after {config.stop_after_stage} as requested; "
            f"completed stages: {', '.join(completed)}; "
            f"candidates discovered: {candidate_count}"
        )
    else:
        terminal, message = _decide_terminal(
            state,
            stage_results,
            min_e2e_speedup=max(config.min_e2e_speedup, workload_threshold),
        )
    state["status"] = terminal
    state["terminal"] = terminal
    if terminal in {"promoted", "no_worthwhile_candidate"}:
        final_candidate_status = (
            "promoted" if terminal == "promoted" else "not_promoted"
        )
        state["candidates"] = [
            {**candidate, "status": final_candidate_status}
            if isinstance(candidate, dict)
            else candidate
            for candidate in (state.get("candidates") or [])
        ]
    state.setdefault("messages", []).append({"at": utc_now(), "text": message})
    save_state(layout["state"], state)
    receipt = build_receipt(state, config, terminal=terminal, message=message)
    write_json_atomic(layout["receipt"], receipt)
    write_morning_report(
        layout["morning_report"],
        receipt=receipt,
        stage_summaries={
            name: payload.get("message") for name, payload in stage_results.items()
        },
    )
    return receipt
