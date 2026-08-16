"""Per-artifact end-to-end A/B trials.

Run ltx-v1-overnight-20260801-r4-sol enabled four artifacts simultaneously,
made 56 candidate calls, failed byte_equal parity and regressed end-to-end from
3.2818s to 3.9410s. Nothing in that evidence says *which* artifact broke parity
or which one cost the time, and recovering that after the fact means bisecting
the artifact directory by hand.

This module runs one full FastVideo generation per artifact, with only that
artifact admitted, and records the same measurements for every trial:

    artifact id, dispatch calls, candidate calls, runtime fallbacks,
    parity result, median wall seconds, speedup, peak memory

so a single pass produces a table that attributes parity and latency to
individual artifacts. A combined trial is worth running only over the artifacts
that were individually safe and individually worthwhile.

The isolation itself is FastVideo's: ``FASTVIDEO_OPTIMIZATION_ARTIFACT_ENABLE``
admits only the named artifact ids from an otherwise unchanged directory, so no
trial restages or mutates the artifact tree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autokernel.artifact.kinds import MODULE, evaluate_execution_signal
from autokernel.verification.ceiling import CeilingError, gate_basis_for_workload
from autokernel.workload import ParitySpec
from autokernel.workload.launcher import run_mode
from autokernel.workload.result import (
    classify_end_to_end,
    compare_frame_outputs,
    load_generation_result,
)

__all__ = [
    "IsolationReport",
    "TrialRecord",
    "artifact_ids_in",
    "artifact_kinds_in",
    "dispatch_counts_for",
    "run_isolation_trials",
]

ISOLATION_SCHEMA_VERSION = 1


class IsolationError(RuntimeError):
    """A per-artifact trial could not be run or measured."""


@dataclass(frozen=True)
class TrialRecord:
    """One full generation with a known set of artifacts admitted."""

    trial: str
    artifact_ids: tuple[str, ...]
    status: str
    dispatch_calls: int = 0
    candidate_calls: int = 0
    runtime_fallbacks: int = 0
    scopes_selected: tuple[str, ...] = ()
    target_kinds: tuple[str, ...] = ()
    execution_evidence: Mapping[str, Any] = field(default_factory=dict)
    gate_basis: Mapping[str, Any] = field(default_factory=dict)
    parity_passed: bool | None = None
    parity_reason: str = ""
    median_wall_seconds: float | None = None
    native_median_wall_seconds: float | None = None
    end_to_end_speedup: float | None = None
    peak_memory_mb: float | None = None
    peak_memory_regression: float | None = None
    error: str = ""

    @property
    def safe(self) -> bool:
        """Parity held and no candidate fell back mid-run."""
        return (
            self.status == "ok"
            and self.parity_passed is True
            and self.runtime_fallbacks == 0
        )

    @property
    def ran(self) -> bool:
        """Did every artifact in this trial demonstrably execute?

        Each kind declares its own proof (``artifact.kinds``): a kernel counts
        candidate calls, an attention backend echoes which implementation ran,
        a schedule transform counts hook invocations. Asserting the kernel
        counter for all of them would read "did not run" as "ran" for the kinds
        that never touch it -- and reading a timing whose intervention never
        happened is the failure this whole check exists for.
        """
        return not self.execution_failures

    @property
    def execution_failures(self) -> tuple[str, ...]:
        """Per-kind reasons this trial cannot be read as differentiated."""
        kinds = self.target_kinds or (MODULE,)
        # A record always carries the dispatch counter as a field, so it is
        # evidence even when no richer mapping was collected. Kinds whose
        # signal is not the dispatch counter still find nothing here, which is
        # the correct answer rather than a convenient one.
        evidence = dict(self.execution_evidence) or {
            "candidate_calls": self.candidate_calls
        }
        reasons: list[str] = []
        for kind in kinds:
            ran, reason = evaluate_execution_signal(kind, evidence)
            if not ran:
                reasons.append(f"{kind}: {reason}")
        return tuple(reasons)

    @property
    def worthwhile(self) -> bool:
        """Demonstrably ran and did not make the model slower."""
        return (
            self.ran
            and self.end_to_end_speedup is not None
            and self.end_to_end_speedup > 1.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            "artifact_ids": list(self.artifact_ids),
            "status": self.status,
            "dispatch_calls": self.dispatch_calls,
            "candidate_calls": self.candidate_calls,
            "runtime_fallbacks": self.runtime_fallbacks,
            "scopes_selected": list(self.scopes_selected),
            "target_kinds": list(self.target_kinds),
            "execution_evidence": dict(self.execution_evidence),
            "ran": self.ran,
            "execution_failures": list(self.execution_failures),
            "gate_basis": dict(self.gate_basis),
            "parity_passed": self.parity_passed,
            "parity_reason": self.parity_reason,
            "median_wall_seconds": self.median_wall_seconds,
            "native_median_wall_seconds": self.native_median_wall_seconds,
            "end_to_end_speedup": self.end_to_end_speedup,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_memory_regression": self.peak_memory_regression,
            "safe": self.safe,
            "worthwhile": self.worthwhile,
            "error": self.error,
        }


@dataclass
class IsolationReport:
    """Every trial in one isolation pass, plus the resulting recommendation."""

    trials: list[TrialRecord] = field(default_factory=list)
    native_median_wall_seconds: float | None = None

    @property
    def safe_and_worthwhile(self) -> tuple[str, ...]:
        """Artifacts that individually preserved parity and paid for themselves."""
        ids: list[str] = []
        for trial in self.trials:
            if len(trial.artifact_ids) != 1:
                continue
            if trial.safe and trial.worthwhile:
                ids.extend(trial.artifact_ids)
        return tuple(sorted(set(ids)))

    @property
    def parity_offenders(self) -> tuple[str, ...]:
        ids: list[str] = []
        for trial in self.trials:
            if len(trial.artifact_ids) == 1 and trial.parity_passed is False:
                ids.extend(trial.artifact_ids)
        return tuple(sorted(set(ids)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "isolation_schema_version": ISOLATION_SCHEMA_VERSION,
            "native_median_wall_seconds": self.native_median_wall_seconds,
            "trials": [trial.as_dict() for trial in self.trials],
            "safe_and_worthwhile": list(self.safe_and_worthwhile),
            "parity_offenders": list(self.parity_offenders),
        }

    def table(self) -> str:
        """A fixed-width summary suitable for a run report."""
        header = (
            f"{'trial':28s} {'disp':>5s} {'cand':>5s} {'fb':>3s} "
            f"{'parity':>7s} {'median_s':>9s} {'speedup':>8s} {'peak_mb':>9s}"
        )
        lines = [header, "-" * len(header)]
        for trial in self.trials:
            parity = (
                "n/a" if trial.parity_passed is None
                else ("pass" if trial.parity_passed else "FAIL")
            )
            median = "-" if trial.median_wall_seconds is None else f"{trial.median_wall_seconds:9.4f}"
            speedup = "-" if trial.end_to_end_speedup is None else f"{trial.end_to_end_speedup:8.4f}"
            peak = "-" if trial.peak_memory_mb is None else f"{trial.peak_memory_mb:9.1f}"
            lines.append(
                f"{trial.trial[:28]:28s} {trial.dispatch_calls:5d} "
                f"{trial.candidate_calls:5d} {trial.runtime_fallbacks:3d} "
                f"{parity:>7s} {median:>9s} {speedup:>8s} {peak:>9s}"
            )
        return "\n".join(lines)


def artifact_ids_in(artifact_root: Path) -> tuple[str, ...]:
    """Artifact ids present under ``artifact_root``, in stable order."""
    ids: list[str] = []
    for manifest in sorted(Path(artifact_root).glob("*/artifact.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IsolationError(f"cannot read {manifest}: {exc}") from exc
        artifact_id = payload.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id:
            ids.append(artifact_id)
    return tuple(ids)


def artifact_kinds_in(artifact_root: Path) -> dict[str, str]:
    """Map artifact id -> declared ``operation.target_kind``.

    The kind decides which execution signal a trial must assert, so a bundle
    that does not declare one is read as ``module``: the most constrained kind,
    and the one whose signal is hardest to satisfy by accident.
    """
    kinds: dict[str, str] = {}
    for manifest in sorted(Path(artifact_root).glob("*/artifact.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IsolationError(f"cannot read {manifest}: {exc}") from exc
        artifact_id = payload.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            continue
        operation = payload.get("operation")
        kind = MODULE
        if isinstance(operation, Mapping):
            declared = operation.get("target_kind")
            if isinstance(declared, str) and declared:
                kind = declared
        kinds[artifact_id] = kind
    return kinds


def execution_evidence_for(
    payload: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect every execution signal a runtime reported, for any kind.

    Kernel kinds read ``candidate_calls`` from the dispatch counts; attention
    and schedule-transform kinds read fields the runtime writes alongside the
    dispatch decisions. Absent fields stay absent rather than defaulting to
    zero, because "the runtime never reported this" and "it reported zero" must
    not collapse into the same verdict.
    """
    evidence: dict[str, Any] = {"candidate_calls": counts.get("candidate_calls")}
    for source_key in ("loop_transform", "attention", "execution_evidence"):
        source = payload.get(source_key)
        if isinstance(source, Mapping):
            for key, value in source.items():
                evidence.setdefault(key, value)
    for key in (
        "attention_backend",
        "effective_backend",
        "steps_skipped",
        "hook_invocations",
    ):
        if key in payload and key not in evidence:
            evidence[key] = payload[key]
    return evidence


def dispatch_counts_for(
    payload: Mapping[str, Any],
    artifact_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate dispatch decisions, optionally restricted to some artifacts.

    ``calls`` counts every invocation the dispatcher saw for a selected
    signature and ``candidate_calls`` counts the subset the artifact actually
    executed; a gap between them is a runtime fallback, which is why both are
    reported rather than one.
    """
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise IsolationError("dispatch diagnostics contain no decisions list")
    wanted = set(artifact_ids) if artifact_ids is not None else None
    calls = candidate_calls = fallbacks = 0
    scopes: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        artifact_id = decision.get("artifact_id")
        if not isinstance(artifact_id, str):
            continue
        if wanted is not None and artifact_id not in wanted:
            continue
        calls += int(decision.get("calls") or 0)
        candidate_calls += int(decision.get("candidate_calls") or 0)
        fallbacks += int(decision.get("runtime_fallbacks") or 0)
        scope = decision.get("scope")
        if isinstance(scope, str):
            scopes.append(scope)
    return {
        "calls": calls,
        "candidate_calls": candidate_calls,
        "runtime_fallbacks": fallbacks,
        "scopes": tuple(sorted(set(scopes))),
    }


def _peak(result: Any) -> float | None:
    values = getattr(result, "peak_memory_mb", None)
    if not values:
        return None
    return max(float(value) for value in values)


def run_isolation_trials(
    *,
    fastvideo_checkout: Path,
    workload_path: Path,
    artifact_root: Path,
    output_dir: Path,
    model: str,
    native_result_path: Path,
    workload: Any,
    artifact_ids: Sequence[str] | None = None,
    include_combined: bool = True,
    base_env: Mapping[str, str] | None = None,
) -> IsolationReport:
    """Run one generation per artifact, then optionally one combined trial.

    A combined trial only runs over artifacts that were individually safe and
    individually worthwhile: combining a set that already contains a known
    parity offender cannot produce an interpretable result.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    native = load_generation_result(Path(native_result_path))
    report = IsolationReport(native_median_wall_seconds=native.median_wall_seconds)

    ids = tuple(artifact_ids) if artifact_ids is not None else artifact_ids_in(artifact_root)
    if not ids:
        raise IsolationError(f"no artifacts found under {artifact_root}")
    artifact_kinds = artifact_kinds_in(artifact_root)

    plan: list[tuple[str, tuple[str, ...]]] = [
        (artifact_id, (artifact_id,)) for artifact_id in ids
    ]

    for trial_name, trial_ids in plan:
        report.trials.append(
            _run_one_trial(
                trial_name=trial_name,
                trial_ids=trial_ids,
                fastvideo_checkout=Path(fastvideo_checkout),
                workload_path=Path(workload_path),
                artifact_root=Path(artifact_root),
                output_dir=output_dir,
                model=model,
                native=native,
                workload=workload,
                base_env=base_env,
                artifact_kinds=artifact_kinds,
            )
        )

    if include_combined:
        combined = report.safe_and_worthwhile
        if len(combined) > 1:
            report.trials.append(
                _run_one_trial(
                    trial_name="combined",
                    trial_ids=combined,
                    fastvideo_checkout=Path(fastvideo_checkout),
                    workload_path=Path(workload_path),
                    artifact_root=Path(artifact_root),
                    output_dir=output_dir,
                    model=model,
                    native=native,
                    workload=workload,
                    base_env=base_env,
                    artifact_kinds=artifact_kinds,
                )
            )

    (output_dir / "isolation.json").write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "isolation_table.txt").write_text(
        report.table() + "\n", encoding="utf-8"
    )
    return report


def _run_one_trial(
    *,
    trial_name: str,
    trial_ids: tuple[str, ...],
    fastvideo_checkout: Path,
    workload_path: Path,
    artifact_root: Path,
    output_dir: Path,
    model: str,
    native: Any,
    workload: Any,
    base_env: Mapping[str, str] | None,
    artifact_kinds: Mapping[str, str],
) -> TrialRecord:
    trial_dir = output_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = trial_dir / "dispatch.json"

    env = {
        **(workload.mode_env.for_mode("candidate") if getattr(workload, "mode_env", None) else {}),
        **(dict(base_env) if base_env else {}),
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR": str(artifact_root),
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID": str(model),
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_VALIDATION": "1",
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS": str(diagnostics_path),
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_ENABLE": ",".join(trial_ids),
    }

    try:
        run_mode(
            fastvideo_checkout=fastvideo_checkout,
            workload=workload_path,
            mode="candidate",
            output_dir=trial_dir,
            model_override=str(model),
            env=env,
        )
        candidate = load_generation_result(trial_dir / "candidate_result.json")
    except Exception as exc:  # noqa: BLE001 - one failed trial must not stop the pass
        return TrialRecord(
            trial=trial_name,
            artifact_ids=trial_ids,
            status="error",
            target_kinds=tuple(
                dict.fromkeys(
                    artifact_kinds.get(artifact_id, MODULE)
                    for artifact_id in trial_ids
                )
            ),
            native_median_wall_seconds=native.median_wall_seconds,
            error=f"{type(exc).__name__}: {exc}",
        )

    counts: dict[str, Any] = {"calls": 0, "candidate_calls": 0, "runtime_fallbacks": 0, "scopes": ()}
    diagnostics: dict[str, Any] = {}
    with suppress(Exception):
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        counts = dispatch_counts_for(diagnostics, trial_ids)
    # When the diagnostics could not be read at all, report no evidence rather
    # than zeroed counters: an unreadable runtime record is not proof of a
    # no-op, and must not be read as one.
    evidence = (
        execution_evidence_for(diagnostics, counts) if diagnostics else {}
    )
    trial_kinds = tuple(
        dict.fromkeys(artifact_kinds.get(artifact_id, MODULE) for artifact_id in trial_ids)
    )

    frame_atol, frame_rtol = (
        workload.parity or ParitySpec()
    ).frame_tolerances()
    parity = compare_frame_outputs(
        native.frames_path,
        candidate.frames_path,
        policy=workload.parity.policy if getattr(workload, "parity", None) else "byte_equal",
        atol=frame_atol,
        rtol=frame_rtol,
    )
    # The gate is derived from this workload's own Amdahl ceiling when it
    # declares a profiled share, so a candidate is asked for a share of what is
    # achievable rather than a flat number that may be unreachable.
    try:
        gate_basis = gate_basis_for_workload(workload)
    except (CeilingError, AttributeError, TypeError, ValueError):
        gate_basis = {"gate": 1.01, "basis": "default", "rule": "fallback 1.01"}
    performance = classify_end_to_end(
        native,
        candidate,
        min_speedup=float(gate_basis["gate"]),
        max_peak_memory_regression=(
            workload.performance.max_peak_memory_regression
            if getattr(workload, "performance", None) is not None
            else 0.05
        ),
    )

    return TrialRecord(
        trial=trial_name,
        artifact_ids=trial_ids,
        status="ok",
        dispatch_calls=int(counts["calls"]),
        candidate_calls=int(counts["candidate_calls"]),
        runtime_fallbacks=int(counts["runtime_fallbacks"]),
        scopes_selected=tuple(counts["scopes"]),
        target_kinds=trial_kinds,
        execution_evidence=evidence,
        gate_basis=gate_basis,
        parity_passed=bool(parity.get("passed")),
        parity_reason=str(parity.get("reason") or ""),
        median_wall_seconds=candidate.median_wall_seconds,
        native_median_wall_seconds=native.median_wall_seconds,
        end_to_end_speedup=performance.get("end_to_end_speedup"),
        peak_memory_mb=_peak(candidate),
        peak_memory_regression=performance.get("peak_memory_regression"),
    )
