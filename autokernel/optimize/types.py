"""Types and constants for the top-level MotionKernel optimize control plane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_repo_root() -> Path:
    """Resolve the runtime harness root in a checkout or installed wheel."""
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "bench.py").is_file():
        return source_root
    packaged_runtime = Path(__file__).resolve().parents[1] / "_runtime"
    if (packaged_runtime / "bench.py").is_file():
        return packaged_runtime
    # Let preflight produce its structured missing-entrypoint error.
    return source_root

# Ordered campaign stages (linear control plane).
PIPELINE_STAGES: tuple[str, ...] = (
    "baseline",
    "profile",
    "discover",
    "specgen",
    "search",
    "isolated_validate",
    "package",
    "end_to_end_validate",
    "finalize",
)

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "promoted",
        "no_worthwhile_candidate",
        "failed",
        "budget_exhausted",
        # The campaign was deliberately stopped after an early stage
        # (stop_after_stage), e.g. a discovery-only nightly. Not a verdict
        # on any candidate; the receipt records how far the run got.
        "discovery_complete",
    }
)

STAGE_RESULT_SCHEMA_VERSION = 1
CAMPAIGN_STATE_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1

#: Candidate status after each stage. These record *how far a candidate got*,
#: not whether it succeeded: the authoritative verdict is the artifact's
#: promotion decision, which finalize writes into each bundle.
#:
#: Run r4 ended with all four candidates reading ``status: finalized`` while
#: every artifact decision was ``quarantined`` and nothing was promoted. Read as
#: outcomes -- which is how they read -- that says the opposite of what
#: happened. The names now say "reached this stage" so the two cannot be
#: confused.
CANDIDATE_STAGE_STATUS: dict[str, str] = {
    "discover": "discovered",
    "specgen": "specified",
    "search": "searched",
    "isolated_validate": "isolated_validate_reached",
    "package": "packaged",
    "end_to_end_validate": "end_to_end_validate_reached",
    "finalize": "finalize_reached",
}

# Default minimum end-to-end speedup required for promotion (1%).
DEFAULT_MIN_E2E_SPEEDUP = 1.01


@dataclass(frozen=True)
class OptimizeConfig:
    """User-facing optimize invocation."""

    fastvideo_checkout: Path
    model: str
    workload: Path
    output: Path
    budget_hours: float = 10.0
    resume: bool = True
    baseline: str = "eager"  # eager | compile
    min_e2e_speedup: float = DEFAULT_MIN_E2E_SPEEDUP
    # Optional absolute overrides for tests / fake FastVideo + bench.
    stage_commands: Mapping[str, Sequence[str]] | None = None
    # Opaque artifact directory written by the packager stage (Task 2 contract).
    artifact_dir_name: str = "artifacts"
    per_candidate_budget_seconds: float | None = None
    # Optional argv for the autonomous search agent. Placeholders are
    # expanded without a shell; when omitted, the built-in adapter uses the
    # installed Codex CLI.
    search_agent_command: Sequence[str] | None = None
    # Named agent preset (see search.AGENT_PRESETS) used when no explicit
    # command is given, and the model it should run. The model is a campaign
    # parameter because a multi-provider agent has no single honest default.
    search_agent: str = "codex"
    search_model: str | None = None
    repo_root: Path | None = None
    # Stop after this pipeline stage completes instead of running to
    # finalize. Must be one of PIPELINE_STAGES. An operational control, not
    # run identity: resuming with a later (or no) stop stage continues the
    # same campaign.
    stop_after_stage: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fastvideo_checkout": str(self.fastvideo_checkout),
            "model": self.model,
            "workload": str(self.workload),
            "output": str(self.output),
            "budget_hours": self.budget_hours,
            "resume": self.resume,
            "baseline": self.baseline,
            "min_e2e_speedup": self.min_e2e_speedup,
            "stage_commands": (
                {k: list(v) for k, v in self.stage_commands.items()}
                if self.stage_commands
                else None
            ),
            "artifact_dir_name": self.artifact_dir_name,
            "per_candidate_budget_seconds": self.per_candidate_budget_seconds,
            "search_agent_command": (
                list(self.search_agent_command)
                if self.search_agent_command
                else None
            ),
            "search_agent": self.search_agent,
            "search_model": self.search_model,
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "stop_after_stage": self.stop_after_stage,
        }


@dataclass
class StageRecord:
    """Durable record for one completed or failed stage."""

    name: str
    status: str  # ok | failed | skipped
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    result_path: str | None = None
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "result_path": self.result_path,
            "message": self.message,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StageRecord:
        return cls(
            name=str(raw["name"]),
            status=str(raw["status"]),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            exit_code=raw.get("exit_code"),
            result_path=raw.get("result_path"),
            message=str(raw.get("message") or ""),
            metrics=dict(raw.get("metrics") or {}),
        )
