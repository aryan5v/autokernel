"""Fail-closed preflight validation and immutable run contracts.

An unattended overnight campaign must fail in the first seconds rather than
after hours of GPU time.  This module validates every precondition the
pipeline depends on *before* any expensive stage runs or any campaign state
is mutated, and pins the material configuration into a write-once
``run_contract.json`` so a later resume cannot silently mix results from a
different model, workload, checkout, or promotion policy.

Two artifacts are produced in the run directory:

``preflight.json``
    Rewritten on every invocation.  Records pass/fail, structured reason
    codes, environment identity, and the execution policy.

``run_contract.json``
    Written once when a campaign begins and never rewritten.  Resuming
    compares the live configuration against it and fails closed on any
    material difference.

Nothing here is model-specific, and no secret material is persisted: command
configurations are recorded as SHA-256 digests plus a program basename, never
as raw argv.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state import OptimizeError, read_json, utc_now, write_json_atomic
from .types import PIPELINE_STAGES, OptimizeConfig, default_repo_root

PREFLIGHT_SCHEMA_VERSION = 1
RUN_CONTRACT_SCHEMA_VERSION = 1

#: Placeholders a stage command may reference (mirrors ``stages.py``).
STAGE_COMMAND_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "{stage}",
        "{run_dir}",
        "{repo_root}",
        "{fastvideo_checkout}",
        "{workload}",
        "{model}",
        "{baseline}",
        "{artifact_dir}",
    }
)

#: A ``{lower_snake_case}`` token, the only shape a placeholder may take.
_PLACEHOLDER_PATTERN = re.compile(r"\{[a-z][a-z0-9_]*\}")

#: Relative paths a usable FastVideo checkout must provide.
FASTVIDEO_REQUIRED_PATHS: tuple[str, ...] = (
    "fastvideo",
    "examples/inference/optimizations/generation_launcher.py",
)

#: Repository files the built-in stage adapters invoke as subprocesses.
REPO_REQUIRED_FILES: tuple[str, ...] = ("bench.py",)

_GIT_TIMEOUT_SECONDS = 10.0
MIN_OUTPUT_FREE_BYTES = 512 * 1024**2
WARN_OUTPUT_FREE_BYTES = 10 * 1024**3


class PreflightError(OptimizeError):
    """Preflight refused to start or resume a campaign."""


@dataclass(frozen=True)
class PreflightFinding:
    """One structured preflight outcome."""

    code: str
    message: str
    severity: str = "error"  # error | warning
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": dict(self.detail),
        }


@dataclass
class PreflightReport:
    """Structured preflight result, serialized to ``preflight.json``."""

    findings: list[PreflightFinding] = field(default_factory=list)
    sections: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None

    def add(self, finding: PreflightFinding) -> None:
        self.findings.append(finding)

    def error(self, code: str, message: str, **detail: Any) -> None:
        self.add(PreflightFinding(code, message, "error", dict(detail)))

    def warn(self, code: str, message: str, **detail: Any) -> None:
        self.add(PreflightFinding(code, message, "warning", dict(detail)))

    @property
    def errors(self) -> list[PreflightFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[PreflightFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def reason_codes(self) -> list[str]:
        return [f.code for f in self.errors]

    def failure_message(self) -> str:
        if self.passed:
            return "preflight passed"
        lines = [f"{f.code}: {f.message}" for f in self.errors]
        return "preflight failed:\n  " + "\n  ".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "pass" if self.passed else "fail",
            "passed": self.passed,
            "started_at": self.started_at,
            "finished_at": self.finished_at or utc_now(),
            "reason_codes": self.reason_codes,
            "findings": [f.as_dict() for f in self.findings],
            **self.sections,
        }


# --------------------------------------------------------------------------
# Redaction-safe identity helpers
# --------------------------------------------------------------------------


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def command_identity(argv: Sequence[str] | None) -> dict[str, Any] | None:
    """Redaction-safe identity for an argv list.

    Arguments may carry tokens, endpoints, or prompt text, so the raw list is
    never persisted.  A digest pins the exact command for contract
    comparison, while the program basename and argument count keep the record
    diagnosable.
    """
    if argv is None:
        return None
    parts = [str(part) for part in argv]
    program = Path(parts[0]).name if parts else ""
    return {
        "program": program,
        "argc": len(parts),
        "sha256": _sha256_text("\x00".join(parts)),
    }


def _stage_commands_identity(
    stage_commands: Mapping[str, Sequence[str]] | None,
) -> dict[str, Any] | None:
    if not stage_commands:
        return None
    per_stage = {
        stage: command_identity(command)
        for stage, command in sorted(stage_commands.items())
    }
    combined = "\x00".join(
        f"{stage}={(identity or {}).get('sha256', '')}"
        for stage, identity in sorted(per_stage.items())
    )
    return {
        "stages": sorted(per_stage),
        "per_stage": per_stage,
        "sha256": _sha256_text(combined),
    }


def _git(path: Path, *args: str) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_identity(path: Path) -> dict[str, Any]:
    """Best-effort git identity for a checkout.

    Absolute paths are recorded only as a digest; the commit is the stable
    identity an operator and the run contract both rely on.
    """
    resolved = path.resolve()
    identity: dict[str, Any] = {
        "name": resolved.name,
        "path_digest": _sha256_text(str(resolved)),
        "commit": None,
        "branch": None,
        "dirty": None,
    }
    if not resolved.is_dir():
        return identity
    if _git(resolved, "rev-parse", "--is-inside-work-tree") != "true":
        return identity
    identity["commit"] = _git(resolved, "rev-parse", "HEAD")
    identity["branch"] = _git(resolved, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(resolved, "status", "--porcelain")
    if status is not None:
        identity["dirty"] = bool(status.strip())
    return identity


def _checkout_matches(stored: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Compare checkouts by commit when both are known, else by path.

    A checkout moved to a new absolute path but pinned to the same commit is
    the same checkout for campaign purposes; a changed commit never is.
    """
    stored_commit = stored.get("commit")
    current_commit = current.get("commit")
    if stored_commit and current_commit:
        return bool(stored_commit == current_commit)
    return bool(stored.get("path_digest") == current.get("path_digest"))


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def _check_policy(config: OptimizeConfig, report: PreflightReport) -> None:
    """Validate numeric and enumerated execution policy."""
    if config.baseline not in {"eager", "compile"}:
        report.error(
            "baseline_invalid",
            "baseline must be 'eager' or 'compile'",
            value=config.baseline,
        )
    if not config.model.strip():
        report.error("model_empty", "model must not be empty")
    if not math.isfinite(config.budget_hours) or config.budget_hours <= 0:
        report.error(
            "budget_invalid",
            "budget_hours must be finite and positive",
            value=repr(config.budget_hours),
        )
    if not math.isfinite(config.min_e2e_speedup) or config.min_e2e_speedup <= 0:
        report.error(
            "promotion_threshold_invalid",
            "min_e2e_speedup must be finite and positive",
            value=repr(config.min_e2e_speedup),
        )
    per_candidate = config.per_candidate_budget_seconds
    if per_candidate is not None and (
        not math.isfinite(per_candidate) or per_candidate <= 0
    ):
        report.error(
            "candidate_timeout_invalid",
            "per_candidate_budget_seconds must be finite and positive",
            value=repr(per_candidate),
        )
    if (
        not config.artifact_dir_name
        or config.artifact_dir_name in {".", ".."}
        or Path(config.artifact_dir_name).name != config.artifact_dir_name
    ):
        report.error(
            "artifact_dir_invalid",
            "artifact_dir_name must be one directory name",
            value=config.artifact_dir_name,
        )


def _check_fastvideo(config: OptimizeConfig, report: PreflightReport) -> dict[str, Any]:
    """Validate the FastVideo checkout exists with the expected structure."""
    checkout = config.fastvideo_checkout
    identity = git_identity(checkout)
    if not checkout.exists():
        report.error(
            "fastvideo_checkout_missing",
            f"FastVideo checkout not found: {checkout}",
        )
        return identity
    if not checkout.is_dir():
        report.error(
            "fastvideo_checkout_not_a_directory",
            f"FastVideo checkout is not a directory: {checkout}",
        )
        return identity
    missing = [
        relative
        for relative in FASTVIDEO_REQUIRED_PATHS
        if not (checkout / relative).exists()
    ]
    if missing:
        report.error(
            "fastvideo_structure_invalid",
            (
                "FastVideo checkout is missing required paths: "
                + ", ".join(missing)
                + "; install/update the FastVideo branch that provides "
                "examples/inference/optimizations/generation_launcher.py"
            ),
            missing=missing,
        )
    if identity["commit"] is None:
        report.warn(
            "fastvideo_commit_unknown",
            "FastVideo checkout has no resolvable git commit; resume will "
            "compare the checkout path instead of a commit",
        )
    elif identity["dirty"]:
        report.warn(
            "fastvideo_worktree_dirty",
            "FastVideo checkout has uncommitted changes; the recorded commit "
            "does not fully describe the code that will run",
        )
    return identity


def _check_workload(
    config: OptimizeConfig, report: PreflightReport
) -> dict[str, Any]:
    """Validate the workload exists and parses under the shared schema."""
    workload = config.workload
    record: dict[str, Any] = {
        "name": workload.name,
        "sha256": None,
        "workload_id": None,
        "model_id": None,
        "parity_policy": None,
    }
    if not workload.is_file():
        report.error("workload_missing", f"workload not found: {workload}")
        return record
    record["sha256"] = _sha256_file(workload)
    if record["sha256"] is None:
        report.error(
            "workload_unreadable", f"workload could not be read: {workload}"
        )
        return record
    try:
        from autokernel.workload import load_workload

        manifest = load_workload(workload)
    except Exception as exc:  # noqa: BLE001 - any parse failure must fail closed
        report.error(
            "workload_invalid",
            f"workload failed schema validation: {exc}",
        )
        return record
    record["workload_id"] = manifest.workload_id
    record["model_id"] = manifest.model.model_id
    record["parity_policy"] = (
        manifest.parity.policy if manifest.parity is not None else None
    )
    return record


def _check_output(
    config: OptimizeConfig, report: PreflightReport
) -> dict[str, int | None]:
    """Validate the output directory can be created and atomically written."""
    output = config.output
    storage: dict[str, int | None] = {
        "total_bytes": None,
        "used_bytes": None,
        "free_bytes": None,
        "minimum_free_bytes": MIN_OUTPUT_FREE_BYTES,
        "warning_free_bytes": WARN_OUTPUT_FREE_BYTES,
    }
    if output.exists() and not output.is_dir():
        report.error(
            "output_not_a_directory",
            f"output path exists and is not a directory: {output}",
        )
        return storage
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report.error(
            "output_not_creatable",
            f"output directory could not be created: {output}: {exc}",
        )
        return storage
    # Atomic writes replace a temporary file in the same directory, so probe
    # exactly that capability rather than merely testing os.access.
    probe: Path | None = None
    target = output / ".preflight.probe"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output,
            prefix=".preflight.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write("preflight")
            probe = Path(handle.name)
        probe.replace(target)
    except OSError as exc:
        report.error(
            "output_not_writable",
            f"output directory is not atomically writable: {output}: {exc}",
        )
    finally:
        for temporary in (probe, target):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    try:
        usage = shutil.disk_usage(output)
    except OSError as exc:
        report.warn(
            "output_free_space_unknown",
            "could not determine free space for the output directory",
            error_type=type(exc).__name__,
        )
        return storage

    storage.update(
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
    )
    if usage.free < MIN_OUTPUT_FREE_BYTES:
        report.error(
            "output_free_space_insufficient",
            "output filesystem has less than 512 MiB free",
            free_bytes=usage.free,
            required_bytes=MIN_OUTPUT_FREE_BYTES,
        )
    elif usage.free < WARN_OUTPUT_FREE_BYTES:
        report.warn(
            "output_free_space_low",
            "output filesystem has less than 10 GiB free; an overnight "
            "campaign may exhaust it",
            free_bytes=usage.free,
            warning_bytes=WARN_OUTPUT_FREE_BYTES,
        )
    return storage


def _check_stage_commands(
    config: OptimizeConfig, report: PreflightReport
) -> None:
    """Validate stage names, argv shape, placeholders, and programs."""
    if not config.stage_commands:
        return
    unknown = sorted(set(config.stage_commands) - set(PIPELINE_STAGES))
    if unknown:
        report.error(
            "stage_command_unknown_stage",
            f"unknown stage command(s): {', '.join(unknown)}",
            stages=unknown,
        )
    for stage, command in sorted(config.stage_commands.items()):
        parts = [str(part) for part in command]
        if not parts or any(not part for part in parts):
            report.error(
                "stage_command_empty",
                f"stage command for {stage!r} must not be empty",
                stage=stage,
            )
            continue
        for part in parts:
            for token in _placeholders(part):
                if token not in STAGE_COMMAND_PLACEHOLDERS:
                    report.error(
                        "stage_command_placeholder_invalid",
                        (
                            f"stage command for {stage!r} uses unknown "
                            f"placeholder {token}"
                        ),
                        stage=stage,
                        placeholder=token,
                    )
        if _resolve_program(parts[0]) is None:
            report.error(
                "stage_command_executable_missing",
                (
                    f"stage command for {stage!r} program is not executable: "
                    f"{Path(parts[0]).name}"
                ),
                stage=stage,
                program=Path(parts[0]).name,
            )


def _placeholders(value: str) -> list[str]:
    """Return placeholder-shaped ``{name}`` tokens in a stage-command argument.

    ``stages.py`` expands known placeholders by literal substitution and leaves
    everything else untouched, so a stage command may legitimately carry JSON
    or Python literals containing braces.  Only tokens shaped like a
    placeholder are candidates for a typo, which keeps the check useful
    without rejecting arbitrary payloads.
    """
    return _PLACEHOLDER_PATTERN.findall(value)


def _resolve_program(program: str) -> str | None:
    """Resolve an argv[0] to an executable path, or ``None``."""
    candidate = Path(program)
    if candidate.is_absolute() or os.sep in program:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None
    return shutil.which(program)


def _check_search_agent(
    config: OptimizeConfig, report: PreflightReport
) -> dict[str, Any]:
    """Validate the resolved search-agent executable exists.

    The built-in adapter shells out to the Codex CLI when no explicit command
    is configured.  The check is skipped when a stage command replaces the
    search stage outright, or in simulation, and the skip is recorded rather
    than hidden.
    """
    record: dict[str, Any] = {
        "checked": True,
        "skipped_reason": None,
        "configured": config.search_agent_command is not None,
        "identity": command_identity(config.search_agent_command),
        "resolved": None,
    }
    simulated = os.environ.get("MOTIONKERNEL_SIMULATE") == "1"
    overridden = bool(config.stage_commands and "search" in config.stage_commands)
    if simulated or overridden:
        record["checked"] = False
        record["skipped_reason"] = (
            "simulation" if simulated else "search stage command override"
        )
        return record

    # A campaign that stops before the search stage never invokes the agent,
    # so requiring it would block discovery-only nightlies on a dependency
    # they do not have. The skip is recorded, not hidden.
    stop = config.stop_after_stage
    if (
        stop is not None
        and stop in PIPELINE_STAGES
        and PIPELINE_STAGES.index(stop) < PIPELINE_STAGES.index("search")
    ):
        record["checked"] = False
        record["skipped_reason"] = f"campaign stops after {stop}; search never runs"
        return record

    if config.search_agent_command is not None:
        parts = [str(part) for part in config.search_agent_command]
        if not parts or any(not part for part in parts):
            report.error(
                "search_agent_command_invalid",
                "search_agent_command must not contain empty arguments",
            )
            return record
        resolved = _resolve_program(parts[0])
        if resolved is None:
            report.error(
                "search_agent_missing",
                (
                    "configured search agent is not executable: "
                    f"{Path(parts[0]).name}"
                ),
                program=Path(parts[0]).name,
            )
            return record
        record["resolved"] = Path(resolved).name
        return record

    from .search import AGENT_PRESETS

    agent = getattr(config, "search_agent", "codex") or "codex"
    preset = AGENT_PRESETS.get(agent)
    if preset is None:
        report.error(
            "search_agent_unknown",
            (
                f"unknown search agent {agent!r}; known presets: "
                f"{', '.join(sorted(AGENT_PRESETS))}"
            ),
            program=agent,
        )
        return record
    record["agent"] = agent
    record["sandboxed"] = bool(preset.get("sandboxed"))

    model = getattr(config, "search_model", None) or preset.get("default_model")
    if "{model}" in "".join(preset["argv"]) and not model:
        report.error(
            "search_model_missing",
            (
                f"search agent {agent!r} serves several providers and has no "
                f"honest default; pass --search-model"
            ),
            program=agent,
        )
        return record
    if model:
        record["model"] = str(model)

    program = str(preset["program"])
    resolved = shutil.which(program)
    if resolved is None:
        report.error(
            "search_agent_missing",
            (
                f"autonomous search requires the {program!r} CLI on PATH or an "
                f"explicit --search-agent-command"
            ),
            program=program,
        )
        return record
    if not preset.get("sandboxed"):
        # Not fatal, but it changes what the search stage is trusting: this
        # agent does not confine its own writes, so the harness digest check
        # in the search stage is the only thing keeping the fixed harness
        # fixed. Recorded so a reader of preflight.json knows which regime the
        # campaign ran under.
        report.warn(
            "search_agent_unsandboxed",
            (
                f"{program!r} does not confine its own writes; the search "
                f"stage will verify the harness is untouched instead. Prefer "
                f"running it inside a container."
            ),
            program=program,
        )
    record["resolved"] = Path(resolved).name
    return record


def _check_executables(
    config: OptimizeConfig, report: PreflightReport
) -> dict[str, Any]:
    """Validate the interpreter and repository entry points the stages use."""
    repo_root = config.repo_root or default_repo_root()
    record: dict[str, Any] = {
        "python": Path(sys.executable).name if sys.executable else None,
        "python_version": ".".join(str(p) for p in sys.version_info[:3]),
        "git": shutil.which("git") is not None,
        "repo_files": {},
    }
    if not sys.executable or not Path(sys.executable).is_file():
        report.error(
            "python_executable_missing",
            "sys.executable does not resolve to a usable interpreter",
        )
    if not repo_root.is_dir():
        report.error(
            "repo_root_missing",
            f"MotionKernel repo root not found: {repo_root}",
        )
        return record
    for relative in REPO_REQUIRED_FILES:
        present = (repo_root / relative).is_file()
        record["repo_files"][relative] = present
        if not present:
            report.error(
                "repo_entry_point_missing",
                f"MotionKernel repo root is missing {relative}",
                path=relative,
            )
    if not record["git"]:
        report.warn(
            "git_unavailable",
            "git is not on PATH; commit identities will be unavailable for "
            "this campaign",
        )
    return record


# --------------------------------------------------------------------------
# Run contract
# --------------------------------------------------------------------------


def build_run_contract(sections: Mapping[str, Any]) -> dict[str, Any]:
    """Build the immutable contract payload from validated preflight sections."""
    # Duration and resume mode are invocation controls, not evidence identity.
    # Operators may grant an interrupted campaign more time without changing
    # the model, code, workload, search policy, or promotion threshold.
    policy = {
        key: value
        for key, value in sections["policy"].items()
        if key not in {"budget_hours", "resume"}
    }
    return {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "model": sections["model"],
        "workload": {
            "workload_id": sections["workload"].get("workload_id"),
            "sha256": sections["workload"].get("sha256"),
            "name": sections["workload"].get("name"),
        },
        "fastvideo": {
            "commit": sections["fastvideo"].get("commit"),
            "path_digest": sections["fastvideo"].get("path_digest"),
            "name": sections["fastvideo"].get("name"),
        },
        "motionkernel": {
            "commit": sections["motionkernel"].get("commit"),
            "path_digest": sections["motionkernel"].get("path_digest"),
        },
        "policy": policy,
        "commands": {
            "stage_commands": sections["commands"].get("stage_commands"),
            "search_agent": sections["commands"].get("search_agent"),
        },
    }


#: Stable reason code per material contract field.
CONTRACT_MISMATCH_CODES: dict[str, str] = {
    "model": "contract_mismatch_model",
    "workload_content": "contract_mismatch_workload_content",
    "workload_identity": "contract_mismatch_workload_identity",
    "fastvideo": "contract_mismatch_fastvideo_checkout",
    "baseline": "contract_mismatch_baseline",
    "min_e2e_speedup": "contract_mismatch_promotion_threshold",
    "per_candidate_budget_seconds": "contract_mismatch_candidate_timeout",
    "artifact_dir_name": "contract_mismatch_artifact_dir",
    "stage_commands": "contract_mismatch_stage_commands",
    "search_agent": "contract_mismatch_search_agent_command",
    "motionkernel": "contract_mismatch_harness",
}


def compare_run_contract(
    stored: Mapping[str, Any], current: Mapping[str, Any]
) -> list[PreflightFinding]:
    """Return a finding per material difference between two contracts."""
    findings: list[PreflightFinding] = []

    def mismatch(key: str, was: Any, now: Any) -> None:
        findings.append(
            PreflightFinding(
                CONTRACT_MISMATCH_CODES[key],
                f"{key} changed since the campaign began",
                "error",
                {"field": key, "contract": was, "requested": now},
            )
        )

    if stored.get("model") != current.get("model"):
        mismatch("model", stored.get("model"), current.get("model"))

    stored_workload = stored.get("workload") or {}
    current_workload = current.get("workload") or {}
    if stored_workload.get("sha256") != current_workload.get("sha256"):
        mismatch(
            "workload_content",
            stored_workload.get("sha256"),
            current_workload.get("sha256"),
        )
    elif stored_workload.get("workload_id") != current_workload.get("workload_id"):
        mismatch(
            "workload_identity",
            stored_workload.get("workload_id"),
            current_workload.get("workload_id"),
        )

    if not _checkout_matches(
        stored.get("fastvideo") or {}, current.get("fastvideo") or {}
    ):
        stored_fv = stored.get("fastvideo") or {}
        current_fv = current.get("fastvideo") or {}
        mismatch(
            "fastvideo",
            stored_fv.get("commit") or stored_fv.get("path_digest"),
            current_fv.get("commit") or current_fv.get("path_digest"),
        )

    stored_policy = stored.get("policy") or {}
    current_policy = current.get("policy") or {}
    for key in (
        "baseline",
        "min_e2e_speedup",
        "per_candidate_budget_seconds",
        "artifact_dir_name",
    ):
        if stored_policy.get(key) != current_policy.get(key):
            mismatch(key, stored_policy.get(key), current_policy.get(key))

    # The harness is evidence identity, not an invocation control. Editing
    # bench, verification or measurement code between runs of one campaign
    # invalidates whatever ran before it: a resumed campaign would mix results
    # produced by two different harnesses under one verdict. This was recorded
    # in the contract from the start but never compared, so six gap-7 attempts
    # each re-ran against a harness the previous attempt had not used.
    if not _checkout_matches(
        stored.get("motionkernel") or {}, current.get("motionkernel") or {}
    ):
        stored_mk = stored.get("motionkernel") or {}
        current_mk = current.get("motionkernel") or {}
        mismatch(
            "motionkernel",
            stored_mk.get("commit") or stored_mk.get("path_digest"),
            current_mk.get("commit") or current_mk.get("path_digest"),
        )

    stored_commands = stored.get("commands") or {}
    current_commands = current.get("commands") or {}
    stored_stage = (stored_commands.get("stage_commands") or {}).get("sha256")
    current_stage = (current_commands.get("stage_commands") or {}).get("sha256")
    if stored_stage != current_stage:
        mismatch("stage_commands", stored_stage, current_stage)
    stored_agent = (stored_commands.get("search_agent") or {}).get("sha256")
    current_agent = (current_commands.get("search_agent") or {}).get("sha256")
    if stored_agent != current_agent:
        mismatch("search_agent", stored_agent, current_agent)

    return findings


def load_run_contract(path: Path) -> dict[str, Any]:
    """Load and version-check a stored run contract."""
    try:
        stored = read_json(path)
    except (OSError, ValueError) as exc:
        raise PreflightError(f"run contract corrupt: {path}: {exc}") from exc
    if not isinstance(stored, dict):
        raise PreflightError(f"run contract must be a JSON object: {path}")
    if stored.get("schema_version") != RUN_CONTRACT_SCHEMA_VERSION:
        raise PreflightError(
            f"unsupported run contract schema_version "
            f"{stored.get('schema_version')!r}: {path}"
        )
    return stored


def write_run_contract(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Write the contract once; an existing contract is never rewritten."""
    if path.is_file():
        return load_run_contract(path)
    write_json_atomic(path, dict(contract))
    return dict(contract)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def execute_preflight(
    config: OptimizeConfig,
    *,
    contract_path: Path,
    resuming: bool,
) -> tuple[PreflightReport, dict[str, Any]]:
    """Run every precondition check and compare against the run contract.

    Returns the report and the contract describing *this* invocation.  The
    caller persists the report, aborts when it failed, and only then writes
    the contract for a fresh campaign.
    """
    report = PreflightReport()

    _check_policy(config, report)
    fastvideo = _check_fastvideo(config, report)
    workload = _check_workload(config, report)
    output_storage = _check_output(config, report)
    _check_stage_commands(config, report)
    search_agent = _check_search_agent(config, report)
    executables = _check_executables(config, report)

    repo_root = config.repo_root or default_repo_root()
    report.sections = {
        "model": config.model,
        "workload": workload,
        "fastvideo": fastvideo,
        "motionkernel": git_identity(repo_root),
        "policy": {
            "baseline": config.baseline,
            "min_e2e_speedup": config.min_e2e_speedup,
            "budget_hours": config.budget_hours,
            "per_candidate_budget_seconds": config.per_candidate_budget_seconds,
            "artifact_dir_name": config.artifact_dir_name,
            "resume": config.resume,
        },
        "commands": {
            "stage_commands": _stage_commands_identity(config.stage_commands),
            "search_agent": search_agent["identity"],
        },
        "search_agent": {
            key: value for key, value in search_agent.items() if key != "identity"
        },
        "environment": executables,
        "output_storage": output_storage,
        "simulated": os.environ.get("MOTIONKERNEL_SIMULATE") == "1",
        "resuming": resuming,
    }

    contract = build_run_contract(report.sections)

    if resuming:
        if not contract_path.is_file():
            report.error(
                "contract_missing",
                (
                    "cannot resume: run_contract.json is missing from an "
                    "existing campaign; start a new --output or use "
                    "--no-resume"
                ),
            )
        else:
            try:
                stored = load_run_contract(contract_path)
            except PreflightError as exc:
                report.error("contract_unreadable", str(exc))
            else:
                for finding in compare_run_contract(stored, contract):
                    report.add(finding)

    report.finished_at = utc_now()
    return report, contract
