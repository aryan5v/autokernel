"""Command-line entry point for an unattended optimization campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import OptimizeConfig, OptimizeError, run_optimize
from .types import PIPELINE_STAGES, default_repo_root


def _load_stage_commands(path: Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OptimizeError(f"cannot load stage commands {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise OptimizeError("stage commands file must contain a JSON object")
    commands: dict[str, list[str]] = {}
    for stage, command in raw.items():
        if not isinstance(stage, str) or not isinstance(command, list) or not all(
            isinstance(part, str) for part in command
        ):
            raise OptimizeError(
                "stage commands must map stage names to JSON arrays of strings"
            )
        commands[stage] = command
    return commands


def _load_argv(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OptimizeError(f"cannot load agent command {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw or not all(
        isinstance(part, str) and part for part in raw
    ):
        raise OptimizeError("agent command file must contain a non-empty JSON argv array")
    return raw


def parser() -> argparse.ArgumentParser:
    """Build the public optimize argument parser."""
    result = argparse.ArgumentParser(
        prog="motionkernel optimize",
        description=(
            "Run a resumable MotionKernel overnight optimize campaign "
            "(preflight → baseline → profile → discover → specgen → search → "
            "isolated_validate → package → end_to_end_validate → finalize). "
            "Preflight validates the environment and pins an immutable run "
            "contract before any stage runs; a resume fails closed when the "
            "model, workload content, checkout, or policy has changed."
        ),
    )
    result.add_argument(
        "--fastvideo-checkout",
        type=Path,
        required=True,
        help="Path to a FastVideo checkout (isolated subprocess only)",
    )
    result.add_argument("--model", required=True, help="FastVideo-resolvable model identifier")
    result.add_argument(
        "--workload", type=Path, required=True, help="Workload manifest (.yaml/.json)"
    )
    result.add_argument(
        "--budget-hours", type=float, default=10.0, help="Wall-clock campaign budget in hours"
    )
    result.add_argument(
        "--output", type=Path, required=True, help="Run directory for state, logs, receipts, and artifacts"
    )
    result.add_argument(
        "--baseline",
        choices=("eager", "compile"),
        default="eager",
        help="Honest PyTorch performance baseline for isolated benches",
    )
    result.add_argument(
        "--min-e2e-speedup",
        type=float,
        default=1.01,
        help="Minimum end-to-end speedup required for promotion (default 1.01)",
    )
    result.add_argument(
        "--per-candidate-budget-seconds",
        type=float,
        default=None,
        help="Maximum search/isolated-validation time for one candidate",
    )
    result.add_argument(
        "--stage-commands",
        type=Path,
        default=None,
        metavar="JSON",
        help=(
            "Optional stage-to-argv JSON manifest. Arguments may use "
            "{stage}, {run_dir}, {repo_root}, {fastvideo_checkout}, "
            "{workload}, {model}, {baseline}, and {artifact_dir}."
        ),
    )
    result.add_argument(
        "--search-agent-command",
        type=Path,
        default=None,
        metavar="JSON",
        help=(
            "Optional JSON argv array for the autonomous search agent. "
            "Supports {repo_root}, {run_dir}, {candidate_dir}, {prompt_file}, "
            "and {last_message}; defaults to the installed Codex CLI."
        ),
    )
    result.add_argument(
        "--search-agent",
        default="codex",
        help=(
            "Named search-agent preset: codex (sandboxes its own writes) or "
            "pi (multi-provider; requires --search-model). Ignored when "
            "--search-agent-command is given."
        ),
    )
    result.add_argument(
        "--search-model",
        default=None,
        help=(
            "Model for the search agent, e.g. anthropic/claude-sonnet-4-5. "
            "Required by presets that serve several providers."
        ),
    )
    result.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate every precondition, write preflight.json, and exit "
            "without running a stage or creating campaign state"
        ),
    )
    result.add_argument(
        "--no-resume", action="store_true", help="Ignore completed stages and start fresh"
    )
    result.add_argument(
        "--stop-after-stage",
        choices=PIPELINE_STAGES,
        default=None,
        help=(
            "Stop after this pipeline stage completes instead of running to "
            "finalize. The campaign terminates as discovery_complete and a "
            "later resume is idempotent. Used by discovery-only nightlies."
        ),
    )
    result.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="MotionKernel repo root (default: installed package or source checkout)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the optimize command and return a process exit status."""
    args = parser().parse_args(argv)
    repo_root = args.repo_root or default_repo_root()
    try:
        config = OptimizeConfig(
            fastvideo_checkout=args.fastvideo_checkout,
            model=args.model,
            workload=args.workload,
            output=args.output,
            budget_hours=args.budget_hours,
            resume=not args.no_resume,
            baseline=args.baseline,
            min_e2e_speedup=args.min_e2e_speedup,
            per_candidate_budget_seconds=args.per_candidate_budget_seconds,
            stage_commands=_load_stage_commands(args.stage_commands),
            search_agent_command=_load_argv(args.search_agent_command),
            search_agent=args.search_agent,
            search_model=args.search_model,
            repo_root=repo_root,
            stop_after_stage=args.stop_after_stage,
        )
        receipt = run_optimize(config, preflight_only=args.preflight_only)
    except OptimizeError as exc:
        print(f"OPTIMIZE: FAIL\n{exc}", file=sys.stderr)
        return 2

    print(json.dumps(receipt, indent=2))
    terminal = receipt.get("terminal") or receipt.get("status")
    if terminal in {
        "promoted",
        "no_worthwhile_candidate",
        "preflight_passed",
        "discovery_complete",
    }:
        return 0
    return 1
