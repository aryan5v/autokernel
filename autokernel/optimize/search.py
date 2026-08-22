"""Built-in autonomous search and independent isolated validation.

The search agent may edit only a generated candidate's ``kernel.py``. The
validator then runs the fixed MotionKernel harness in a separate process and
derives the artifact request exclusively from measured JSON and the validated
graph manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autokernel.discovery.ranking import (
    measured_e2e_improvement,
    measured_e2e_improvement_from_latency,
    projected_end_to_end_speedup,
)
from autokernel.specgen import (
    build_dispatch_contract,
    spec_from_manifest,
    write_runtime_adapter,
)
from autokernel.verification.policy import ParityPolicy
from autokernel.workload import load_workload

from .types import default_repo_root


class BuiltinSearchError(RuntimeError):
    """Search or isolated validation could not satisfy its contract."""


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else deadline - time.monotonic()


def _generated(candidate: Mapping[str, Any]) -> dict[str, Path]:
    raw = candidate.get("generated")
    if not isinstance(raw, Mapping):
        raise BuiltinSearchError("specified candidate has no generated artifacts")
    required = ("manifest", "spec", "kernel", "corpus")
    paths: dict[str, Path] = {}
    for name in required:
        value = raw.get(name)
        if not isinstance(value, str) or not value:
            raise BuiltinSearchError(f"generated candidate has no {name!r} path")
        path = Path(value).resolve()
        if not path.is_file():
            raise BuiltinSearchError(f"generated candidate file is missing: {path}")
        paths[name] = path
    return paths


def _parity_settings(config: Mapping[str, Any]) -> tuple[str, float | None]:
    """Resolve the workload's output contract for the kernel-level gates.

    The workload file is authoritative: it is what declares
    ``parity.policy``. ``config`` may override the absolute-error ceiling,
    which has no workload-level home yet. A config naming no workload gets the
    strictest policy, because a gate that cannot find its contract must not
    invent a permissive one.
    """
    policy = "byte_equal"
    workload_path = config.get("workload")
    if workload_path:
        try:
            workload = load_workload(Path(str(workload_path)))
        except Exception:  # noqa: BLE001 - an unreadable workload stays strict
            workload = None
        if workload is not None:
            policy = ParityPolicy.from_workload(workload).policy
    override = config.get("parity_policy")
    if override:
        policy = str(override)
    ceiling = config.get("max_absolute_error")
    return policy, (float(ceiling) if ceiling is not None else None)


def _realized_impact(
    candidate: Mapping[str, Any],
    primary: Mapping[str, Any],
    share: float,
    speedup: float,
) -> tuple[float, str]:
    """End-to-end gain this measured kernel returns, and how it was derived.

    A ``derived_subregion`` candidate replaces part of a larger timed region,
    so its ``share_of_e2e`` -- the parent's -- must not be multiplied by a
    speedup measured on the subregion alone. When discovery supplied a real
    invocation count and a model total, the absolute per-call saving answers
    the question directly. Otherwise fall back to the share-based form and say
    so, because for a whole-region candidate the two agree.
    """
    # ``calls`` is the invocation count observed across ``profiled_generations``
    # generations, while ``total_cuda_time_us`` is already a per-generation
    # total. Only the call count is scaled; dividing both would silently drop
    # the ratio by a factor of ``profiled_generations``.
    calls = candidate.get("calls")
    total_us = candidate.get("total_cuda_time_us")
    generations = float(candidate.get("profiled_generations") or 1.0) or 1.0
    if (
        candidate.get("selection_mode") == "derived_subregion"
        and isinstance(calls, (int, float))
        and not isinstance(calls, bool)
        and calls > 0
        and isinstance(total_us, (int, float))
        and float(total_us) > 0
    ):
        return (
            measured_e2e_improvement_from_latency(
                baseline_us=float(primary["pytorch_latency_us"]),
                candidate_us=float(primary["kernel_latency_us"]),
                calls_per_generation=float(calls) / generations,
                total_generation_us=float(total_us),
            ),
            "measured_latency_x_invocations",
        )
    return measured_e2e_improvement(share, speedup), "region_share_x_speedup"


def _benchmark_command(
    repo_root: Path,
    generated: Mapping[str, Path],
    result_path: Path,
    *,
    baseline: str,
    quick: bool,
    parity_policy: str = "byte_equal",
    max_absolute_error: float | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "bench.py"),
        "--spec",
        f"{generated['spec']}:SPEC",
        "--shape-corpus",
        str(generated["corpus"]),
        "--shape-corpus-only",
        "--baseline",
        baseline,
        # The workload's output contract governs every gate, not just the final
        # frame comparison. Without this the quick search benchmark reports a
        # candidate as correctness-passing on a 1e-2 tolerance that the
        # workload never agreed to.
        "--parity-policy",
        parity_policy,
        "--dispatch-stress",
        "--result-json",
        str(result_path),
    ]
    if max_absolute_error is not None:
        command.extend(["--max-absolute-error", repr(float(max_absolute_error))])
    if quick:
        command.append("--quick")
    return command


def _run_benchmark(
    repo_root: Path,
    generated: Mapping[str, Path],
    result_path: Path,
    log_path: Path,
    *,
    baseline: str,
    quick: bool,
    parity_policy: str = "byte_equal",
    max_absolute_error: float | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = _benchmark_command(
        repo_root,
        generated,
        result_path,
        baseline=baseline,
        quick=quick,
        parity_policy=parity_policy,
        max_absolute_error=max_absolute_error,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=generated["kernel"].parent,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(timeout, 0.001) if timeout is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_path.write_text(stdout + stderr, encoding="utf-8")
        raise BuiltinSearchError(
            "per-candidate budget exhausted during benchmark"
        ) from exc
    log_path.write_text(
        (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise BuiltinSearchError(
            f"benchmark failed with exit {completed.returncode}; see {log_path}"
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuiltinSearchError(f"invalid benchmark result {result_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise BuiltinSearchError("benchmark result must use schema_version 2")
    return payload


def _primary(payload: Mapping[str, Any]) -> dict[str, Any]:
    performance = payload.get("performance")
    primary = performance.get("primary") if isinstance(performance, Mapping) else None
    if not isinstance(primary, dict):
        raise BuiltinSearchError("benchmark produced no primary measurement")
    for name in ("pytorch_latency_us", "kernel_latency_us", "speedup_vs_pytorch"):
        value = primary.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise BuiltinSearchError(f"benchmark primary {name} is missing")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise BuiltinSearchError(f"benchmark primary {name} must be positive")
    return primary


def _dispatch_stress(payload: Mapping[str, Any]) -> dict[str, Any]:
    performance = payload.get("performance")
    stress = (
        performance.get("dispatch_stress")
        if isinstance(performance, Mapping)
        else None
    )
    if not isinstance(stress, dict):
        raise BuiltinSearchError("benchmark produced no dispatch-stress evidence")
    if stress.get("status") != "PASS" or stress.get("correctness") != "PASS":
        raise BuiltinSearchError(
            "candidate failed repeated-dispatch validation: "
            f"{stress.get('reason') or 'unknown reason'}"
        )
    for name in ("fresh_input_speedup", "steady_input_speedup"):
        value = stress.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise BuiltinSearchError(f"dispatch stress {name} is missing")
        number = float(value)
        if not math.isfinite(number) or number <= 1.0:
            raise BuiltinSearchError(
                f"dispatch stress {name} must be greater than 1.0"
            )
    return stress


def _agent_prompt(
    candidate: Mapping[str, Any],
    generated: Mapping[str, Path],
    benchmark: Sequence[str],
) -> str:
    # Discovery emits ``estimated_max_e2e_improvement`` as a fraction. The
    # prompt used to read ``estimated_max_e2e_improvement_pct``, a key nothing
    # ever wrote, so every r4 search agent was told its target was worth
    # "None%". Label the number for what it is: an upper bound that assumes the
    # region's cost drops to nearly zero, not a measurement.
    share = float(candidate.get("share_of_e2e") or 0.0)
    upper_bound = float(candidate.get("estimated_max_e2e_improvement") or 0.0)
    return f"""\
Optimize one graph-derived CUDA kernel autonomously. Do not ask questions.

Target fingerprint: {candidate.get('fingerprint')}
Region share of end-to-end CUDA time: {share * 100:.3f}%
Upper-bound model impact if this region became free: {upper_bound * 100:.3f}%
(That bound is not achievable. A kernel measuring Nx faster returns
share * (1 - 1/N) of end-to-end; at 1.10x on this region that is
{share * (1 - 1 / 1.10) * 100:.3f}%.)
Editable file: {generated['kernel']}
Read-only semantic inputs: {generated['manifest']}, {generated['spec']}, {generated['corpus']}

Run this exact benchmark after every meaningful experiment:
{' '.join(benchmark)}

You may change only kernel.py in the candidate directory. Never edit the
manifest, spec, corpus, benchmark harness, references, tolerances, or verifier.
Keep only changes that pass correctness and improve speedup_vs_pytorch. Explore
Triton, torch.compile, and fused PyTorch implementations as appropriate for the
captured operations and observed shapes. Do not create or capture CUDA graphs,
and do not create or switch CUDA streams: FastVideo owns the execution context
and may apply framework-level graph replay after validation. The benchmark's
dispatch-stress gate uses several fresh tensor identities and repeated calls;
both its fresh and steady speedups must exceed 1.0x. Leave the fastest passing
candidate in kernel.py. If no implementation beats the reference, restore the
best passing version and say so in the final message.
"""


#: Agent presets. The search interface is an argv list, so any CLI agent works
#: through --search-agent-command; these are the ones we keep working.
#:
#: ``sandboxed`` records whether the agent confines its own writes. Codex does
#: (``-s workspace-write``). Pi ships no permission system and inherits the
#: user's, so a pi search is only as bounded as the process around it -- which
#: is why the search stage verifies afterwards that the harness tree is
#: untouched rather than trusting the agent to have stayed inside its
#: candidate directory.
AGENT_PRESETS: dict[str, dict[str, Any]] = {
    "codex": {
        "program": "codex",
        "sandboxed": True,
        "argv": [
            "exec",
            "-C",
            "{candidate_dir}",
            "-s",
            "workspace-write",
            "--skip-git-repo-check",
            "--output-last-message",
            "{last_message}",
            "{prompt}",
        ],
    },
    # pi (github.com/earendil-works/pi): multi-provider, so the model is a
    # campaign parameter rather than a property of the installed CLI.
    "pi": {
        "program": "pi",
        "sandboxed": False,
        "argv": [
            "--print",
            "--no-session",
            "--model",
            "{model}",
            "{prompt}",
        ],
    },
}

DEFAULT_AGENT = "codex"


def agent_preset(name: str) -> dict[str, Any]:
    preset = AGENT_PRESETS.get(name)
    if preset is None:
        known = ", ".join(sorted(AGENT_PRESETS))
        raise BuiltinSearchError(
            f"unknown search agent {name!r}; known presets: {known}. Any other "
            f"CLI agent can be used through --search-agent-command."
        )
    return preset


def _agent_command(
    configured: Sequence[str] | None,
    *,
    repo_root: Path,
    run_dir: Path,
    candidate_dir: Path,
    prompt_path: Path,
    last_message: Path,
    agent: str = DEFAULT_AGENT,
    model: str | None = None,
) -> list[str]:
    replacements = {
        "{repo_root}": str(repo_root),
        "{run_dir}": str(run_dir),
        "{candidate_dir}": str(candidate_dir),
        "{prompt_file}": str(prompt_path),
        "{last_message}": str(last_message),
        "{prompt}": prompt_path.read_text(encoding="utf-8"),
        "{model}": str(model or ""),
    }

    def expand(parts: Sequence[str]) -> list[str]:
        command = []
        for raw in parts:
            value = str(raw)
            for placeholder, replacement in replacements.items():
                value = value.replace(placeholder, replacement)
            command.append(value)
        return command

    if configured:
        return expand(configured)

    preset = agent_preset(agent)
    if model is None and preset.get("default_model"):
        replacements["{model}"] = str(preset["default_model"])
    if "{model}" in "".join(preset["argv"]) and not replacements["{model}"]:
        raise BuiltinSearchError(
            f"search agent {agent!r} needs a model; pass --search-model "
            f"(it serves several providers, so there is no single default that "
            f"is honest about cost or capability)"
        )
    executable = shutil.which(preset["program"])
    if executable is None:
        raise BuiltinSearchError(
            f"autonomous search requires the {preset['program']!r} CLI on PATH "
            f"or an explicit --search-agent-command"
        )
    return [executable, *expand(preset["argv"])]


def _harness_digest(repo_root: Path) -> str:
    """A digest of the fixed harness the search must not touch.

    Only the files whose contents decide a verdict: the benchmark, the
    verification package, and the specs. A search that edits any of them has
    changed what "passing" means, and its result is not evidence about a
    kernel.
    """
    digest = hashlib.sha256()
    targets: list[Path] = [repo_root / "bench.py"]
    for directory in ("autokernel/verification", "autokernel/specs"):
        targets.extend(sorted((repo_root / directory).rglob("*.py")))
    for path in targets:
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(str(path.relative_to(repo_root)).encode())
    return digest.hexdigest()


def search_candidates(
    run_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the configured coding agent, then measure every resulting kernel."""
    repo_root = Path(str(config.get("repo_root") or default_repo_root())).resolve()
    baseline = str(config.get("baseline") or "eager")
    parity_policy, max_absolute_error = _parity_settings(config)
    configured = config.get("search_agent_command")
    if configured is not None and not isinstance(configured, list):
        raise BuiltinSearchError("search_agent_command must be an argv list")
    agent = str(config.get("search_agent") or DEFAULT_AGENT)
    search_model = config.get("search_model")
    search_model = str(search_model) if search_model else None
    # An agent that does not confine its own writes gets checked instead of
    # trusted: the harness, specs and corpora sit outside the candidate
    # directory and must be byte-identical after the agent exits.
    sandboxed = bool(configured) or bool(agent_preset(agent).get("sandboxed"))
    harness_before = None if sandboxed else _harness_digest(repo_root)
    budget_value = config.get("per_candidate_budget_seconds")
    per_candidate_budget = (
        float(budget_value) if budget_value is not None else None
    )

    stage_dir = run_dir / "stages" / "search"
    searched: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    measured_count = 0
    for candidate in candidates:
        deadline = (
            time.monotonic() + per_candidate_budget
            if per_candidate_budget is not None
            else None
        )

        generated = _generated(candidate)
        fingerprint = str(candidate.get("fingerprint") or "unknown")
        work = stage_dir / fingerprint
        work.mkdir(parents=True, exist_ok=True)
        quick_result = work / "search_benchmark.json"
        benchmark = _benchmark_command(
            repo_root,
            generated,
            quick_result,
            baseline=baseline,
            quick=True,
            parity_policy=parity_policy,
            max_absolute_error=max_absolute_error,
        )
        prompt_path = work / "prompt.md"
        prompt_path.write_text(
            _agent_prompt(candidate, generated, benchmark), encoding="utf-8"
        )
        last_message = work / "agent_last_message.md"
        log_path = work / "agent.log"
        try:
            command = _agent_command(
                configured,
                repo_root=repo_root,
                run_dir=run_dir,
                candidate_dir=generated["kernel"].parent,
                prompt_path=prompt_path,
                last_message=last_message,
                agent=agent,
                model=search_model,
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=(
                        max(_remaining(deadline), 0.001)
                        if _remaining(deadline) is not None
                        else None
                    ),
                )
            except subprocess.TimeoutExpired as exc:
                stdout = (
                    exc.stdout.decode(errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else (exc.stdout or "")
                )
                stderr = (
                    exc.stderr.decode(errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else (exc.stderr or "")
                )
                log_path.write_text(stdout + stderr, encoding="utf-8")
                raise BuiltinSearchError(
                    "per-candidate budget exhausted during agent search"
                ) from exc
            log_path.write_text(
                (completed.stdout or "") + (completed.stderr or ""),
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise BuiltinSearchError(
                    f"search agent exited {completed.returncode}; see {log_path}"
                )
            payload = _run_benchmark(
                repo_root,
                generated,
                quick_result,
                work / "benchmark.log",
                baseline=baseline,
                parity_policy=parity_policy,
                max_absolute_error=max_absolute_error,
                quick=True,
                timeout=_remaining(deadline),
            )
            measured_count += 1
            primary = _primary(payload)
            stress = _dispatch_stress(payload)
            forward = payload.get("forward")
            correct = (
                isinstance(forward, Mapping)
                and forward.get("correctness") == "PASS"
            )
            speedup = float(primary["speedup_vs_pytorch"])
            if not correct or speedup <= 1.0:
                failures.append(
                    {
                        "fingerprint": fingerprint,
                        "reason": "candidate did not beat the isolated reference",
                    }
                )
                continue
            searched.append(
                {
                    **dict(candidate),
                    "search": {
                        "kernel": str(generated["kernel"]),
                        "result": str(quick_result),
                        "speedup": speedup,
                        "fresh_dispatch_speedup": float(
                            stress["fresh_input_speedup"]
                        ),
                        "steady_dispatch_speedup": float(
                            stress["steady_input_speedup"]
                        ),
                        "agent_log": str(log_path),
                        "agent_summary": str(last_message),
                    },
                }
            )
        except BuiltinSearchError as exc:
            failures.append({"fingerprint": fingerprint, "reason": str(exc)})

    if harness_before is not None and _harness_digest(repo_root) != harness_before:
        # Not a warning. Every measurement this stage produced was taken
        # against a harness the agent had modified, so none of them says
        # anything about a kernel.
        raise BuiltinSearchError(
            "the search agent modified the fixed harness (bench.py, "
            "autokernel/verification or autokernel/specs). Every result from "
            "this stage is void. Run an agent that confines its own writes, or "
            "run this one inside a container."
        )
    if not searched and not measured_count and failures:
        summary = "; ".join(item["reason"] for item in failures[:3])
        raise BuiltinSearchError(f"autonomous search produced no measurement: {summary}")
    if not searched:
        return {
            "candidates": [],
            "failures": failures,
            "recommendation": "no_worthwhile_candidate",
        }
    return {
        "candidates": searched,
        "failures": failures,
        "search_agent": {"agent": agent, "model": search_model, "sandboxed": sandboxed},
    }


def _gpu_architecture(payload: Mapping[str, Any]) -> str:
    gpu = payload.get("gpu")
    capability = gpu.get("compute_capability") if isinstance(gpu, Mapping) else None
    if (
        not isinstance(capability, list | tuple)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
    ):
        raise BuiltinSearchError("benchmark result has no GPU compute capability")
    major, minor = capability
    if major <= 0 or minor < 0:
        raise BuiltinSearchError("isolated validation requires a CUDA GPU")
    return f"sm{major}{minor}"


def _error_maxima(forward: Mapping[str, Any]) -> tuple[float, float]:
    absolute: list[float] = []
    relative: list[float] = []
    details = forward.get("leaf_details")
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, Mapping):
                continue
            for name, output in (
                ("max_abs_error", absolute),
                ("max_rel_error", relative),
            ):
                value = item.get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    number = float(value)
                    if math.isfinite(number) and number >= 0:
                        output.append(number)
    return max(absolute, default=0.0), max(relative, default=0.0)


def validate_candidates(
    run_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate searched kernels and build package requests."""
    repo_root = Path(str(config.get("repo_root") or default_repo_root())).resolve()
    baseline = str(config.get("baseline") or "eager")
    parity_policy, max_absolute_error = _parity_settings(config)
    workload = load_workload(Path(str(config["workload"])))
    # The campaign's end-to-end gate, applied per candidate at package time
    # rather than only after a full A/B generation pair has been spent.
    min_end_to_end_speedup = float(
        config.get("min_end_to_end_speedup")
        or getattr(workload.performance, "min_end_to_end_speedup", 1.0)
        or 1.0
    )
    dispatch_overhead = float(config.get("dispatch_overhead_fraction") or 0.0)
    budget_value = config.get("per_candidate_budget_seconds")
    per_candidate_budget = (
        float(budget_value) if budget_value is not None else None
    )
    stage_dir = run_dir / "stages" / "isolated_validate"
    validated: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    measured_count = 0

    for candidate in candidates:
        generated = _generated(candidate)
        fingerprint = str(candidate.get("fingerprint") or "unknown")
        work = stage_dir / fingerprint
        work.mkdir(parents=True, exist_ok=True)
        result_path = work / "benchmark.json"
        try:
            payload = _run_benchmark(
                repo_root,
                generated,
                result_path,
                work / "benchmark.log",
                baseline=baseline,
                parity_policy=parity_policy,
                max_absolute_error=max_absolute_error,
                quick=False,
                timeout=per_candidate_budget,
            )
            measured_count += 1
            forward = payload.get("forward")
            if not isinstance(forward, Mapping) or forward.get("correctness") != "PASS":
                raise BuiltinSearchError("candidate failed full isolated correctness")
            primary = _primary(payload)
            stress = _dispatch_stress(payload)
            speedup = float(primary["speedup_vs_pytorch"])
            if speedup <= 1.0:
                raise BuiltinSearchError("candidate did not beat the isolated reference")

            # An isolated micro-speedup is not an end-to-end improvement. Now
            # that the kernel has been measured, replace discovery's optimistic
            # upper bound with what this speedup actually returns on this
            # region's share of the model, and require that it clear the
            # campaign's gate with dispatch overhead charged against it.
            share = float(candidate.get("share_of_e2e") or 0.0)
            realized, impact_basis = _realized_impact(candidate, primary, share, speedup)
            projected = projected_end_to_end_speedup(
                [realized], dispatch_overhead_fraction=dispatch_overhead
            )
            if projected < min_end_to_end_speedup:
                optimistic = float(
                    candidate.get("estimated_max_e2e_improvement") or 0.0
                )
                raise BuiltinSearchError(
                    f"candidate cannot reach the campaign's end-to-end target: "
                    f"region is {share * 100:.3f}% of end-to-end and measured "
                    f"{speedup:.4f}x, returning {realized * 100:.3f}% "
                    f"(basis: {impact_basis}) "
                    f"(projected {projected:.5f}x < required "
                    f"{min_end_to_end_speedup:.5f}x). Discovery's upper bound "
                    f"of {optimistic * 100:.3f}% assumed the region's cost "
                    f"dropped to nearly zero"
                )
            architecture = _gpu_architecture(payload)
            manifest_value = json.loads(
                generated["manifest"].read_text(encoding="utf-8")
            )
            contract = build_dispatch_contract(manifest_value)
            spec = spec_from_manifest(generated["manifest"])
            tolerance = spec.tolerance_for(spec.dtypes[0])
            max_abs, max_rel = _error_maxima(forward)

            kernel_digest = hashlib.sha256(generated["kernel"].read_bytes()).hexdigest()
            artifact_id = (
                f"mk-{fingerprint[:16]}-{kernel_digest[:8]}-{architecture}"
            )
            payload_dir = work / "payload"
            payload_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generated["kernel"], payload_dir / "candidate.py")
            shutil.copyfile(generated["manifest"], payload_dir / "manifest.json")
            write_runtime_adapter(
                manifest_value,
                payload_dir / "entry.py",
                candidate_file="candidate.py",
            )

            now = datetime.now(timezone.utc).isoformat()
            sections = {
                "artifact_id": artifact_id,
                **contract,
                "entry_point": {"file": "entry.py", "symbol": "fused_subgraph"},
                "compatibility": {
                    "model_id": str(config["model"]),
                    "model_revision": getattr(workload.model, "revision", None) or "*",
                    "gpu_architectures": [architecture],
                    "torch": {},
                    "cuda": {},
                    "triton": {},
                    "execution_modes": ["inference"],
                    "distributed_modes": ["single"],
                },
                "evidence": {
                    "benchmark": {
                        "harness": "motionkernel-bench",
                        "device": "cuda:0",
                        "samples": 100,
                        "baseline_us": float(primary["pytorch_latency_us"]),
                        "candidate_us": float(primary["kernel_latency_us"]),
                        "speedup": speedup,
                        # Artifact schema 1 is intentionally strict and its
                        # benchmark object records isolated harness evidence
                        # only. Whole-model impact -- region share, realized
                        # improvement, its basis, the projection and the gate --
                        # stays authoritative in the search receipt below, which
                        # has fields for it. Writing it here instead made every
                        # bundle fail validation on an unknown field, so the
                        # search produced candidates that could never package.
                        "max_abs_error": max_abs,
                        "max_rel_error": max_rel,
                        "atol": tolerance.atol,
                        "rtol": tolerance.rtol,
                        "passed": True,
                        "result_ref": str(result_path),
                    },
                    "generation": {
                        "workload_id": workload.workload_id,
                        "steps": workload.sampling.num_inference_steps,
                        "metric": "pending_full_generation_validation",
                        "value": 0.0,
                        "threshold": 0.0,
                        "passed": False,
                        "baseline_ref": str(
                            run_dir / "generation" / "native_result.json"
                        ),
                        "candidate_ref": str(
                            run_dir / "generation" / "candidate_result.json"
                        ),
                    },
                },
                "promotion": {
                    "decision": "quarantined",
                    "reason": "awaiting full-generation A/B validation",
                    "decided_at": now,
                    "campaign": {
                        "campaign_id": run_dir.name or "motionkernel-optimize",
                        "source": "motionkernel-optimize",
                        "target_name": str(manifest_value.get("name") or fingerprint),
                    },
                },
            }
            validated.append(
                {
                    **dict(candidate),
                    "validation": {
                        "result": str(result_path),
                        "speedup": speedup,
                        "fresh_dispatch_speedup": float(
                            stress["fresh_input_speedup"]
                        ),
                        "steady_dispatch_speedup": float(
                            stress["steady_input_speedup"]
                        ),
                        "artifact_id": artifact_id,
                        # Whole-model impact lives here rather than in the
                        # bundle's benchmark object, whose schema admits
                        # isolated harness evidence only. A reader should never
                        # have to reconstruct this from a share and a bound.
                        "region_share_of_e2e": share,
                        "measured_e2e_improvement": realized,
                        "impact_basis": impact_basis,
                        "projected_end_to_end_speedup": projected,
                        "min_end_to_end_speedup": min_end_to_end_speedup,
                        "dispatch_overhead_fraction": dispatch_overhead,
                        "parity_policy": parity_policy,
                    },
                }
            )
            requests.append({"source_dir": str(payload_dir), "sections": sections})
        except (BuiltinSearchError, OSError, json.JSONDecodeError) as exc:
            failures.append({"fingerprint": fingerprint, "reason": str(exc)})

    if not validated and not measured_count and failures:
        summary = "; ".join(item["reason"] for item in failures[:3])
        raise BuiltinSearchError(
            f"isolated validation produced no measurement: {summary}"
        )
    if not validated:
        return {
            "candidates": [],
            "package_requests": [],
            "failures": failures,
            "recommendation": "no_worthwhile_candidate",
        }
    return {
        "candidates": validated,
        "package_requests": requests,
        "failures": failures,
    }
