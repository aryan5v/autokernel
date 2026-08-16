# Running an optimization campaign

One command per campaign. Preflight validates every precondition and pins an
immutable run contract before any stage runs, so a misconfigured campaign
fails in seconds rather than after an hour of GPU time.

## 1. Preflight first, always

```bash
motionkernel optimize \
  --fastvideo-checkout /path/to/FastVideo \
  --model FastVideo/LTX2-Distilled-Diffusers \
  --workload "$(motionkernel workload path ltx_480p)" \
  --output ./runs/ltx2 \
  --search-agent pi --search-model anthropic/claude-sonnet-4-5 \
  --preflight-only
```

Exit code 0 and `preflight_passed` means every precondition holds: the
FastVideo checkout has the launcher, the workload schema validates, the
output directory is writable with free space, every stage command resolves,
the search agent is on PATH, and every budget and threshold is a number.

Read `./runs/ltx2/preflight.json` before spending GPU time. It records the
MotionKernel and FastVideo commits, the workload's SHA-256, and the search
agent and model — the identity the resulting evidence will carry.

## 2. Choosing the search agent

| preset | confines its own writes | model |
|---|---|---|
| `codex` (default) | yes, `-s workspace-write` | whatever the CLI is configured with |
| `pi` | **no** | required: `--search-model provider/id` |

`pi` serves several providers, so the model is a campaign parameter rather
than a property of the installed CLI, and it is required rather than
guessed. Any other agent works through `--search-agent-command` with a JSON
argv array supporting `{repo_root}`, `{run_dir}`, `{candidate_dir}`,
`{prompt_file}`, `{last_message}`, `{prompt}` and `{model}`.

**On unsandboxed agents.** The search stage digests the fixed harness —
`bench.py`, `autokernel/verification`, `autokernel/specs` — before and after
the agent runs, and voids the stage if it changed. An agent that edits the
harness has changed what "passing" means, so nothing it measured is evidence
about a kernel. This check is what makes an agent without a permission
system usable at all; prefer running such an agent inside a container
anyway.

## 3. Running

```bash
motionkernel optimize \
  --fastvideo-checkout /path/to/FastVideo \
  --model FastVideo/LTX2-Distilled-Diffusers \
  --workload "$(motionkernel workload path ltx_480p)" \
  --baseline compile \
  --budget-hours 10 \
  --per-candidate-budget-seconds 3600 \
  --output ./runs/ltx2 \
  --search-agent pi --search-model anthropic/claude-sonnet-4-5
```

`--baseline compile` measures candidates against `torch.compile` rather than
eager. Eager is the flattering baseline and produces speedups that vanish in
production.

Stages run in order and each writes a durable record:

    baseline → profile → discover → specgen → search →
    isolated_validate → package → end_to_end_validate → finalize

`--stop-after-stage discover` runs discovery only, which is the cheap way to
ask whether a model has anything worth optimizing before committing a search
budget to it.

## 4. Resuming

```bash
motionkernel optimize ... --output ./runs/ltx2   # same command; resume is the default
```

A resume keeps every completed stage and reopens the first incomplete one.
This now applies to campaigns that ended in `failed` or `budget_exhausted`
too: a stage dying to an OOM or a node preemption no longer discards the
hours of durable work before it.

A resume fails closed when anything material changed — model, workload
content, FastVideo checkout, baseline, promotion threshold, stage commands,
search agent command, **or the MotionKernel harness commit**. Editing
`bench.py` or the verification code mid-campaign invalidates whatever ran
before it, so `contract_mismatch_harness` stops the resume rather than
letting one verdict mix results from two harnesses. `--budget-hours` is an
invocation allowance, not identity: granting more time is always allowed.

## 5. Reading the outcome

The command exits successfully only on `promoted`,
`no_worthwhile_candidate`, `discovery_complete`, or `preflight_passed`.

`no_worthwhile_candidate` is a real result, not a failure. It means the
pipeline looked and found nothing that survived its gates — which is the
honest answer for most regions of most models.

```bash
motionkernel artifact verify ./runs/ltx2/artifacts/<artifact-id>
motionkernel artifact inspect ./runs/ltx2/artifacts/<artifact-id>
```

Read these in the run directory:

| file | what it answers |
|---|---|
| `receipt.json` | terminal state and why |
| `morning_report.md` | the same, readable |
| `preflight.json` | what the campaign was, exactly |
| `run_contract.json` | what a resume is allowed to change |
| `stages/*/` | per-stage records, logs, and candidate directories |
| `isolation.json` / `isolation_table.txt` | per-artifact parity, dispatch, speedup |

## 6. What promotion requires

All of these, measured, or the artifact stays quarantined:

- isolated correctness over the weighted shape corpus, on regenerated inputs
  rather than replayed captures;
- **proof the artifact ran** — each kind declares its own signal (a kernel
  counts candidate calls, an attention backend echoes the implementation that
  executed, a schedule transform counts hook invocations). Absent evidence
  reads as "not proven", never as zero;
- output parity under the workload's declared fidelity tier;
- no unacceptable peak-memory regression;
- an end-to-end speedup clearing the gate. When the workload declares
  `performance.profiled_share`, that gate is derived from its Amdahl ceiling
  — `max(1.10, 1 + 0.5 × (ceiling − 1))` — so a candidate is asked for about
  half of the headroom that actually exists. A flat gate cannot say whether
  it is reachable; on `ltx-480p` attention is 15.13% of device time, so a
  flat 1.3× is unreachable by *any* backend, including one that takes zero
  time.

## 7. Known limits

- End-to-end results on this cluster are single-session. Sessions 1082 and
  1083 produced non-overlapping bootstrap intervals for the same artifact,
  so a single session's CI understates real uncertainty: treat the session,
  not the pair, as the unit of replication and repeat before believing a
  number.
- Clock locking is unavailable here. The paired protocol's sustained warmup
  reaches the same steady state and records the trace, but a measurement
  whose native arm exceeds the CV ceiling is marked invalid for gating
  rather than quietly used.
