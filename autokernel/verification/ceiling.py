"""Ceiling-derived promotion gates: ask for a share of what is achievable.

A flat gate asks every candidate for the same speedup regardless of how much
speedup the workload can physically give. That is wrong in both directions.

On `ltx-480p`, attention is 15.13% of device time, so the Amdahl ceiling is
1.178x: a flat 1.3x gate is unreachable there by *any* backend, including one
that takes zero time. Candidates get rejected for failing an impossible bar,
and the rejection says nothing about the candidate.

On a workload where attention is 60% of device time the ceiling is 2.5x, and a
flat 1.10x gate would promote something capturing 7% of what was available --
which is a worse outcome dressed as a pass.

So the gate is derived from the ceiling:

    gate = max(1.10, 1 + 0.5 * (ceiling - 1))

Two properties, both deliberate. The floor at 1.10 means a promotion is always
worth the fidelity risk and the maintenance cost of carrying an artifact, even
where the ceiling is low. And the 0.5 coefficient means a candidate must
capture **at least half** of the headroom that exists -- a share of the
achievable, not an absolute that ignores it.

Worked examples:

===============  =======  =====  =====================================
workload         ceiling  gate   note
===============  =======  =====  =====================================
ltx-480p          1.178   1.10   floor binds; half of 17.8% is 8.9%
wan-480p (low)    1.373   1.19   ~half the headroom
wan-480p (high)   1.537   1.27   ~half the headroom
hypothetical      2.500   1.75   a high ceiling demands a high gate
===============  =======  =====  =====================================

The rule never lowers a *quality* gate. Fidelity budgets are independent and
both must pass; this decides only what counts as enough speed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CEILING_GATE_RULE",
    "GATE_COEFFICIENT",
    "GATE_FLOOR",
    "CeilingError",
    "GateDecision",
    "amdahl_ceiling",
    "derive_gate",
    "evaluate_gate",
    "gate_basis_for_workload",
    "gate_from_workload",
]

#: Never promote for less than this, however low the ceiling.
GATE_FLOOR = 1.10
#: Fraction of the available headroom a candidate must capture.
GATE_COEFFICIENT = 0.5
#: Recorded in every verdict so a stored decision stays interpretable when the
#: rule changes.
CEILING_GATE_RULE = "max(1.10, 1 + 0.5 * (ceiling - 1))"


class CeilingError(ValueError):
    """A ceiling or gate cannot be derived from the inputs given."""


def amdahl_ceiling(share: float) -> float:
    """Maximum end-to-end speedup when a component of ``share`` becomes free.

    ``1 / (1 - share)``. A share of 1.0 has no finite ceiling and is refused
    rather than returned as infinity, because an infinite ceiling would derive
    an infinite gate and nothing would ever promote.
    """
    if isinstance(share, bool) or not isinstance(share, (int, float)):
        raise CeilingError(f"share must be a number, got {share!r}")
    value = float(share)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise CeilingError(
            f"share must be in [0, 1); got {share!r}. A share of 1.0 would mean "
            f"the workload is entirely this component"
        )
    return 1.0 / (1.0 - value)


def derive_gate(
    ceiling: float,
    *,
    floor: float = GATE_FLOOR,
    coefficient: float = GATE_COEFFICIENT,
) -> float:
    """The speedup a candidate must reach on a workload with this ceiling."""
    if isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)):
        raise CeilingError(f"ceiling must be a number, got {ceiling!r}")
    value = float(ceiling)
    if not math.isfinite(value) or value < 1.0:
        raise CeilingError(
            f"ceiling must be finite and at least 1.0, got {ceiling!r}"
        )
    return max(floor, 1.0 + coefficient * (value - 1.0))


@dataclass(frozen=True)
class GateDecision:
    """Whether a measured speedup clears its derived gate, and by how much."""

    measured: float
    ceiling: float
    gate: float
    margin: float
    passed: bool
    rule: str = CEILING_GATE_RULE
    ceiling_source: str = ""
    reachable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "measured_speedup": round(self.measured, 6),
            "measured_ceiling": round(self.ceiling, 6),
            "gate": round(self.gate, 6),
            "margin": round(self.margin, 6),
            "passed": self.passed,
            "rule": self.rule,
            "ceiling_source": self.ceiling_source,
            "ceiling_reachable": self.reachable,
        }


def evaluate_gate(
    measured: float,
    *,
    ceiling: float,
    ceiling_source: str = "",
    floor: float = GATE_FLOOR,
    coefficient: float = GATE_COEFFICIENT,
) -> GateDecision:
    """Decide a candidate against its workload's ceiling-derived gate.

    ``ceiling_source`` records where the ceiling came from (which profile,
    which share) so a stored verdict can be re-derived later. A decision whose
    ceiling cannot be traced is not auditable, which is the same standard the
    experiment store applies to rows.

    ``reachable`` is False when the derived gate exceeds the ceiling itself --
    which the floor can cause on a very low-ceiling workload. That is not a
    contradiction to hide: it means *no* candidate can pass here, and the
    honest response is to skip the workload rather than to run candidates that
    are guaranteed to fail.
    """
    if isinstance(measured, bool) or not isinstance(measured, (int, float)):
        raise CeilingError(f"measured speedup must be a number, got {measured!r}")
    value = float(measured)
    if not math.isfinite(value) or value <= 0:
        raise CeilingError(
            f"measured speedup must be finite and positive, got {measured!r}"
        )
    gate = derive_gate(ceiling, floor=floor, coefficient=coefficient)
    return GateDecision(
        measured=value,
        ceiling=float(ceiling),
        gate=gate,
        margin=value - gate,
        passed=value >= gate,
        ceiling_source=ceiling_source,
        reachable=gate <= float(ceiling),
    )


def gate_from_workload(workload: Any, *, share: float | None = None) -> float:
    """Derive the gate for ``workload``, preferring a declared ceiling.

    A workload carrying a flat ``performance.min_end_to_end_speedup`` and no
    share still validates -- it returns the flat value -- but callers are
    expected to emit a deprecation, because a flat gate cannot say whether it
    is unreachable.
    """
    performance = getattr(workload, "performance", None)
    if share is None:
        share = getattr(performance, "profiled_share", None)
    if share is not None:
        return derive_gate(amdahl_ceiling(float(share)))
    declared = getattr(performance, "min_end_to_end_speedup", None)
    if declared is None:
        raise CeilingError(
            "workload declares neither a profiled share nor a flat gate"
        )
    return float(declared)


def gate_basis_for_workload(
    workload: Any, *, share: float | None = None
) -> dict[str, Any]:
    """The gate and how it was derived, for the record a verdict carries.

    A stored decision has to stay interpretable after the rule changes, so the
    basis travels with the number: the share it came from, the ceiling that
    share implies, the rule, and whether the gate is flat -- which is the case
    a reader should distrust, because a flat gate cannot say whether it was
    reachable.
    """
    performance = getattr(workload, "performance", None)
    if share is None:
        share = getattr(performance, "profiled_share", None)
    if share is not None:
        ceiling = amdahl_ceiling(float(share))
        return {
            "gate": derive_gate(ceiling),
            "basis": "ceiling",
            "profiled_share": float(share),
            "ceiling": ceiling,
            "rule": CEILING_GATE_RULE,
        }
    declared = getattr(performance, "min_end_to_end_speedup", None)
    if declared is None:
        raise CeilingError(
            "workload declares neither a profiled share nor a flat gate"
        )
    return {
        "gate": float(declared),
        "basis": "flat",
        "profiled_share": None,
        "ceiling": None,
        "rule": "declared min_end_to_end_speedup",
        "deprecation": (
            "flat gate: declare performance.profiled_share so the gate can be "
            "derived from the workload's Amdahl ceiling"
        ),
    }
