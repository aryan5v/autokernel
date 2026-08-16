"""What an artifact *is*, as an extensible contract rather than a literal set.

Until now every artifact replaced a region of the graph, and two kinds covered
it: a ``module`` target swaps a whole module's forward, a ``subgraph`` target
rewrites selected nodes inside one. Both were validated against a hardcoded
``{"module", "subgraph"}``, with the subgraph rewrite fields special-cased
inline.

That stops working as soon as artifacts stop being regions. An *attention
implementation* is not a captured subgraph -- it is a choice of backend for a
call that survives capture as a single opaque op. A *schedule transform* is not
a region at all -- it wraps the denoising loop and decides whether a step runs.
Neither has the fields a subgraph target needs, and both need fields no region
target has.

So kinds are registered rather than enumerated. Each declares which extra
fields it permits and which it requires, and validation is derived from that
declaration. Adding a kind is a registration, not an edit to a conditional that
every other kind also reads.

Two properties are deliberate:

* **Unknown fields are rejected per kind, not globally.** A ``module`` target
  carrying ``selected_node_ids`` is a malformed bundle, not a harmless extra --
  it means something upstream believed it was writing a subgraph target.
* **The default kind is the most constrained one.** A bundle that does not say
  what it is gets ``module``, which permits no extra fields at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ATTENTION",
    "MODULE",
    "SCHEDULE_TRANSFORM",
    "SUBGRAPH",
    "TargetKind",
    "backend_echo_evidence",
    "counter_evidence",
    "evaluate_execution_signal",
    "execution_signal_for",
    "known_target_kinds",
    "register_target_kind",
    "target_kind_spec",
]

MODULE = "module"
SUBGRAPH = "subgraph"
ATTENTION = "attention"
SCHEDULE_TRANSFORM = "schedule_transform"


@dataclass(frozen=True)
class TargetKind:
    """The shape of one artifact kind's operation identity.

    Every kind must declare an ``execution_signal``: the observable proof that
    its artifact did something. This is a registration requirement rather than
    a convention because of a specific incident. A paired A/B of the promoted
    V1 LTX artifact (SLURM 1078) returned 0.9011x with a tight confidence
    interval, a significant p-value, ``conclusive: true`` and a clean clock
    trace -- an apparent retraction of the project's headline. It measured
    nothing: the checkout read only ``FASTVIDEO_OPTIMIZATION_CAPTURE``, so
    every artifact variable was inert and both arms ran native.

    Every check the measurement had passed. None asked whether the
    intervention happened. The same shape had already bitten attention
    campaigns, where FastVideo silently substitutes FlashAttention and the
    fallback gets recorded under the requested backend's name.

    So a kind that cannot say how you would *know* it ran is not registrable.

    Args:
        name: the value that appears as ``operation.target_kind``.
        required: fields a bundle of this kind must declare.
        optional: fields it may declare.
        replaces_region: whether this kind substitutes a captured region of the
            graph. False for kinds that wrap execution instead (a schedule
            transform) or that select an implementation for an op that survives
            capture whole (an attention backend). Region-replacing kinds are
            matched by ``graph_fingerprint``; the others need their own
            compatibility check and must not be dispatched by fingerprint
            alone.
        description: one line, surfaced in validation errors.
    """

    name: str
    required: frozenset[str] = field(default_factory=frozenset)
    optional: frozenset[str] = field(default_factory=frozenset)
    replaces_region: bool = True
    description: str = ""
    #: The positive evidence that this kind's artifact actually executed --
    #: a counter or echo the runtime emits, named here so a measurement can be
    #: checked rather than assumed. See the class docstring.
    execution_signal: str = ""
    #: The machine-evaluable form of ``execution_signal``. Takes the evidence a
    #: runtime reported and returns ``(ran, reason)``. A prose signal nothing
    #: can evaluate is how the incident in the class docstring happened twice,
    #: so registration requires this too.
    evidence_check: Any = None

    @property
    def permitted(self) -> frozenset[str]:
        return self.required | self.optional


_REGISTRY: dict[str, TargetKind] = {}


def counter_evidence(*names: str) -> Any:
    """Evidence check: at least one of ``names`` counted above zero.

    A missing counter is not zero -- it means the runtime never reported the
    thing, which is exactly the case that must not be read as "ran".
    """

    def check(observed: Mapping[str, Any]) -> tuple[bool, str]:
        missing = [name for name in names if observed.get(name) is None]
        if missing:
            return False, f"no evidence reported for {', '.join(sorted(missing))}"
        total = 0
        for name in names:
            try:
                total += int(observed[name] or 0)
            except (TypeError, ValueError):
                return False, f"{name} is not a count: {observed[name]!r}"
        if total > 0:
            return True, ""
        return False, f"{' + '.join(names)} == 0"

    return check


def backend_echo_evidence(
    declared_field: str, effective_field: str
) -> Any:
    """Evidence check: the runtime echoed back the implementation we asked for.

    Frameworks substitute a fallback when an optional backend cannot be
    imported, so equality here is the only proof the requested one ran.
    """

    def check(observed: Mapping[str, Any]) -> tuple[bool, str]:
        declared = observed.get(declared_field)
        effective = observed.get(effective_field)
        if not declared:
            return False, f"no {declared_field} declared"
        if not effective:
            return False, f"runtime reported no {effective_field}"
        if declared != effective:
            return (
                False,
                f"{effective_field}={effective!r} but {declared_field}={declared!r}",
            )
        return True, ""

    return check


def register_target_kind(kind: TargetKind) -> TargetKind:
    """Register an artifact kind. Re-registering the same name is an error.

    Silent replacement would let an import-order change alter what validates,
    which is the kind of failure that only shows up in a campaign.
    """
    if kind.name in _REGISTRY:
        raise ValueError(f"target kind {kind.name!r} is already registered")
    if not kind.execution_signal:
        raise ValueError(
            f"target kind {kind.name!r} must declare an execution_signal: the "
            f"observable proof its artifact actually ran. A kind without one "
            f"admits measurements of interventions that never happened "
            f"(see the class docstring)."
        )
    if not callable(kind.evidence_check):
        raise ValueError(
            f"target kind {kind.name!r} must declare a callable evidence_check "
            f"so its execution_signal can be evaluated rather than read. A "
            f"prose-only signal is one nothing enforces."
        )
    _REGISTRY[kind.name] = kind
    return kind


def target_kind_spec(name: str) -> TargetKind | None:
    return _REGISTRY.get(name)


def known_target_kinds() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# -- built-in kinds -----------------------------------------------------

register_target_kind(
    TargetKind(
        name=MODULE,
        description="replaces a whole module's forward",
        execution_signal="dispatch.candidate_calls > 0",
        evidence_check=counter_evidence("candidate_calls"),
    )
)

register_target_kind(
    TargetKind(
        name=SUBGRAPH,
        required=frozenset({"capture_mode"}),
        optional=frozenset(
            {"selected_node_ids", "boundary_refs", "output_node_ids"}
        ),
        description="rewrites selected nodes inside a captured region",
        execution_signal="dispatch.candidate_calls > 0",
        evidence_check=counter_evidence("candidate_calls"),
    )
)

register_target_kind(
    TargetKind(
        name=ATTENTION,
        required=frozenset({"attention_backend"}),
        optional=frozenset({"attention_config"}),
        replaces_region=False,
        description="selects an attention backend implementation",
        execution_signal="effective_backend == declared attention_backend",
        evidence_check=backend_echo_evidence(
            "attention_backend", "effective_backend"
        ),
    )
)


register_target_kind(
    TargetKind(
        name=SCHEDULE_TRANSFORM,
        required=frozenset({"transform_family", "transform_policy"}),
        replaces_region=False,
        description="wraps the denoising loop and may skip steps",
        execution_signal="loop_transform.steps_skipped + hook_invocations > 0",
        evidence_check=counter_evidence("steps_skipped", "hook_invocations"),
    )
)


def validate_kind_fields(
    kind_name: str,
    declared: Mapping[str, Any],
    *,
    common: frozenset[str] = frozenset(),
    region_fields: Mapping[str, frozenset[str]] | None = None,
) -> TargetKind:
    """Check that ``declared`` carries exactly what ``kind_name`` allows.

    Args:
        kind_name: the declared ``target_kind``.
        declared: the raw operation-identity mapping.
        common: fields every kind shares (identity, fingerprint, and so on).
            Anything outside ``common`` and outside this kind's permitted set
            is rejected. Passing it explicitly is what keeps this check
            fail-closed: without it, a field added to the schema but never
            attached to a kind would validate silently for *every* kind, which
            is the quiet way a contract stops contracting.
        region_fields: kind-specific field names keyed by owning kind.
            Defaults to the registry, so a field belonging to a *different*
            kind is reported specifically -- that almost always means an
            upstream writer picked the wrong kind, and "unknown field" would
            send the reader looking for a typo instead.

    Returns:
        The resolved :class:`TargetKind`.

    Raises:
        ValueError: with a message naming the field and the kind. The caller
            wraps this in its own error type so bundle and workload validation
            keep their existing message shapes.
    """
    spec = target_kind_spec(kind_name)
    if spec is None:
        raise ValueError(
            f"unknown target_kind {kind_name!r}; "
            f"expected one of {list(known_target_kinds())}"
        )

    missing = sorted(name for name in spec.required if declared.get(name) is None)
    if missing:
        raise ValueError(
            f"target_kind {kind_name!r} ({spec.description}) requires "
            f"{missing}"
        )

    if region_fields is None:
        region_fields = {
            name: spec.permitted for name, spec in _REGISTRY.items()
        }
    owned_elsewhere: dict[str, str] = {}
    for other_name, fields in region_fields.items():
        if other_name == kind_name:
            continue
        for name in fields:
            owned_elsewhere.setdefault(name, other_name)

    for name in declared:
        if name in spec.permitted or name in common:
            continue
        owner = owned_elsewhere.get(name)
        if owner is not None:
            raise ValueError(
                f"target_kind {kind_name!r} must not declare {name!r}, "
                f"which belongs to a {owner!r} target"
            )
        raise ValueError(
            f"target_kind {kind_name!r} must not declare {name!r}; it belongs "
            f"to no registered kind"
        )
    return spec


def execution_signal_for(kind_name: str) -> str:
    """The observable proof that ``kind_name``'s artifact actually ran.

    Used by measurement harnesses to decide what to assert before interpreting
    any timing. A harness that cannot evaluate the signal should treat the arms
    as undifferentiated rather than assume they differed.
    """
    spec = target_kind_spec(kind_name)
    if spec is None:
        raise ValueError(f"unknown target_kind {kind_name!r}")
    return spec.execution_signal


def evaluate_execution_signal(
    kind_name: str, observed: Mapping[str, Any]
) -> tuple[bool, str]:
    """Did ``kind_name``'s artifact demonstrably run, given ``observed``?

    Returns ``(ran, reason)``; ``reason`` is empty when it ran and otherwise
    says what was missing or zero. An unknown kind returns ``False`` rather
    than raising: a harness that cannot identify the kind has no basis for
    claiming the arms differed, which is the whole point of the check.
    """
    spec = target_kind_spec(kind_name)
    if spec is None:
        return False, f"unknown target_kind {kind_name!r}"
    return spec.evidence_check(observed)
