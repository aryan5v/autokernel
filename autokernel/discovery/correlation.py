"""Offline correlation between profiler rows and captured FX regions.

This module implements the logic to match profiler operator rows with FX graph
regions, aggregate timing data, and populate DiscoveryReport with correlated
metrics without double-counting nested scopes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .profiler_parse import parse_key_averages_rows
from .ranking import optimistic_e2e_improvement
from .safety import normalize_op_name, reject_region
from .types import (
    DiscoveryReport,
    GraphRegion,
    OperatorHotspot,
)


@dataclass(frozen=True)
class ScopeMatch:
    """Result of matching a profiler row to a region."""

    profiler_row: OperatorHotspot
    region: GraphRegion
    confidence: float
    match_reason: str


@dataclass
class RegionAccumulator:
    """Aggregate data for regions with the same fingerprint."""

    base_region: GraphRegion
    total_cuda_time_us: float = 0.0
    total_self_cuda_time_us: float = 0.0
    total_calls: int = 0
    shape_frequencies: dict[str, int] = field(default_factory=dict)
    matched_profiler_rows: list[OperatorHotspot] = field(default_factory=list)
    unmatched_scopes: list[str] = field(default_factory=list)
    capture_failures: list[str] = field(default_factory=list)


def _match_scope_to_region(
    profiler_row: OperatorHotspot,
    region: GraphRegion,
) -> ScopeMatch | None:
    """Attempt to match a profiler row to a region based on scope heuristics.

    Matching strategies (in order of precedence):
    1. Exact parent_module match
    2. Op key appears in region operations
    3. Name similarity between profiler row and region

    Returns None if no reasonable match is found.
    """
    # Shape-specific record_function ranges exported by FastVideo carry the
    # captured region name verbatim. Prefer this identity over broader module
    # or operation heuristics.
    if profiler_row.name == region.name:
        return ScopeMatch(
            profiler_row=profiler_row,
            region=region,
            confidence=1.0,
            match_reason="exact_region_name",
        )

    # Strategy 1: Exact parent_module match
    if profiler_row.parent_module and region.parent_module:
        if profiler_row.parent_module == region.parent_module:
            return ScopeMatch(
                profiler_row=profiler_row,
                region=region,
                confidence=0.9,
                match_reason="exact_parent_module",
            )
        # Check for hierarchical relationship (e.g., "blocks.0.attn" matches "blocks.0")
        if (profiler_row.parent_module.startswith(region.parent_module + ".") or
            region.parent_module.startswith(profiler_row.parent_module + ".")):
            return ScopeMatch(
                profiler_row=profiler_row,
                region=region,
                confidence=0.7,
                match_reason="hierarchical_parent_module",
            )

    # Strategy 2: Op key appears in region operations. Region operations are
    # normalized during FX capture (overload suffixes stripped), while
    # profiler rows keep overload-qualified names, so normalize both sides.
    normalized_op_key = normalize_op_name(profiler_row.op_key)
    if normalized_op_key in {
        normalize_op_name(op) for op in region.operations
    }:
        return ScopeMatch(
            profiler_row=profiler_row,
            region=region,
            confidence=0.6,
            match_reason="op_key_in_operations",
        )

    # Strategy 3: Name similarity
    profiler_name_lower = profiler_row.name.lower()
    region_name_lower = region.name.lower()
    if (profiler_name_lower in region_name_lower or
        region_name_lower in profiler_name_lower):
        return ScopeMatch(
            profiler_row=profiler_row,
            region=region,
            confidence=0.5,
            match_reason="name_similarity",
        )

    return None


def _calculate_exclusive_time(
    profiler_row: OperatorHotspot,
    parent_time_us: float,
) -> tuple[float, float]:
    """Calculate exclusive CUDA time avoiding double-counting.

    Returns (cuda_time_us, self_cuda_time_us) where:
    - cuda_time_us is the total time (inclusive)
    - self_cuda_time_us is exclusive time, capped at parent_time_us if nested
    """
    # Use the profiler's self_cuda_time_us as the exclusive time
    exclusive = profiler_row.self_cuda_time_us
    
    # If this is nested within a parent region, ensure we don't exceed parent time
    if parent_time_us > 0 and exclusive > parent_time_us:
        # This can happen if profiler attribution is inconsistent
        # Conservative approach: cap at parent time
        exclusive = parent_time_us
    
    return profiler_row.cuda_time_us, exclusive


def _aggregate_region_timing(
    region: GraphRegion,
    profiler_rows: Sequence[OperatorHotspot],
) -> tuple[float, float, int]:
    """Aggregate timing data for a region from matched profiler rows.

    Returns (total_cuda_time_us, total_self_cuda_time_us, total_calls).
    Uses exclusive time to avoid double-counting nested scopes.
    """
    if not profiler_rows:
        return region.cuda_time_us, region.self_cuda_time_us, region.calls

    # A shape-specific record_function range names the region itself, and its
    # inclusive CUDA time is the authoritative duration for the whole region:
    # every operator row that also matched is a *nested child* of that range,
    # not additional work beside it. Adding the two inflated Cosmos's
    # transformer attribution from 275.7s to 485.1s and made a 21.8ms text
    # encoder range look like 199.5s. When FastVideo emitted the named ranges,
    # use only those.
    #
    # ``calls`` means "how many times was this region invoked", because that is
    # what the impact model multiplies a per-call saving by. Summing the call
    # counts of every matched profiler row answers a different question -- how
    # many kernel launches happened inside the region -- and the two diverge by
    # orders of magnitude once a region matches more than one row. Run r4
    # reported calls=195661 for transformer.model.transformer_blocks, a region
    # FastVideo's dispatcher observed being invoked 1151 times; the 195661 was
    # the aten-event count across 48 blocks x 8 steps x 3 generations. The
    # named range is one range per invocation, so it is authoritative here too.
    named_rows = [row for row in profiler_rows if row.name == region.name]
    if named_rows:
        total_cuda = sum(row.cuda_time_us for row in named_rows)
        # The range itself launches no kernel, so its self time is normally
        # zero; the inclusive duration is what the region actually costs.
        total_self_cuda = total_cuda
        total_calls = sum(row.calls for row in named_rows)
    else:
        # No range for this region: operator rows are siblings, so exclusive
        # self time is the only non-double-counting attribution available.
        total_cuda = sum(row.cuda_time_us for row in profiler_rows)
        total_self_cuda = sum(row.self_cuda_time_us for row in profiler_rows)
        # Every matched row saw the region at least once, so the largest single
        # row bounds the invocation count from below without inventing launches
        # the region never made.
        total_calls = max(row.calls for row in profiler_rows)

    # Fall back to region's own timing if no profiler data
    if total_self_cuda == 0:
        total_self_cuda = region.self_cuda_time_us
    if total_cuda == 0:
        total_cuda = region.cuda_time_us
    if total_calls == 0:
        total_calls = region.calls

    return total_cuda, total_self_cuda, total_calls


def _deduplicate_regions_by_fingerprint(
    regions: Sequence[GraphRegion],
) -> dict[str, RegionAccumulator]:
    """Group equivalent regions by their stable graph fingerprint."""
    accumulators: dict[str, RegionAccumulator] = {}

    for region in regions:
        fingerprint = region.fingerprint
        if fingerprint not in accumulators:
            accumulators[fingerprint] = RegionAccumulator(base_region=region)
        
        # Merge shape frequencies
        if region.shape_frequency:
            for shape_key, count in region.shape_frequency.items():
                accumulators[fingerprint].shape_frequencies[shape_key] = (
                    accumulators[fingerprint].shape_frequencies.get(shape_key, 0) + count
                )

    return accumulators


def correlate_profiler_to_regions(
    profiler_rows: Sequence[OperatorHotspot],
    fx_regions: Sequence[GraphRegion],
    *,
    total_cuda_time_us: float,
) -> tuple[tuple[GraphRegion, ...], tuple[OperatorHotspot, ...]]:
    """Correlate profiler rows with FX regions and populate timing data.

    This is the main entry point for offline correlation. It:
    1. Matches profiler rows to regions based on scope heuristics
    2. Calculates exclusive CUDA time without double-counting nested scopes
    3. Deduplicates equivalent regions using stable graph fingerprints
    4. Aggregates timing, call counts, and shape frequencies
    5. Computes confidence and rejection reasons
    6. Returns populated GraphRegion tuples plus the unmatched rows

    Args:
        profiler_rows: OperatorHotspot rows from the profiler
        fx_regions: GraphRegion instances from FX capture
        total_cuda_time_us: Total end-to-end CUDA time for percentage calculations

    Returns:
        A ``(regions, unmatched_rows)`` pair: populated GraphRegion
        instances, and the profiler rows that matched no captured region.
        Unmatched rows are reported separately because they carry no input
        metadata and therefore cannot form a valid, serializable region.
    """
    # Step 1: Group regions by fingerprint for deduplication
    fingerprint_groups = _deduplicate_regions_by_fingerprint(fx_regions)

    # Step 2: Match profiler rows to regions
    matched_regions: dict[str, list[ScopeMatch]] = defaultdict(list)
    unmatched_rows: list[OperatorHotspot] = []

    for row in profiler_rows:
        candidates: list[ScopeMatch] = []

        # Try to match against each region group (use first region as representative)
        for accumulator in fingerprint_groups.values():
            match = _match_scope_to_region(row, accumulator.base_region)
            if match:
                candidates.append(match)

        best_match: ScopeMatch | None = None
        if candidates:
            best_confidence = max(match.confidence for match in candidates)
            strongest = [
                match for match in candidates
                if match.confidence == best_confidence
            ]
            # Never assign an aggregate operator row arbitrarily when the same
            # op occurs in several captured regions. A shape-specific scope
            # range or other unique match is required for trustworthy timing.
            if len(strongest) == 1:
                best_match = strongest[0]

        if best_match is not None:
            matched_regions[best_match.region.fingerprint].append(best_match)
        else:
            unmatched_rows.append(row)

    # Step 3: Aggregate timing data and build final regions
    final_regions: list[GraphRegion] = []

    for fingerprint, accumulator in fingerprint_groups.items():
        matches = matched_regions.get(fingerprint, [])
        matched_rows = [m.profiler_row for m in matches]

        # Aggregate timing from matched profiler rows
        total_cuda, total_self_cuda, total_calls = _aggregate_region_timing(
            accumulator.base_region,
            matched_rows,
        )

        # Merge shape frequencies
        merged_shape_freq = dict(accumulator.shape_frequencies)
        for match in matches:
            if match.profiler_row.input_shapes:
                # Create a shape key from input shapes
                shape_parts = [
                    "x".join(str(d) for d in shape)
                    for shape in match.profiler_row.input_shapes
                    if shape
                ]
                if shape_parts:
                    shape_key = "|".join(shape_parts)
                    merged_shape_freq[shape_key] = (
                        merged_shape_freq.get(shape_key, 0)
                        + match.profiler_row.calls
                    )

        # Calculate confidence
        safety_reasons = tuple(reject_region(accumulator.base_region.operations))
        confidence = 0.4  # Base confidence
        if not safety_reasons:
            confidence += 0.3  # Safety bonus
        if matches:
            confidence += 0.2  # Profiler match bonus
        if total_calls >= 8:
            confidence += 0.1  # Call count bonus
        if merged_shape_freq:
            confidence += 0.1  # Shape frequency bonus
        confidence = min(1.0, confidence)

        # Calculate percentage of end-to-end time
        e2e_share = 0.0
        if total_cuda_time_us > 0:
            e2e_share = 100.0 * total_self_cuda / total_cuda_time_us

        # Estimate maximum end-to-end improvement
        estimated_improvement = optimistic_e2e_improvement(e2e_share / 100.0)

        # Build rejection reasons
        rejection_reasons = list(safety_reasons)
        if not matches:
            rejection_reasons.append("no_profiler_match")
        if accumulator.base_region.rejection_reasons:
            rejection_reasons.extend(accumulator.base_region.rejection_reasons)

        # Create the populated region
        populated_region = GraphRegion.build(
            name=accumulator.base_region.name,
            operations=accumulator.base_region.operations,
            inputs=accumulator.base_region.inputs,
            outputs=accumulator.base_region.outputs,
            dependencies=accumulator.base_region.dependencies,
            parent_module=accumulator.base_region.parent_module,
            safe_constants=accumulator.base_region.safe_constants,
            pattern_family=accumulator.base_region.pattern_family,
            rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
            calls=total_calls,
            shape_frequency=merged_shape_freq or None,
            cuda_time_us=total_cuda,
            self_cuda_time_us=total_self_cuda,
            attributes={
                **(accumulator.base_region.attributes or {}),
                "e2e_share_pct": round(e2e_share, 4),
                "estimated_max_e2e_improvement": round(estimated_improvement, 4),
                "confidence": round(confidence, 4),
                "matched_profiler_rows": len(matched_rows),
            },
        )

        final_regions.append(populated_region)

    return tuple(final_regions), tuple(unmatched_rows)


def correlate_discovery_report(
    profiler_export_rows: Sequence[Mapping[str, Any]],
    fx_discovery_report: DiscoveryReport,
) -> DiscoveryReport:
    """Correlate profiler data with an existing FX discovery report.

    This function takes profiler export rows and a discovery report containing
    FX-captured regions, correlates them, and returns a new discovery report with
    populated timing data.

    Args:
        profiler_export_rows: Raw profiler export rows (list of dicts)
        fx_discovery_report: Discovery report from FX capture (CPU-only)

    Returns:
        New DiscoveryReport with correlated timing data
    """
    # Parse profiler rows
    profiler_operators = parse_key_averages_rows(profiler_export_rows)
    total_cuda_time_us = fx_discovery_report.total_cuda_time_us
    if total_cuda_time_us <= 0:
        # FX capture is intentionally CPU-only and therefore normally carries
        # no device total.  Use the profiler's exclusive-time accounting here;
        # summing inclusive CUDA time would count nested ranges more than once.
        total_cuda_time_us = sum(
            max(operator.self_cuda_time_us, 0.0)
            for operator in profiler_operators
        )

    # Correlate profiler rows with FX regions
    searchable_regions, unmatched_rows = correlate_profiler_to_regions(
        profiler_operators,
        fx_discovery_report.regions,
        total_cuda_time_us=total_cuda_time_us,
    )
    unsupported = [
        item.as_dict() for item in fx_discovery_report.unsupported
    ]
    if unmatched_rows:
        unsupported.append(
            {
                "op_name": "profiler::unmatched",
                "reason": (
                    f"{len(unmatched_rows)} profiler row(s) did not match a "
                    "captured FX region"
                ),
                "count": len(unmatched_rows),
                "scope": "profiler_correlation",
            }
        )

    # Create new discovery report with populated regions
    return DiscoveryReport.from_dict(
        {
            "schema_version": fx_discovery_report.schema_version,
            "producer": dict(fx_discovery_report.producer),
            "workload": dict(fx_discovery_report.workload),
            "environment": dict(fx_discovery_report.environment),
            "total_cuda_time_us": total_cuda_time_us,
            "operators": [op.as_dict() for op in profiler_operators],
            "regions": [region.as_dict() for region in searchable_regions],
            "graph_breaks": [item.as_dict() for item in fx_discovery_report.graph_breaks],
            "unsupported": unsupported,
        }
    )
