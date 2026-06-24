#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FUSION_TYPES = {
    "Add",
    "AddRmsNormBias",
    "Cast",
    "Reshape",
    "RmsNorm",
    "Slice",
    "SwiGlu",
    "Transpose",
}
ROUTING_TYPES = {
    "MoeGatingTopK",
    "MoeInitRoutingCustom",
    "MoeTokenPermute",
    "MoeTokenUnpermute",
    "Sort",
    "TopK",
}
PREFILL_TYPES = {
    "FusedInferAttentionScore",
    "FlashAttentionScore",
    "MatMul",
    "MatMulV2",
    "ReshapeAndCache",
    "ReshapeAndCacheNdKernel",
}


def _float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _resolve_output_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.name == "ASCEND_PROFILER_OUTPUT":
        return candidate
    nested = candidate / "ASCEND_PROFILER_OUTPUT"
    if nested.exists():
        return nested
    return candidate


def _load_benchmark(path: str | Path | None) -> dict[str, float | str]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    with source.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}

    keys = (
        "label",
        "median_ttft_ms",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "output_throughput",
        "total_token_throughput",
        "request_throughput",
    )
    result: dict[str, float | str] = {}
    for key in keys:
        if key not in payload:
            continue
        if key == "label":
            result[key] = str(payload[key])
        else:
            result[key] = _float(payload[key])
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _find_sew_moe_trace(output_dir: Path) -> Path | None:
    candidates = (
        output_dir.parent / "moe_offload_trace.jsonl",
        output_dir / "moe_offload_trace.jsonl",
        output_dir.parent / "sew_moe_trace.jsonl",
        output_dir / "sew_moe_trace.jsonl",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_pipeline_profile(output_dir: Path) -> Path | None:
    candidates = (
        output_dir.parent / "sew_moe_profile.jsonl",
        output_dir / "sew_moe_profile.jsonl",
        output_dir.parent / "moe_pipeline_profile.jsonl",
        output_dir / "moe_pipeline_profile.jsonl",
        output_dir.parent / "moe_offload_profile.jsonl",
        output_dir / "moe_offload_profile.jsonl",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _summarize_sew_moe_trace(trace_path: Path | None) -> dict[str, Any]:
    if trace_path is None:
        return {
            "record_count": 0,
            "note": "No SEW-MoE active expert trace found.",
        }

    records = _read_jsonl(trace_path)
    if not records:
        return {
            "record_count": 0,
            "trace_path": str(trace_path),
            "note": "SEW-MoE active expert trace is empty.",
        }

    fanouts_by_source: dict[str, list[int]] = defaultdict(list)
    layer_fanout: dict[tuple[int, str], int] = defaultdict(int)
    layer_grouped_tokens: dict[int, int] = defaultdict(int)
    grouped_fanouts: list[int] = []
    grouped_layer_fanout: dict[int, int] = defaultdict(int)
    signature_counts: dict[str, int] = defaultdict(int)

    for record in records:
        source = str(record.get("source") or "unknown")
        layer_id = _int(record.get("layer_id"))
        fanout = _int(record.get("fanout"))
        fanouts_by_source[source].append(fanout)
        layer_fanout[(layer_id, source)] = max(layer_fanout[(layer_id, source)], fanout)

        if source == "grouped_dispatch":
            grouped_fanouts.append(fanout)
            grouped_layer_fanout[layer_id] = max(grouped_layer_fanout[layer_id], fanout)
            layer_grouped_tokens[layer_id] += _int(record.get("num_tokens"))
            signature = record.get("group_list_signature")
            if signature:
                signature_counts[str(signature)] += 1

    top_layers_by_fanout = [
        {
            "layer_id": layer_id,
            "source": source,
            "max_fanout": max_fanout,
        }
        for (layer_id, source), max_fanout in layer_fanout.items()
    ]
    top_layers_by_fanout.sort(key=lambda item: item["max_fanout"], reverse=True)

    top_layers_by_grouped_tokens = [
        {
            "layer_id": layer_id,
            "grouped_tokens": grouped_tokens,
        }
        for layer_id, grouped_tokens in layer_grouped_tokens.items()
    ]
    top_layers_by_grouped_tokens.sort(key=lambda item: item["grouped_tokens"], reverse=True)

    top_group_list_signatures = [
        {
            "signature": signature,
            "count": count,
        }
        for signature, count in signature_counts.items()
    ]
    top_group_list_signatures.sort(key=lambda item: item["count"], reverse=True)
    max_grouped_fanout = max(grouped_fanouts) if grouped_fanouts else 0
    high_fanout_layers = [
        {
            "layer_id": layer_id,
            "max_fanout": max_fanout,
        }
        for layer_id, max_fanout in grouped_layer_fanout.items()
        if max_fanout == max_grouped_fanout and max_grouped_fanout > 0
    ]
    high_fanout_layers.sort(key=lambda item: item["layer_id"])

    return {
        "record_count": len(records),
        "trace_path": str(trace_path),
        "fanout_by_source": {
            source: _stats(values) for source, values in sorted(fanouts_by_source.items())
        },
        "top_layers_by_fanout": top_layers_by_fanout[:10],
        "top_layers_by_grouped_tokens": top_layers_by_grouped_tokens[:10],
        "grouped_signature_total_count": sum(signature_counts.values()),
        "top_group_list_signatures": top_group_list_signatures[:10],
        "slot_budget_hint": {
            "min_slots_per_layer": max_grouped_fanout,
            "max_grouped_fanout": max_grouped_fanout,
            "mean_grouped_fanout": round(sum(grouped_fanouts) / len(grouped_fanouts), 2)
            if grouped_fanouts
            else 0.0,
            "high_fanout_layers": high_fanout_layers[:10],
        },
    }


def _summarize_pipeline_profile(profile_path: Path | None) -> dict[str, Any]:
    if profile_path is None:
        return {
            "record_count": 0,
            "note": "No SEW-MoE pipeline profile found.",
        }

    all_records = _read_jsonl(profile_path)
    records = [
        record for record in all_records
        if record.get("event") == "moe_pipeline_timing"
    ]
    if not records:
        summary = {
            "record_count": 0,
            "profile_path": str(profile_path),
            "note": "SEW-MoE pipeline profile is empty.",
        }
        gate_summary = _summarize_compute_bucket_gate_events(all_records)
        if gate_summary:
            summary["compute_bucket_fast_path_gate"] = gate_summary
        return summary

    stage_keys = ("stage_t_ms", "stage_r_ms", "stage_c_ms", "stage_m_ms")
    means = {}
    for key in stage_keys:
        values = [_float(record.get(key)) for record in records]
        means[key] = sum(values) / len(values) if values else 0.0

    total_mean = sum(means.values())
    total_excl_t = means["stage_r_ms"] + means["stage_c_ms"] + means["stage_m_ms"]
    overlap_ratio = (
        min(1.0, (means["stage_r_ms"] + means["stage_c_ms"]) / means["stage_t_ms"])
        if means["stage_t_ms"] > 0
        else 0.0
    )
    summary = {
        "record_count": len(records),
        "profile_path": str(profile_path),
        "stages": {key: {"mean": round(value, 4)} for key, value in means.items()},
        "total_pipeline_ms": {"mean": round(total_mean, 4)},
        "total_excl_transfer_ms": {"mean": round(total_excl_t, 4)},
        "fractions": {
            "t_frac": means["stage_t_ms"] / total_mean if total_mean > 0 else 0.0,
            "r_frac": means["stage_r_ms"] / total_mean if total_mean > 0 else 0.0,
            "c_frac": means["stage_c_ms"] / total_mean if total_mean > 0 else 0.0,
            "m_frac": means["stage_m_ms"] / total_mean if total_mean > 0 else 0.0,
        },
        "overlap_potential_ratio": overlap_ratio,
    }
    gate_summary = _summarize_compute_bucket_gate_events(all_records)
    if gate_summary:
        summary["compute_bucket_fast_path_gate"] = gate_summary
    offload_timeline_summary = _summarize_offload_timeline_events(all_records)
    if offload_timeline_summary:
        summary["offload_timeline"] = offload_timeline_summary
    return summary


def _summarize_offload_timeline_events(records: list[dict[str, Any]]) -> dict[str, Any]:
    timeline_records = [
        record for record in records
        if record.get("event") == "moe_offload_timeline"
    ]
    if not timeline_records:
        return {}

    by_name: dict[str, list[float]] = defaultdict(list)
    cache_hits = 0
    cache_misses = 0
    h2d_bytes = 0
    for record in timeline_records:
        name = str(record.get("name") or "unknown")
        duration_us = _float(record.get("duration_us"))
        if duration_us <= 0 and _float(record.get("seconds")) > 0:
            duration_us = _float(record.get("seconds")) * 1_000_000
        by_name[name].append(duration_us / 1000.0)
        payload = record.get("payload") or {}
        if name == "slot_cache_lookup":
            if bool(payload.get("cache_hit")):
                cache_hits += 1
            else:
                cache_misses += 1
        if name == "expert_h2d_load_sync":
            h2d_bytes += _int(payload.get("bytes"))

    stage_rows = []
    for name, values_ms in by_name.items():
        total_ms = sum(values_ms)
        stage_rows.append({
            "name": name,
            "count": len(values_ms),
            "total_ms": round(total_ms, 4),
            "mean_ms": round(total_ms / len(values_ms), 4) if values_ms else 0.0,
            "max_ms": round(max(values_ms), 4) if values_ms else 0.0,
        })
    stage_rows.sort(key=lambda item: item["total_ms"], reverse=True)
    lookup_total = cache_hits + cache_misses
    return {
        "record_count": len(timeline_records),
        "stages": stage_rows,
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": round(cache_hits / lookup_total, 4) if lookup_total else 0.0,
        },
        "h2d_bytes": h2d_bytes,
    }


def _summarize_compute_bucket_gate_events(records: list[dict[str, Any]]) -> dict[str, Any]:
    gate_records = [
        record for record in records
        if record.get("event") == "compute_bucket_fast_path_gate"
        or record.get("name") == "compute_bucket_fast_path_gate"
    ]
    if not gate_records:
        return {}

    enabled_count = 0
    original_counts: list[int] = []
    compact_counts: list[int] = []
    fallback_reasons: dict[str, int] = defaultdict(int)
    for record in gate_records:
        payload = record.get("payload") or {}
        enabled = bool(payload.get("enabled"))
        if enabled:
            enabled_count += 1
        else:
            reason = str(payload.get("reason") or "unknown")
            fallback_reasons[reason] += 1
        original = _int(payload.get("original_expert_count"))
        compact = _int(payload.get("compact_expert_count"))
        if original > 0:
            original_counts.append(original)
        if compact > 0:
            compact_counts.append(compact)

    record_count = len(gate_records)
    mean_original = sum(original_counts) / len(original_counts) if original_counts else 0.0
    mean_compact = sum(compact_counts) / len(compact_counts) if compact_counts else 0.0
    fallback_items = [
        {"reason": reason, "count": count}
        for reason, count in fallback_reasons.items()
    ]
    fallback_items.sort(key=lambda item: item["count"], reverse=True)
    return {
        "record_count": record_count,
        "enabled_count": enabled_count,
        "fallback_count": record_count - enabled_count,
        "enabled_percent": round(enabled_count / record_count * 100.0, 1),
        "mean_original_expert_count": round(mean_original, 2),
        "mean_compact_expert_count": round(mean_compact, 2),
        "mean_compaction_ratio": round(mean_compact / mean_original, 4) if mean_original > 0 else 0.0,
        "top_fallback_reasons": fallback_items[:5],
    }


def _summarize_ops(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    hotspots = []
    total_us = sum(_float(row.get("Total Time(us)")) for row in rows)
    for row in rows:
        time_us = _float(row.get("Total Time(us)"))
        if time_us <= 0:
            continue
        ratio = _float(row.get("Ratio(%)"))
        if ratio == 0 and total_us > 0:
            ratio = time_us / total_us * 100.0
        hotspots.append({
            "name": row.get("OP Type", ""),
            "core_type": row.get("Core Type", ""),
            "count": _int(row.get("Count")),
            "total_us": time_us,
            "avg_us": _float(row.get("Avg Time(us)")),
            "max_us": _float(row.get("Max Time(us)")),
            "ratio_percent": ratio,
        })
    hotspots.sort(key=lambda item: item["total_us"], reverse=True)
    return hotspots


def _summarize_kernels(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "type": "",
            "count": 0,
            "duration_us": 0.0,
            "wait_us": 0.0,
            "max_us": 0.0,
            "cube_sum": 0.0,
            "cube_count": 0,
            "mte_us": 0.0,
        })
    top_by_duration = []
    short_kernel_counts: dict[str, int] = defaultdict(int)
    total_duration_us = 0.0
    total_wait_us = 0.0
    total_mte_us = 0.0

    for row in rows:
        op_type = row.get("Type") or row.get("Name", "")
        duration_us = _float(row.get("Duration(us)"))
        wait_us = _float(row.get("Wait Time(us)"))
        cube = _float(row.get("cube_utilization(%)"))
        mte_us = (
            _float(row.get("aic_mte1_time(us)"))
            + _float(row.get("aic_mte2_time(us)"))
            + _float(row.get("aiv_mte2_time(us)"))
            + _float(row.get("aiv_mte3_time(us)"))
        )

        total_duration_us += duration_us
        total_wait_us += wait_us
        total_mte_us += mte_us
        bucket = by_type[op_type]
        bucket["type"] = op_type
        bucket["count"] += 1
        bucket["duration_us"] += duration_us
        bucket["wait_us"] += wait_us
        bucket["max_us"] = max(bucket["max_us"], duration_us)
        bucket["mte_us"] += mte_us
        if cube > 0:
            bucket["cube_sum"] += cube
            bucket["cube_count"] += 1
        if duration_us and duration_us < 50.0:
            short_kernel_counts[op_type] += 1
        top_by_duration.append({
            "name": row.get("Name", ""),
            "type": op_type,
            "core": row.get("Accelerator Core", ""),
            "duration_us": duration_us,
            "wait_us": wait_us,
            "cube_utilization_percent": cube,
        })

    type_rows = []
    for item in by_type.values():
        count = item["count"]
        cube_count = item.pop("cube_count")
        cube_sum = item.pop("cube_sum")
        item["avg_us"] = item["duration_us"] / count if count else 0.0
        item["avg_cube_utilization_percent"] = cube_sum / cube_count if cube_count else 0.0
        type_rows.append(item)

    type_rows.sort(key=lambda item: item["duration_us"], reverse=True)
    top_by_duration.sort(key=lambda item: item["duration_us"], reverse=True)

    return {
        "total_duration_us": total_duration_us,
        "total_wait_us": total_wait_us,
        "wait_ratio_percent": total_wait_us / total_duration_us * 100.0 if total_duration_us else 0.0,
        "mte_ratio_percent": total_mte_us / total_duration_us * 100.0 if total_duration_us else 0.0,
        "by_type": type_rows[:20],
        "top_by_duration": top_by_duration[:20],
        "short_kernel_counts": dict(
            sorted(short_kernel_counts.items(), key=lambda item: item[1], reverse=True)[:20]),
    }


def _summarize_operators(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    operators = []
    for row in rows:
        host_us = _float(row.get("Host Total Duration(us)"))
        device_us = _float(row.get("Device Total Duration(us)"))
        if host_us <= 0 and device_us <= 0:
            continue
        operators.append({
            "name": row.get("Name", ""),
            "host_total_us": host_us,
            "device_total_us": device_us,
            "device_aicore_total_us": _float(row.get("Device Total Duration With AICore(us)")),
        })
    operators.sort(key=lambda item: max(item["host_total_us"], item["device_total_us"]), reverse=True)
    return operators[:30]


def _summarize_step_trace(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            if key and key not in {"Device_id", "Step"}:
                totals[key] += _float(value)
    stage = totals.get("Stage", 0.0)
    if stage:
        totals["Free Ratio(%)"] = totals.get("Free", 0.0) / stage * 100.0
        totals["Computing Ratio(%)"] = totals.get("Computing", 0.0) / stage * 100.0
    return dict(totals)


def _op_by_name(op_hotspots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in op_hotspots}


def _add_opportunity(
    items: list[dict[str, Any]],
    category: str,
    stage: str,
    evidence: str,
    recommendation: str,
    expected_metric: str,
):
    items.append({
        "category": category,
        "stage": stage,
        "evidence": evidence,
        "recommendation": recommendation,
        "expected_metric": expected_metric,
    })


def _build_opportunities(
    phase: str,
    op_hotspots: list[dict[str, Any]],
    kernel_summary: dict[str, Any],
    step_trace: dict[str, float],
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    ops = _op_by_name(op_hotspots)

    grouped = ops.get("GroupedMatmul")
    if grouped:
        expected = "TPOT" if phase == "decode" else "TTFT and TPOT"
        _add_opportunity(
            opportunities,
            "moe_grouped_matmul",
            "Decode" if phase != "prefill" else "Prefill spillover",
            (
                f"GroupedMatmul uses {grouped['total_us']:.1f} us "
                f"across {grouped['count']} calls ({grouped['ratio_percent']:.1f}%)."
            ),
            (
                "Inspect token-per-expert distribution and grouped matmul shapes; "
                "prioritize tiling, expert batching, and stable-shape grouped matmul paths."
            ),
            expected,
        )

    prefill_total = sum(ops[name]["total_us"] for name in PREFILL_TYPES if name in ops)
    all_total = sum(item["total_us"] for item in op_hotspots)
    if prefill_total > 0 and (phase in {"mixed", "prefill"} or prefill_total / max(all_total, 1.0) > 0.15):
        _add_opportunity(
            opportunities,
            "prefill_attention_or_kv",
            "Prefill",
            f"Prefill-heavy ops account for {prefill_total:.1f} us.",
            (
                "Compare attention, large matmul, RoPE, and KV-cache write kernels by shape; "
                "optimize long-prompt TTFT before applying decode-only fusion."
            ),
            "TTFT",
        )

    fusion_total = sum(ops[name]["total_us"] for name in FUSION_TYPES if name in ops)
    fusion_count = sum(ops[name]["count"] for name in FUSION_TYPES if name in ops)
    if fusion_total > 0 or fusion_count >= 10:
        _add_opportunity(
            opportunities,
            "fusion_candidate",
            "Prefill and Decode",
            f"Fusion candidate ops total {fusion_total:.1f} us across {fusion_count} calls.",
            (
                "Look for adjacent RMSNorm, residual add, bias, SwiGLU, cast, slice, and reshape chains; "
                "merge high-frequency short kernels when tensor shapes match."
            ),
            "TTFT and TPOT",
        )

    routing_total = sum(ops[name]["total_us"] for name in ROUTING_TYPES if name in ops)
    routing_count = sum(ops[name]["count"] for name in ROUTING_TYPES if name in ops)
    if routing_total > 0:
        _add_opportunity(
            opportunities,
            "moe_routing",
            "Decode",
            f"Routing ops total {routing_total:.1f} us across {routing_count} calls.",
            (
                "Profile top-k, routing init, token permute, and unpermute together; "
                "consider fusing routing post-processing with token reorder."
            ),
            "TPOT",
        )

    if kernel_summary.get("wait_ratio_percent", 0.0) >= 10.0:
        _add_opportunity(
            opportunities,
            "scheduler_wait",
            "Prefill and Decode",
            (
                f"Cumulative kernel wait/duration is {kernel_summary['wait_ratio_percent']:.1f}%. "
                "This can exceed 100% because Ascend wait time is summed across kernels and streams."
            ),
            (
                "Check stream dependencies, host launch gaps, hidden synchronizations, "
                "and whether small kernels serialize the decode loop."
            ),
            "TTFT and TPOT",
        )

    if kernel_summary.get("mte_ratio_percent", 0.0) >= 20.0:
        _add_opportunity(
            opportunities,
            "memory_movement",
            "Prefill and Decode",
            f"MTE time ratio is {kernel_summary['mte_ratio_percent']:.1f}%.",
            (
                "Inspect MTE2/MTE3-heavy kernels for memory-layout, format conversion, "
                "and avoidable data movement."
            ),
            "TTFT and TPOT",
        )

    for item in kernel_summary.get("by_type", []):
        if item["type"] == "GroupedMatmul" and item.get("avg_cube_utilization_percent", 0.0) and item[
                "avg_cube_utilization_percent"] < 70.0:
            _add_opportunity(
                opportunities,
                "low_cube_utilization",
                "Decode",
                f"GroupedMatmul average cube utilization is {item['avg_cube_utilization_percent']:.1f}%.",
                (
                    "Treat this as a shape or batching problem before adding more fusion; "
                    "small expert batches often underuse cube."
                ),
                "TPOT",
            )

    free_ratio = step_trace.get("Free Ratio(%)", 0.0)
    if free_ratio >= 20.0:
        _add_opportunity(
            opportunities,
            "service_or_batching_gap",
            "Mixed workload",
            f"Step trace free ratio is {free_ratio:.1f}%.",
            (
                "Correlate profiler windows with request arrival, scheduler batching, and benchmark concurrency; "
                "this may be a service-level gap rather than an operator gap."
            ),
            "TTFT and throughput",
        )

    return opportunities


def _signature_concentration(sew_moe: dict[str, Any]) -> tuple[float, int]:
    signatures = sew_moe.get("top_group_list_signatures") or []
    total = _int(sew_moe.get("grouped_signature_total_count"))
    if total <= 0:
        total = sum(_int(item.get("count")) for item in signatures)
    if total <= 0:
        return 0.0, 0
    top_count = _int(signatures[0].get("count"))
    return top_count / total * 100.0, total


def _build_compute_bucket_hint(sew_moe: dict[str, Any], *, max_buckets: int = 3) -> dict[str, Any]:
    signatures = sew_moe.get("top_group_list_signatures") or []
    total = _int(sew_moe.get("grouped_signature_total_count"))
    if total <= 0:
        total = sum(_int(item.get("count")) for item in signatures)
    if total <= 0:
        return {
            "coverage_percent": 0.0,
            "fallback_percent": 100.0,
            "top_signatures": [],
        }
    buckets = []
    covered = 0
    for item in signatures[:max_buckets]:
        count = _int(item.get("count"))
        covered += count
        buckets.append({
            "signature": str(item.get("signature", "")),
            "count": count,
            "coverage_percent": round(count / total * 100.0, 1),
        })
    coverage = covered / total * 100.0
    return {
        "coverage_percent": round(coverage, 1),
        "fallback_percent": round(100.0 - coverage, 1),
        "top_signatures": buckets,
    }


def _build_compute_bucket_plan(
    phase: str,
    compute_bucket_hint: dict[str, Any],
    *,
    total_grouped_records: int,
) -> dict[str, Any]:
    buckets = []
    for index, item in enumerate(compute_bucket_hint.get("top_signatures") or []):
        signature = str(item.get("signature", ""))
        active_expert_ids, compact_group_list, original_expert_count = _active_plan_from_signature(signature)
        buckets.append({
            "bucket_id": index,
            "signature": signature,
            "sample_count": _int(item.get("count")),
            "coverage_percent": _float(item.get("coverage_percent")),
            "active_expert_ids": list(active_expert_ids),
            "compact_group_list": list(compact_group_list),
            "original_expert_count": original_expert_count,
            "compact_expert_count": len(active_expert_ids),
        })
    return {
        "version": 1,
        "phase": phase,
        "mode": "trace_only",
        "selection": "top_grouped_signatures",
        "total_grouped_records": total_grouped_records,
        "coverage_percent": _float(compute_bucket_hint.get("coverage_percent")),
        "fallback_percent": _float(compute_bucket_hint.get("fallback_percent")),
        "gate": {
            "source": "grouped_dispatch",
            "requires_group_list_signature": True,
            "fallback": "existing_grouped_matmul_path",
        },
        "buckets": buckets,
    }


def _active_plan_from_signature(signature: str) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    prefix, separator, payload = signature.partition(":")
    if separator != ":" or prefix != "counts":
        return (), (), 0
    values = tuple(_int(value) for value in payload.split(",") if value.strip())
    return (
        tuple(index for index, count in enumerate(values) if count > 0),
        tuple(count for count in values if count > 0),
        len(values),
    )


def _ratio_for_ops(op_hotspots: list[dict[str, Any]], names: set[str]) -> float:
    total = sum(_float(item.get("total_us")) for item in op_hotspots)
    if total <= 0:
        return 0.0
    selected = sum(_float(item.get("total_us")) for item in op_hotspots if item.get("name") in names)
    return selected / total


def _build_p1_decision(
    phase: str,
    op_hotspots: list[dict[str, Any]],
    sew_moe: dict[str, Any],
    pipeline_profile: dict[str, Any],
) -> dict[str, Any]:
    if sew_moe.get("record_count", 0) <= 0:
        return {
            "target": "INSUFFICIENT_DATA",
            "reason": "Collect SEW-MoE active expert trace before choosing P1.",
        }

    ops = _op_by_name(op_hotspots)
    grouped_ratio = _float(ops.get("GroupedMatmul", {}).get("ratio_percent")) / 100.0
    routing_ratio = _ratio_for_ops(op_hotspots, ROUTING_TYPES)
    fusion_ratio = _ratio_for_ops(op_hotspots, FUSION_TYPES)
    signature_concentration, signature_samples = _signature_concentration(sew_moe)

    fractions = pipeline_profile.get("fractions") or {}
    t_frac = _float(fractions.get("t_frac"))
    r_frac = _float(fractions.get("r_frac"))
    c_frac = _float(fractions.get("c_frac"))
    m_frac = _float(fractions.get("m_frac"))
    rm_frac = r_frac + m_frac
    overlap_ratio = _float(pipeline_profile.get("overlap_potential_ratio"))

    fanout_stats = (sew_moe.get("fanout_by_source") or {}).get("grouped_dispatch") or {}
    mean_fanout = _float(fanout_stats.get("mean"))
    slot_hint = sew_moe.get("slot_budget_hint") or {}

    evidence = [
        f"GroupedMatmul ratio={grouped_ratio * 100.0:.1f}%",
        f"routing ratio={routing_ratio * 100.0:.1f}%",
        f"fusion-small-op ratio={fusion_ratio * 100.0:.1f}%",
        f"top grouped signature concentration={signature_concentration:.1f}% over {signature_samples} grouped records",
    ]
    if pipeline_profile.get("record_count", 0) > 0:
        evidence.append(
            f"pipeline fractions T/R/C/M={t_frac:.2f}/{r_frac:.2f}/{c_frac:.2f}/{m_frac:.2f}"
        )

    target = "P1-RM"
    reason = (
        "Routing/combine or shape instability is the safer next target before specializing grouped matmul."
    )
    recommendation = (
        "Fuse routing, dispatch metadata generation, and combine/unpermute work; keep current grouped matmul fallback."
    )

    if pipeline_profile.get("record_count", 0) > 0 and t_frac >= 0.65:
        target = "P1-T"
        reason = "Stage T transfer dominates the observed MoE pipeline."
        recommendation = (
            "Run trace-backed slot simulation and residency sweeps to reduce miss bytes before adding overlap."
        )
    elif (
        pipeline_profile.get("record_count", 0) > 0
        and 0.35 <= t_frac <= 0.60
        and overlap_ratio >= 0.70
        and mean_fanout >= 2.0
    ):
        target = "P1-H"
        reason = "Transfer and useful work are close enough that hit-first phasing may hide exposed T."
        recommendation = (
            "Prototype hit-first phased execution with one-phase fallback and measure exposed stall reduction."
        )
    elif (
        grouped_ratio >= 0.45
        and signature_concentration >= 60.0
        and (pipeline_profile.get("record_count", 0) <= 0 or c_frac >= max(t_frac, rm_frac, 0.45))
    ):
        target = "P1-C"
        reason = "Grouped compute dominates and grouped dispatch shapes are concentrated enough to specialize."
        recommendation = (
            "Add a decode-only grouped signature classifier and route dominant shapes to a stable GMM fast path."
        )

    decision = {
        "target": target,
        "reason": reason,
        "recommendation": recommendation,
        "signature_concentration_percent": round(signature_concentration, 1),
        "signature_sample_count": signature_samples,
        "grouped_matmul_ratio_percent": round(grouped_ratio * 100.0, 1),
        "routing_ratio_percent": round(routing_ratio * 100.0, 1),
        "fusion_ratio_percent": round(fusion_ratio * 100.0, 1),
        "mean_grouped_fanout": round(mean_fanout, 2),
        "overlap_potential_ratio": round(overlap_ratio, 4),
        "evidence": evidence,
        "phase": phase,
    }
    if target == "P1-T" and _int(slot_hint.get("min_slots_per_layer")) > 0:
        start_slots = _int(slot_hint.get("min_slots_per_layer"))
        stop_slots = max(start_slots * 8, 64)
        step_slots = max(start_slots, 1)
        trace_path = sew_moe.get("trace_path", "<moe_offload_trace.jsonl>")
        decision["slot_sweep_hint"] = {
            "start_slots": start_slots,
            "stop_slots": stop_slots,
            "step_slots": step_slots,
            "command": (
                "python tools/sew_offload/simulate_expert_slots.py "
                f"--trace {trace_path} --slot-range {start_slots}:{stop_slots}:{step_slots} "
                "--policy lru --output artifacts/sew_offload/sim/slot_sweep_lru.json"
            ),
        }
    if target == "P1-C":
        compute_bucket_hint = _build_compute_bucket_hint(sew_moe)
        decision["compute_bucket_hint"] = compute_bucket_hint
        decision["compute_bucket_plan"] = _build_compute_bucket_plan(
            phase,
            compute_bucket_hint,
            total_grouped_records=_int(sew_moe.get("grouped_signature_total_count")),
        )
    return decision


def analyze_profile(
    phase: str,
    profiler_output: str | Path,
    benchmark_path: str | Path | None = None,
    sew_moe_trace_path: str | Path | None = None,
    pipeline_profile_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = _resolve_output_dir(profiler_output)
    op_rows = _read_csv(output_dir / "op_statistic.csv")
    kernel_rows = _read_csv(output_dir / "kernel_details.csv")
    operator_rows = _read_csv(output_dir / "operator_details.csv")
    step_rows = _read_csv(output_dir / "step_trace_time.csv")

    op_hotspots = _summarize_ops(op_rows)
    kernel_summary = _summarize_kernels(kernel_rows)
    operator_hotspots = _summarize_operators(operator_rows)
    step_trace = _summarize_step_trace(step_rows)
    opportunities = _build_opportunities(phase, op_hotspots, kernel_summary, step_trace)
    sew_moe_trace = Path(sew_moe_trace_path) if sew_moe_trace_path else _find_sew_moe_trace(output_dir)
    pipeline_path = Path(pipeline_profile_path) if pipeline_profile_path else _find_pipeline_profile(output_dir)
    sew_moe = _summarize_sew_moe_trace(sew_moe_trace)
    pipeline_profile = _summarize_pipeline_profile(pipeline_path)
    p1_decision = _build_p1_decision(phase, op_hotspots, sew_moe, pipeline_profile)

    return {
        "phase": phase,
        "profiler_output": str(output_dir),
        "benchmark": _load_benchmark(benchmark_path),
        "op_hotspots": op_hotspots[:20],
        "operator_hotspots": operator_hotspots,
        "kernel_summary": kernel_summary,
        "step_trace": step_trace,
        "sew_moe": sew_moe,
        "pipeline_profile": pipeline_profile,
        "p1_decision": p1_decision,
        "optimization_opportunities": opportunities,
    }


def _fmt_us(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} ms"
    return f"{value:.1f} us"


def _render_phase(report: dict[str, Any]) -> str:
    phase = report["phase"]
    focus = {
        "mixed": "Macro focus: end-to-end TTFT, TPOT, throughput, and phase balance.",
        "prefill": "TTFT focus: long-prompt attention, large matmul, RoPE, and KV-cache writes.",
        "decode": "TPOT focus: MoE routing, grouped matmul, short kernels, and stream wait.",
    }.get(phase, "Stage focus: inspect profiler hotspots.")

    lines = [f"## {phase}", "", focus, ""]
    benchmark = report.get("benchmark") or {}
    if benchmark:
        metrics = []
        for key in ("median_ttft_ms", "median_tpot_ms", "output_throughput", "total_token_throughput"):
            if key in benchmark:
                metrics.append(f"{key}={benchmark[key]}")
        if metrics:
            lines.extend(["Benchmark: " + ", ".join(metrics), ""])

    lines.extend(["Top OP types:", ""])
    lines.extend(["| OP Type | Count | Total | Avg | Ratio |", "|---|---:|---:|---:|---:|"])
    for item in report["op_hotspots"][:10]:
        lines.append(
            f"| {item['name']} | {item['count']} | {_fmt_us(item['total_us'])} | "
            f"{_fmt_us(item['avg_us'])} | {item['ratio_percent']:.1f}% |")

    lines.extend(["", "Top kernels:", ""])
    lines.extend(["| Kernel Type | Count | Total | Avg | Wait | Cube |", "|---|---:|---:|---:|---:|---:|"])
    for item in report["kernel_summary"]["by_type"][:10]:
        lines.append(
            f"| {item['type']} | {item['count']} | {_fmt_us(item['duration_us'])} | "
            f"{_fmt_us(item['avg_us'])} | {_fmt_us(item['wait_us'])} | "
            f"{item['avg_cube_utilization_percent']:.1f}% |")

    lines.extend(["", "SEW-MoE active expert trace:", ""])
    sew_moe = report.get("sew_moe") or {}
    if sew_moe.get("record_count", 0) <= 0:
        lines.append(f"- {sew_moe.get('note', 'No SEW-MoE active expert trace found.')}")
    else:
        lines.append(f"- Records: {sew_moe['record_count']} ({sew_moe.get('trace_path', '')})")
        fanout_by_source = sew_moe.get("fanout_by_source", {})
        for source, stats in fanout_by_source.items():
            lines.append(
                f"- {source}: count={stats['count']}, "
                f"fanout min/mean/max={stats['min']}/{stats['mean']:.1f}/{stats['max']}"
            )
        signatures = sew_moe.get("top_group_list_signatures", [])
        if signatures:
            rendered_signatures = ", ".join(
                f"{item['signature']} x{item['count']}" for item in signatures[:5]
            )
            lines.append(f"- Common grouped signatures: {rendered_signatures}")
        slot_hint = sew_moe.get("slot_budget_hint") or {}
        if slot_hint:
            lines.append(
                "- Minimum per-layer slots: "
                f"{slot_hint.get('min_slots_per_layer', 0)} "
                f"(max grouped fanout={slot_hint.get('max_grouped_fanout', 0)}, "
                f"mean={_float(slot_hint.get('mean_grouped_fanout')):.2f})"
            )

    pipeline_profile = report.get("pipeline_profile") or {}
    if pipeline_profile.get("record_count", 0) > 0:
        fractions = pipeline_profile.get("fractions") or {}
        lines.extend(["", "SEW-MoE pipeline profile:", ""])
        lines.append(
            "- Mean T/R/C/M fractions: "
            f"{_float(fractions.get('t_frac')):.2f}/"
            f"{_float(fractions.get('r_frac')):.2f}/"
            f"{_float(fractions.get('c_frac')):.2f}/"
            f"{_float(fractions.get('m_frac')):.2f}"
        )
        lines.append(
            f"- Overlap potential ratio: {_float(pipeline_profile.get('overlap_potential_ratio')):.2f}"
        )
        gate_summary = pipeline_profile.get("compute_bucket_fast_path_gate") or {}
        if gate_summary:
            lines.append(
                "- Compute bucket gate: "
                f"enabled={_float(gate_summary.get('enabled_percent')):.1f}%, "
                f"experts {_float(gate_summary.get('mean_original_expert_count')):.1f} -> "
                f"{_float(gate_summary.get('mean_compact_expert_count')):.1f}, "
                f"ratio={_float(gate_summary.get('mean_compaction_ratio')):.2f}"
            )
            fallback_reasons = gate_summary.get("top_fallback_reasons") or []
            if fallback_reasons:
                rendered_reasons = ", ".join(
                    f"{item['reason']} x{item['count']}" for item in fallback_reasons[:3]
                )
                lines.append(f"- Compute bucket fallback reasons: {rendered_reasons}")

    p1_decision = report.get("p1_decision") or {}
    if p1_decision:
        lines.extend(["", "SEW-MoE P1 decision:", ""])
        lines.append(f"- Recommended P1 target: {p1_decision.get('target', 'UNKNOWN')}")
        lines.append(f"- Reason: {p1_decision.get('reason', '')}")
        if p1_decision.get("target") != "INSUFFICIENT_DATA":
            lines.append(f"- Recommendation: {p1_decision.get('recommendation', '')}")
            lines.append(
                "- Evidence: "
                + "; ".join(str(item) for item in p1_decision.get("evidence", []))
            )
            slot_sweep_hint = p1_decision.get("slot_sweep_hint") or {}
            if slot_sweep_hint:
                lines.append(
                    "- Slot sweep: "
                    + str(slot_sweep_hint.get("command", ""))
                )
            compute_bucket_hint = p1_decision.get("compute_bucket_hint") or {}
            if compute_bucket_hint:
                buckets = ", ".join(
                    f"{item['signature']} x{item['count']}"
                    for item in compute_bucket_hint.get("top_signatures", [])
                )
                lines.append(
                    "- Stable grouped buckets: "
                    f"coverage={_float(compute_bucket_hint.get('coverage_percent')):.1f}%, "
                    f"fallback={_float(compute_bucket_hint.get('fallback_percent')):.1f}%"
                    + (f" ({buckets})" if buckets else "")
                )
            compute_bucket_plan = p1_decision.get("compute_bucket_plan") or {}
            if compute_bucket_plan:
                lines.append(
                    "- Compute bucket plan: "
                    f"{len(compute_bucket_plan.get('buckets', []))} buckets, "
                    f"fallback={_float(compute_bucket_plan.get('fallback_percent')):.1f}%"
                )

    lines.extend(["", "Optimization opportunities:", ""])
    if report["optimization_opportunities"]:
        for item in report["optimization_opportunities"]:
            lines.append(
                f"- {item['category']} ({item['stage']}, {item['expected_metric']}): "
                f"{item['evidence']} {item['recommendation']}")
    else:
        lines.append("- No heuristic opportunities found; inspect trace_view.json manually.")
    lines.append("")
    return "\n".join(lines)


def render_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Ascend MoE Profile Report",
        "",
        "This report is built from Ascend PyTorch Profiler CSV files. Use the mixed phase as the macro view, "
        "then use the prefill and decode phases to separate TTFT and TPOT optimization work.",
        "",
    ]
    for report in reports:
        lines.append(_render_phase(report))
    return "\n".join(lines)


def _parse_phase_arg(value: str) -> tuple[str, Path, Path | None, Path | None, Path | None]:
    parts = value.split(":", 4)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "phase spec must be phase:profiler_output[:benchmark_json[:sew_trace_jsonl[:sew_profile_jsonl]]]"
        )
    phase = parts[0]
    output = Path(parts[1])
    benchmark = Path(parts[2]) if len(parts) == 3 and parts[2] else None
    if len(parts) >= 4:
        benchmark = Path(parts[2]) if parts[2] else None
    sew_trace = Path(parts[3]) if len(parts) >= 4 and parts[3] else None
    sew_profile = Path(parts[4]) if len(parts) >= 5 and parts[4] else None
    return phase, output, benchmark, sew_trace, sew_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Ascend PyTorch Profiler output for Qwen3 MoE prefill/decode optimization.")
    parser.add_argument(
        "--phase",
        action="append",
        type=_parse_phase_arg,
        required=True,
        help="Phase input as phase:ASCEND_PROFILER_OUTPUT[:benchmark_json]. Can be used multiple times.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path for machine-readable report JSON.")
    parser.add_argument("--markdown-output", type=Path, help="Optional path for Markdown report.")
    args = parser.parse_args()

    reports = [
        analyze_profile(phase, output, benchmark, sew_trace, sew_profile)
        for phase, output, benchmark, sew_trace, sew_profile in args.phase
    ]
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    markdown = render_markdown(reports)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
