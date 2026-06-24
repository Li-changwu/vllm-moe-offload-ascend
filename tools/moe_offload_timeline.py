# SPDX-License-Identifier: Apache-2.0
"""Render SEW-MoE offload timing JSONL into summaries and timeline views."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PIPELINE_STAGE_LABELS = {
    "stage_t_ms": "T offload plan/load",
    "stage_r_ms": "R token dispatch",
    "stage_c_ms": "C expert MLP",
    "stage_m_ms": "M token combine",
}


PIPELINE_DETAIL_STAGE_LABELS = {
    "r_log2phy_map": "R logical->physical ids",
    "r_build_dispatch_input": "R build dispatch input",
    "r_init_routing": "R init routing op",
    "r_expert_tokens_cast": "R expert tokens cast",
    "r_token_dispatch_total": "R token dispatch wrapper",
    "r_residual_wait": "R residual/wait",
    "t_residual_wait": "T residual/wait",
}


OFFLOAD_STAGE_LABELS = {
    "active_expert_normalize": "active expert set",
    "slot_cache_lookup": "slot cache lookup",
    "slot_allocate": "slot allocate/evict",
    "host_bundle_lookup": "host bundle lookup",
    "expert_h2d_load_sync": "expert H2D load",
    "slot_mapping_build": "logical->physical map",
    "prepared_slot_weights": "slot weight view",
    "prepare_fixed_slot_plan": "fixed slot prepare",
}


@dataclass(frozen=True)
class StageDuration:
    name: str
    duration_ms: float
    source: str
    count: int = 1


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def summarize_profile(records: list[dict[str, Any]], *, step_id: int | None = None) -> dict[str, Any]:
    pipeline_records = [
        record for record in records
        if record.get("event") == "moe_pipeline_timing"
        and (step_id is None or _int(record.get("step_id")) == step_id)
    ]
    timeline_records = [
        record for record in records
        if record.get("event") == "moe_offload_timeline"
        and (step_id is None or _int(record.get("step_id")) == step_id)
    ]
    pipeline_detail_records = [
        record for record in records
        if record.get("event") == "moe_pipeline_detail_timing"
        and (step_id is None or _int(record.get("step_id")) == step_id)
    ]

    return {
        "record_counts": {
            "pipeline": len(pipeline_records),
            "offload_timeline": len(timeline_records),
            "pipeline_detail": len(pipeline_detail_records),
        },
        "pipeline": _summarize_pipeline(pipeline_records),
        "pipeline_detail": _summarize_pipeline_detail(pipeline_detail_records),
        "offload_timeline": _summarize_timeline(timeline_records),
        "representative_step_id": _select_representative_step_id(pipeline_records, timeline_records),
    }


def build_stage_durations(
    records: list[dict[str, Any]],
    *,
    step_id: int | None = None,
    include_offload_detail: bool = True,
) -> list[StageDuration]:
    selected_step_id = _select_step_id(records, step_id)
    stages: list[StageDuration] = []
    pipeline = [
        record for record in records
        if record.get("event") == "moe_pipeline_timing"
        and _int(record.get("step_id")) == selected_step_id
    ]
    if pipeline:
        record = pipeline[0]
        for key, label in PIPELINE_STAGE_LABELS.items():
            stages.append(StageDuration(label, _float(record.get(key)), "pipeline"))

    if not include_offload_detail:
        return stages

    timeline_records = [
        record for record in records
        if record.get("event") == "moe_offload_timeline"
        and _int(record.get("step_id")) == selected_step_id
    ]
    timeline_records.sort(key=lambda record: (_int(record.get("start_ns")), _int(record.get("end_ns"))))
    for record in timeline_records:
        name = str(record.get("name") or "unknown")
        label = OFFLOAD_STAGE_LABELS.get(name, name)
        payload = record.get("payload") or {}
        suffix = _timeline_suffix(name, payload)
        stages.append(
            StageDuration(
                f"{label}{suffix}",
                _timeline_duration_ms(record),
                "offload_detail",
            )
        )
    pipeline_detail_records = [
        record for record in records
        if record.get("event") == "moe_pipeline_detail_timing"
        and _int(record.get("step_id")) == selected_step_id
    ]
    for record in pipeline_detail_records:
        name = str(record.get("name") or "unknown")
        stages.append(
            StageDuration(
                PIPELINE_DETAIL_STAGE_LABELS.get(name, name),
                _float(record.get("duration_ms")),
                "pipeline_detail",
            )
        )
    return stages


def render_markdown(records: list[dict[str, Any]], *, step_id: int | None = None) -> str:
    selected_step_id = _select_step_id(records, step_id)
    summary = summarize_profile(records, step_id=selected_step_id)
    stages = build_stage_durations(records, step_id=selected_step_id)
    pipeline_stage_rows = summary["pipeline"].get("stages", [])
    pipeline_detail_rows = summary["pipeline_detail"].get("stages", [])
    offload_stage_rows = summary["offload_timeline"].get("stages", [])

    lines = [
        "# MoE Offload Inference Timeline",
        "",
        f"Representative step: `{selected_step_id}`",
        "",
        "```mermaid",
        "gantt",
        "    title SEW-MoE Offload 推理阶段耗时",
        "    dateFormat X",
        "    axisFormat %Lms",
    ]
    cursors_ms: dict[str, float] = defaultdict(float)
    current_section = None
    for index, stage in enumerate(stages):
        if stage.source != current_section:
            current_section = stage.source
            lines.append(f"    section {_section_title(stage.source)}")
        start_ms = cursors_ms[stage.source]
        duration_ms = max(stage.duration_ms, 0.001)
        lines.append(
            f"    {_escape_mermaid(stage.name)} :s{index}, {start_ms / 1000.0:.6f}, {duration_ms / 1000.0:.6f}"
        )
        cursors_ms[stage.source] += duration_ms
    lines.extend(["```", ""])

    lines.extend(_stage_table("Pipeline T/R/C/M Mean", pipeline_stage_rows))
    lines.append("")
    lines.extend(_stage_table("Pipeline Detail Total", pipeline_detail_rows))
    lines.append("")
    lines.extend(_stage_table("Offload Detail Total", offload_stage_rows))
    cache = summary["offload_timeline"].get("cache", {})
    if cache:
        lines.extend([
            "",
            "## Cache",
            "",
            f"- hits: `{cache.get('hits', 0)}`",
            f"- misses: `{cache.get('misses', 0)}`",
            f"- hit_rate: `{cache.get('hit_rate', 0.0)}`",
        ])
    return "\n".join(lines) + "\n"


def render_chrome_trace(records: list[dict[str, Any]], *, step_id: int | None = None) -> dict[str, Any]:
    selected_step_id = _select_step_id(records, step_id)
    trace_events: list[dict[str, Any]] = []
    pid = 910
    offload_tid = 1
    pipeline_tid = 2
    pipeline_detail_tid = 3

    trace_events.append({"ph": "M", "name": "process_name", "pid": pid, "args": {"name": "SEW-MoE"}})
    trace_events.append(
        {"ph": "M", "name": "thread_name", "pid": pid, "tid": offload_tid, "args": {"name": "offload detail"}}
    )
    trace_events.append(
        {"ph": "M", "name": "thread_name", "pid": pid, "tid": pipeline_tid, "args": {"name": "pipeline T/R/C/M"}}
    )
    trace_events.append(
        {
            "ph": "M",
            "name": "thread_name",
            "pid": pid,
            "tid": pipeline_detail_tid,
            "args": {"name": "pipeline detail"},
        }
    )

    timeline_records = [
        record for record in records
        if record.get("event") == "moe_offload_timeline"
        and _int(record.get("step_id")) == selected_step_id
    ]
    base_ns = min((_int(record.get("start_ns")) for record in timeline_records), default=0)
    for record in timeline_records:
        start_ns = _int(record.get("start_ns"))
        end_ns = _int(record.get("end_ns"))
        payload = record.get("payload") or {}
        trace_events.append({
            "ph": "X",
            "cat": "moe_offload",
            "name": str(record.get("name") or "unknown"),
            "pid": pid,
            "tid": offload_tid,
            "ts": _ns_to_us(start_ns - base_ns),
            "dur": _ns_to_us(max(0, end_ns - start_ns)),
            "args": {
                "layer_id": record.get("layer_id"),
                "step_id": selected_step_id,
                **payload,
            },
        })

    pipeline_records = [
        record for record in records
        if record.get("event") == "moe_pipeline_timing"
        and _int(record.get("step_id")) == selected_step_id
    ]
    if pipeline_records:
        cursor_us = 0.0
        record = pipeline_records[0]
        for key, label in PIPELINE_STAGE_LABELS.items():
            duration_us = _float(record.get(key)) * 1000.0
            trace_events.append({
                "ph": "X",
                "cat": "moe_pipeline",
                "name": label,
                "pid": pid,
                "tid": pipeline_tid,
                "ts": cursor_us,
                "dur": max(duration_us, 0.001),
                "args": {
                    "layer_id": record.get("layer_id"),
                    "step_id": selected_step_id,
                    "stage_key": key,
                },
            })
            cursor_us += max(duration_us, 0.001)

    pipeline_detail_records = [
        record for record in records
        if record.get("event") == "moe_pipeline_detail_timing"
        and _int(record.get("step_id")) == selected_step_id
    ]
    cursor_us = 0.0
    for record in pipeline_detail_records:
        duration_us = _float(record.get("duration_ms")) * 1000.0
        name = str(record.get("name") or "unknown")
        trace_events.append({
            "ph": "X",
            "cat": "moe_pipeline_detail",
            "name": PIPELINE_DETAIL_STAGE_LABELS.get(name, name),
            "pid": pid,
            "tid": pipeline_detail_tid,
            "ts": cursor_us,
            "dur": max(duration_us, 0.001),
            "args": {
                "layer_id": record.get("layer_id"),
                "step_id": selected_step_id,
                "stage_key": name,
                **(record.get("payload") or {}),
            },
        })
        cursor_us += max(duration_us, 0.001)

    return {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
        "metadata": {
            "step_id": selected_step_id,
            "source": "vllm_ascend.tools.sew_offload.moe_offload_timeline",
        },
    }


def render_svg(records: list[dict[str, Any]], *, step_id: int | None = None) -> str:
    """Render a dependency-free SVG timing overview.

    The offload wrapper event is drawn as a separate bar because it contains the
    child timing events; stacking it with the children would double count time.
    """

    summary = summarize_profile(records, step_id=step_id)
    pipeline_rows = summary["pipeline"].get("stages", [])
    pipeline_detail_rows = [
        row
        for row in _ordered_pipeline_detail_rows(summary["pipeline_detail"].get("stages", []))
        if row.get("key") != "r_token_dispatch_total"
    ]
    offload_rows = _ordered_offload_rows(summary["offload_timeline"].get("stages", []))
    prepare_rows = [row for row in offload_rows if row.get("key") == "prepare_fixed_slot_plan"]
    offload_child_rows = [row for row in offload_rows if row.get("key") != "prepare_fixed_slot_plan"]

    bars: list[dict[str, Any]] = []
    if pipeline_rows:
        bars.append(
            {
                "title": "Pipeline T/R/C/M",
                "note": "Measured pipeline stages across the selected MoE invocations.",
                "segments": _rows_to_svg_segments(pipeline_rows, source="pipeline"),
            }
        )
    if offload_child_rows:
        bars.append(
            {
                "title": "Offload Detail",
                "note": "Child events only; wrapper event is shown separately.",
                "segments": _rows_to_svg_segments(offload_child_rows, source="offload_detail"),
            }
        )
    if pipeline_detail_rows:
        bars.append(
            {
                "title": "Pipeline Detail",
                "note": "Fine-grained R-stage records from the selected MoE invocations.",
                "segments": _rows_to_svg_segments(pipeline_detail_rows, source="pipeline_detail"),
            }
        )
    if prepare_rows:
        bars.append(
            {
                "title": "Fixed Slot Prepare",
                "note": "Measured prepare_fixed_slot_plan wrapper.",
                "segments": _rows_to_svg_segments(prepare_rows, source="offload_wrapper"),
            }
        )

    width = 1200
    margin_left = 245
    margin_right = 145
    chart_width = width - margin_left - margin_right
    top = 130
    bar_height = 30
    row_gap = 82
    legend_top = top + max(1, len(bars)) * row_gap + 20
    legend_rows = sum(len(bar["segments"]) for bar in bars)
    height = legend_top + max(1, legend_rows) * 23 + 40
    max_total_ms = max((_segments_total_ms(bar["segments"]) for bar in bars), default=1.0)
    max_total_ms = max(max_total_ms, 1.0)
    scope = "all profiled MoE invocations" if step_id is None else f"step {int(step_id)}"
    cache = summary["offload_timeline"].get("cache", {})
    h2d_bytes = _int(summary["offload_timeline"].get("h2d_bytes"))
    subtitle = (
        f"scope: {scope}; pipeline records: {summary['record_counts']['pipeline']}; "
        f"offload events: {summary['record_counts']['offload_timeline']}; "
        f"H2D: {_human_bytes(h2d_bytes)}; cache hit rate: {_float(cache.get('hit_rate')):.2%}"
    )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        ".bg{fill:#fbfbf8}.title{font:700 26px sans-serif;fill:#1f2328}",
        ".sub{font:14px sans-serif;fill:#59636e}.axis{stroke:#c9d1d9;stroke-width:1}",
        ".label{font:700 15px sans-serif;fill:#24292f}.note{font:12px sans-serif;fill:#6e7781}",
        ".segtext{font:700 12px sans-serif;fill:#ffffff}.small{font:12px sans-serif;fill:#24292f}",
        ".total{font:700 13px sans-serif;fill:#24292f}.legend{font:12px sans-serif;fill:#24292f}",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
        '<text class="title" x="40" y="48">SEW-MoE Offload Timing Overview</text>',
        f'<text class="sub" x="40" y="76">{_svg_escape(subtitle)}</text>',
    ]

    for tick in range(5):
        x = margin_left + chart_width * tick / 4
        value = max_total_ms * tick / 4
        lines.append(f'<line class="axis" x1="{x:.1f}" y1="96" x2="{x:.1f}" y2="{legend_top - 25}"/>')
        lines.append(
            f'<text class="note" x="{x:.1f}" y="112" text-anchor="middle">{value:.0f} ms</text>'
        )

    for row_index, bar in enumerate(bars):
        y = top + row_index * row_gap
        lines.append(f'<text class="label" x="40" y="{y + 20}">{_svg_escape(bar["title"])}</text>')
        lines.append(f'<text class="note" x="40" y="{y + 40}">{_svg_escape(bar["note"])}</text>')
        lines.append(
            f'<rect x="{margin_left}" y="{y}" width="{chart_width}" height="{bar_height}" '
            'rx="4" fill="#eef1f4"/>'
        )
        cursor = margin_left
        for segment in bar["segments"]:
            duration_ms = _float(segment["duration_ms"])
            seg_width = duration_ms / max_total_ms * chart_width
            seg_width = max(seg_width, 0.0)
            title = f'{segment["label"]}: {duration_ms:.4f} ms'
            lines.append(
                f'<rect x="{cursor:.2f}" y="{y}" width="{seg_width:.2f}" height="{bar_height}" '
                f'rx="3" fill="{segment["color"]}"><title>{_svg_escape(title)}</title></rect>'
            )
            if seg_width >= 92:
                lines.append(
                    f'<text class="segtext" x="{cursor + 8:.2f}" y="{y + 20}">'
                    f'{_svg_escape(_short_label(segment["label"]))}</text>'
                )
            cursor += seg_width
        total_ms = _segments_total_ms(bar["segments"])
        lines.append(
            f'<text class="total" x="{margin_left + chart_width + 12}" y="{y + 20}">'
            f'{total_ms:.1f} ms</text>'
        )

    lines.append(f'<text class="label" x="40" y="{legend_top}">Stage Totals</text>')
    legend_y = legend_top + 24
    for bar in bars:
        for segment in bar["segments"]:
            lines.append(
                f'<rect x="40" y="{legend_y - 10}" width="12" height="12" rx="2" '
                f'fill="{segment["color"]}"/>'
            )
            lines.append(
                f'<text class="legend" x="60" y="{legend_y}">'
                f'{_svg_escape(bar["title"])} / {_svg_escape(segment["label"])}: '
                f'{_float(segment["duration_ms"]):.4f} ms</text>'
            )
            legend_y += 23
    if not bars:
        lines.append('<text class="small" x="40" y="145">No MoE timing records found.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_decode_layer_svg(
    records: list[dict[str, Any]],
    *,
    wave_index: int | None = None,
    layers_per_wave: int = 48,
) -> str:
    """Render one decode wave as per-layer T/R/C/M pipeline rows."""

    selected_wave = _select_decode_wave(records, wave_index=wave_index, layers_per_wave=layers_per_wave)
    rows = _decode_layer_rows(records, wave_index=selected_wave, layers_per_wave=layers_per_wave)
    width = 1500
    margin_left = 150
    margin_right = 250
    chart_width = width - margin_left - margin_right
    top = 135
    row_gap = 36
    pipeline_height = 10
    detail_height = 8
    r_detail_height = 8
    legend_top = top + max(1, len(rows)) * row_gap + 38
    height = legend_top + 130
    max_total_ms = max((row["pipeline_total_ms"] for row in rows), default=1.0)
    max_total_ms = max(max_total_ms, 1.0)
    step_range = _wave_step_range(rows)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        ".bg{fill:#fbfbf8}.title{font:700 26px sans-serif;fill:#1f2328}",
        ".sub{font:14px sans-serif;fill:#59636e}.axis{stroke:#d0d7de;stroke-width:1}",
        ".layer{font:700 12px sans-serif;fill:#24292f}.meta{font:11px sans-serif;fill:#59636e}",
        ".segtext{font:700 9px sans-serif;fill:#ffffff}.total{font:700 11px sans-serif;fill:#24292f}",
        ".legend{font:12px sans-serif;fill:#24292f}.small{font:11px sans-serif;fill:#59636e}",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
        f'<text class="title" x="40" y="48">Decode Wave Layer Pipeline</text>',
        f'<text class="sub" x="40" y="76">wave={selected_wave}, steps={_svg_escape(step_range)}, '
        f'layers={len(rows)}, scale=max layer total {_svg_escape(f"{max_total_ms:.3f} ms")}</text>',
        '<text class="small" x="40" y="103">'
        "Each row: pipeline T/R/C/M on top; T detail in the middle; "
        "R detail at the bottom when available.</text>",
    ]

    for tick in range(6):
        x = margin_left + chart_width * tick / 5
        value = max_total_ms * tick / 5
        lines.append(f'<line class="axis" x1="{x:.1f}" y1="115" x2="{x:.1f}" y2="{legend_top - 18}"/>')
        lines.append(
            f'<text class="small" x="{x:.1f}" y="128" text-anchor="middle">{value:.1f} ms</text>'
        )

    for index, row in enumerate(rows):
        y = top + index * row_gap
        layer_label = f'L{int(row["layer_id"]):02d}'
        meta = _decode_row_meta(row)
        lines.append(f'<text class="layer" x="40" y="{y + 10}">{_svg_escape(layer_label)}</text>')
        lines.append(f'<text class="meta" x="78" y="{y + 10}">{_svg_escape(meta)}</text>')
        lines.append(
            f'<rect x="{margin_left}" y="{y}" width="{chart_width}" height="{pipeline_height}" '
            'rx="3" fill="#eef1f4"/>'
        )
        _append_decode_segments(
            lines,
            row["pipeline_segments"],
            x=margin_left,
            y=y,
            height=pipeline_height,
            width=chart_width,
            scale_ms=max_total_ms,
            text_threshold=42,
        )
        if row["detail_segments"]:
            detail_y = y + pipeline_height + 3
            lines.append(
                f'<rect x="{margin_left}" y="{detail_y}" width="{chart_width}" height="{detail_height}" '
                'rx="3" fill="#f1f3f5"/>'
            )
            _append_decode_segments(
                lines,
                row["detail_segments"],
                x=margin_left,
                y=detail_y,
                height=detail_height,
                width=chart_width,
                scale_ms=max_total_ms,
                text_threshold=55,
            )
        if row["r_detail_segments"]:
            r_detail_y = y + pipeline_height + detail_height + 6
            lines.append(
                f'<rect x="{margin_left}" y="{r_detail_y}" width="{chart_width}" height="{r_detail_height}" '
                'rx="3" fill="#f6f8fa"/>'
            )
            _append_decode_segments(
                lines,
                row["r_detail_segments"],
                x=margin_left,
                y=r_detail_y,
                height=r_detail_height,
                width=chart_width,
                scale_ms=max_total_ms,
                text_threshold=55,
                source="pipeline_detail",
            )
        total_x = margin_left + chart_width + 12
        lines.append(
            f'<text class="total" x="{total_x}" y="{y + 10}">'
            f'{_float(row["pipeline_total_ms"]):.3f} ms</text>'
        )
        if row["detail_total_ms"]:
            lines.append(
                f'<text class="meta" x="{total_x + 78}" y="{y + 10}">'
                f'T-detail {_float(row["detail_total_ms"]):.3f} ms</text>'
            )
        if row["r_detail_total_ms"]:
            lines.append(
                f'<text class="meta" x="{total_x + 78}" y="{y + 24}">'
                f'R-detail {_float(row["r_detail_total_ms"]):.3f} ms</text>'
            )

    legend_items = [
        ("stage_t_ms", "T offload plan/load", "pipeline"),
        ("stage_r_ms", "R token dispatch", "pipeline"),
        ("stage_c_ms", "C expert MLP", "pipeline"),
        ("stage_m_ms", "M token combine", "pipeline"),
        ("expert_h2d_load_sync", "H2D load detail", "offload_detail"),
        ("slot_mapping_build", "slot map detail", "offload_detail"),
        ("slot_cache_lookup", "cache lookup detail", "offload_detail"),
        ("t_residual_wait", "T residual/wait", "pipeline_detail"),
        ("r_log2phy_map", "R log2phy map", "pipeline_detail"),
        ("r_init_routing", "R init routing", "pipeline_detail"),
        ("r_expert_tokens_cast", "R token count cast", "pipeline_detail"),
        ("r_residual_wait", "R residual/wait", "pipeline_detail"),
    ]
    lines.append(f'<text class="layer" x="40" y="{legend_top}">Legend</text>')
    for index, (key, label, source) in enumerate(legend_items):
        x = 40 + (index % 3) * 430
        y = legend_top + 24 + (index // 3) * 28
        color = _stage_color(key, source)
        lines.append(f'<rect x="{x}" y="{y - 11}" width="13" height="13" rx="2" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{x + 21}" y="{y}">{_svg_escape(label)}</text>')
    if not rows:
        lines.append('<text class="small" x="40" y="150">No decode wave records found.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _summarize_pipeline(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"record_count": 0, "stages": []}
    rows = []
    for key, label in PIPELINE_STAGE_LABELS.items():
        values = [_float(record.get(key)) for record in records]
        rows.append({
            "name": label,
            "key": key,
            **_duration_stats_ms(values),
        })
    totals = [
        sum(_float(record.get(key)) for key in PIPELINE_STAGE_LABELS)
        for record in records
    ]
    return {
        "record_count": len(records),
        "stages": rows,
        "total_ms": _duration_stats_ms(totals),
    }


def _summarize_timeline(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"record_count": 0, "stages": []}
    by_name: dict[str, list[float]] = defaultdict(list)
    cache_hits = 0
    cache_misses = 0
    h2d_bytes = 0
    for record in records:
        name = str(record.get("name") or "unknown")
        by_name[name].append(_timeline_duration_ms(record))
        payload = record.get("payload") or {}
        if name == "slot_cache_lookup":
            if bool(payload.get("cache_hit")):
                cache_hits += 1
            else:
                cache_misses += 1
        if name == "expert_h2d_load_sync":
            h2d_bytes += _int(payload.get("bytes"))

    rows = []
    for name, values in by_name.items():
        rows.append({
            "name": OFFLOAD_STAGE_LABELS.get(name, name),
            "key": name,
            **_duration_stats_ms(values),
        })
    rows.sort(key=lambda item: item["total_ms"], reverse=True)
    lookups = cache_hits + cache_misses
    return {
        "record_count": len(records),
        "stages": rows,
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": round(cache_hits / lookups, 4) if lookups else 0.0,
        },
        "h2d_bytes": h2d_bytes,
    }


def _summarize_pipeline_detail(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"record_count": 0, "stages": []}
    by_name: dict[str, list[float]] = defaultdict(list)
    for record in records:
        name = str(record.get("name") or "unknown")
        by_name[name].append(_float(record.get("duration_ms")))

    rows = []
    for name, values in by_name.items():
        rows.append({
            "name": PIPELINE_DETAIL_STAGE_LABELS.get(name, name),
            "key": name,
            **_duration_stats_ms(values),
        })
    rows.sort(key=lambda item: item["total_ms"], reverse=True)
    return {
        "record_count": len(records),
        "stages": rows,
    }


def _select_representative_step_id(
    pipeline_records: list[dict[str, Any]],
    timeline_records: list[dict[str, Any]],
) -> int:
    if timeline_records:
        totals: dict[int, float] = defaultdict(float)
        for record in timeline_records:
            totals[_int(record.get("step_id"))] += _timeline_duration_ms(record)
        return max(totals.items(), key=lambda item: (item[1], -item[0]))[0]
    if pipeline_records:
        return _int(pipeline_records[0].get("step_id"))
    return -1


def _select_step_id(records: list[dict[str, Any]], step_id: int | None) -> int:
    if step_id is not None:
        return int(step_id)
    summary = summarize_profile(records)
    return int(summary["representative_step_id"])


def _duration_stats_ms(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "total_ms": round(sum(values), 4),
        "mean_ms": round(statistics.fmean(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "max_ms": round(max(values), 4),
    }


def _stage_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| stage | count | total ms | mean ms | max ms |", "|---|---:|---:|---:|---:|"]
    if not rows:
        lines.append("| none | 0 | 0 | 0 | 0 |")
        return lines
    for row in rows:
        lines.append(
            f"| {row.get('name', '')} | {row.get('count', 0)} | "
            f"{_float(row.get('total_ms')):.4f} | {_float(row.get('mean_ms')):.4f} | "
            f"{_float(row.get('max_ms')):.4f} |"
        )
    return lines


def _ordered_offload_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(OFFLOAD_STAGE_LABELS)}
    return sorted(
        rows,
        key=lambda row: (order.get(str(row.get("key")), len(order)), str(row.get("name") or "")),
    )


def _ordered_pipeline_detail_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(PIPELINE_DETAIL_STAGE_LABELS)}
    return sorted(
        rows,
        key=lambda row: (order.get(str(row.get("key")), len(order)), str(row.get("name") or "")),
    )


def _select_decode_wave(
    records: list[dict[str, Any]],
    *,
    wave_index: int | None,
    layers_per_wave: int,
) -> int:
    if wave_index is not None:
        return int(wave_index)
    pipeline_records = [
        record for record in records
        if record.get("event") == "moe_pipeline_timing"
    ]
    complete_waves: list[int] = []
    by_wave: dict[int, set[int]] = defaultdict(set)
    for record in pipeline_records:
        step_id = _int(record.get("step_id"))
        if step_id < 0:
            continue
        wave = step_id // int(layers_per_wave)
        by_wave[wave].add(_int(record.get("layer_id")))
    for wave, layer_ids in by_wave.items():
        if len(layer_ids) >= int(layers_per_wave):
            complete_waves.append(wave)
    if complete_waves:
        return max(complete_waves)
    if by_wave:
        return max(by_wave)
    return -1


def _decode_layer_rows(
    records: list[dict[str, Any]],
    *,
    wave_index: int,
    layers_per_wave: int,
) -> list[dict[str, Any]]:
    start_step = int(wave_index) * int(layers_per_wave)
    end_step = start_step + int(layers_per_wave)
    pipeline_records = [
        record for record in records
        if record.get("event") == "moe_pipeline_timing"
        and start_step <= _int(record.get("step_id")) < end_step
        and _int(record.get("step_id")) >= 0
    ]
    timeline_records = [
        record for record in records
        if record.get("event") == "moe_offload_timeline"
        and start_step <= _int(record.get("step_id")) < end_step
    ]
    pipeline_detail_records = [
        record for record in records
        if record.get("event") == "moe_pipeline_detail_timing"
        and start_step <= _int(record.get("step_id")) < end_step
    ]
    decision_records = [
        record for record in records
        if record.get("event") == "moe_offload_profile"
        and record.get("name") == "layered_path_decision"
        and start_step <= _int((record.get("payload") or {}).get("step_id")) < end_step
    ]
    details_by_step: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in timeline_records:
        name = str(record.get("name") or "unknown")
        if name == "prepare_fixed_slot_plan":
            continue
        details_by_step[_int(record.get("step_id"))][name] += _timeline_duration_ms(record)

    r_details_by_step: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in pipeline_detail_records:
        name = str(record.get("name") or "unknown")
        if name == "r_token_dispatch_total":
            continue
        if not name.startswith("r_"):
            continue
        r_details_by_step[_int(record.get("step_id"))][name] += _float(record.get("duration_ms"))

    decisions_by_step: dict[int, dict[str, Any]] = {}
    for record in decision_records:
        payload = record.get("payload") or {}
        decisions_by_step[_int(payload.get("step_id"))] = payload

    rows: list[dict[str, Any]] = []
    for record in sorted(pipeline_records, key=lambda item: (_int(item.get("layer_id")), _int(item.get("step_id")))):
        step_id = _int(record.get("step_id"))
        pipeline_segments = [
            (key, PIPELINE_STAGE_LABELS[key], _float(record.get(key)))
            for key in PIPELINE_STAGE_LABELS
        ]
        detail_totals = details_by_step.get(step_id, {})
        detail_segments = [
            (key, OFFLOAD_STAGE_LABELS.get(key, key), detail_totals.get(key, 0.0))
            for key in OFFLOAD_STAGE_LABELS
            if key != "prepare_fixed_slot_plan" and detail_totals.get(key, 0.0) > 0
        ]
        t_detail_total = sum(segment[2] for segment in detail_segments)
        t_residual_ms = max(0.0, _float(record.get("stage_t_ms")) - t_detail_total)
        if t_residual_ms > 0.001 and detail_segments:
            detail_segments.append(
                ("t_residual_wait", PIPELINE_DETAIL_STAGE_LABELS["t_residual_wait"], t_residual_ms)
            )

        r_detail_totals = r_details_by_step.get(step_id, {})
        r_detail_segments = [
            (key, PIPELINE_DETAIL_STAGE_LABELS.get(key, key), r_detail_totals.get(key, 0.0))
            for key in PIPELINE_DETAIL_STAGE_LABELS
            if key.startswith("r_")
            and key != "r_token_dispatch_total"
            and key != "r_residual_wait"
            and r_detail_totals.get(key, 0.0) > 0
        ]
        r_detail_total = sum(segment[2] for segment in r_detail_segments)
        r_residual_ms = max(0.0, _float(record.get("stage_r_ms")) - r_detail_total)
        if r_residual_ms > 0.001 and r_detail_segments:
            r_detail_segments.append(
                ("r_residual_wait", PIPELINE_DETAIL_STAGE_LABELS["r_residual_wait"], r_residual_ms)
            )
        rows.append(
            {
                "layer_id": _int(record.get("layer_id")),
                "step_id": step_id,
                "decision": decisions_by_step.get(step_id, {}),
                "pipeline_segments": pipeline_segments,
                "pipeline_total_ms": sum(segment[2] for segment in pipeline_segments),
                "detail_segments": detail_segments,
                "detail_total_ms": sum(segment[2] for segment in detail_segments),
                "r_detail_segments": r_detail_segments,
                "r_detail_total_ms": sum(segment[2] for segment in r_detail_segments),
            }
        )
    return rows


def _append_decode_segments(
    lines: list[str],
    segments: list[tuple[str, str, float]],
    *,
    x: float,
    y: float,
    height: float,
    width: float,
    scale_ms: float,
    text_threshold: float,
    source: str | None = None,
) -> None:
    cursor = x
    for key, label, duration_ms in segments:
        segment_width = _float(duration_ms) / scale_ms * width if scale_ms > 0 else 0.0
        if segment_width <= 0:
            continue
        if source is not None:
            segment_source = source
        elif key in PIPELINE_STAGE_LABELS:
            segment_source = "pipeline"
        elif key in PIPELINE_DETAIL_STAGE_LABELS:
            segment_source = "pipeline_detail"
        else:
            segment_source = "offload_detail"
        color = _stage_color(key, segment_source)
        title = f"{label}: {_float(duration_ms):.4f} ms"
        lines.append(
            f'<rect x="{cursor:.2f}" y="{y}" width="{segment_width:.2f}" height="{height}" '
            f'rx="2" fill="{color}"><title>{_svg_escape(title)}</title></rect>'
        )
        if segment_width >= text_threshold and height >= 9:
            lines.append(
                f'<text class="segtext" x="{cursor + 4:.2f}" y="{y + height - 2:.2f}">'
                f'{_svg_escape(_tiny_stage_label(key))}</text>'
            )
        cursor += segment_width


def _decode_row_meta(row: dict[str, Any]) -> str:
    decision = row.get("decision") or {}
    path = str(decision.get("path") or "")
    if path == "slot_cache_path":
        return "slot"
    if path == "full_weight_path":
        return "full"
    if path == "fail_closed":
        return "fail"
    return "-"


def _wave_step_range(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    step_ids = [_int(row.get("step_id")) for row in rows]
    return f"{min(step_ids)}..{max(step_ids)}"


def _tiny_stage_label(key: str) -> str:
    return {
        "stage_t_ms": "T",
        "stage_r_ms": "R",
        "stage_c_ms": "C",
        "stage_m_ms": "M",
        "expert_h2d_load_sync": "H2D",
        "slot_mapping_build": "map",
        "slot_cache_lookup": "cache",
        "slot_allocate": "alloc",
        "host_bundle_lookup": "host",
        "active_expert_normalize": "active",
        "prepared_slot_weights": "view",
        "r_log2phy_map": "map",
        "r_build_dispatch_input": "build",
        "r_init_routing": "route",
        "r_expert_tokens_cast": "cast",
        "r_residual_wait": "wait",
        "t_residual_wait": "wait",
    }.get(key, key[:5])


def _rows_to_svg_segments(rows: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    return [
        {
            "label": str(row.get("name") or row.get("key") or "unknown"),
            "duration_ms": _float(row.get("total_ms")),
            "color": _stage_color(str(row.get("key") or row.get("name") or ""), source),
        }
        for row in rows
        if _float(row.get("total_ms")) > 0
    ]


def _segments_total_ms(segments: list[dict[str, Any]]) -> float:
    return sum(_float(segment.get("duration_ms")) for segment in segments)


def _stage_color(key: str, source: str) -> str:
    pipeline_colors = {
        "stage_t_ms": "#4c78a8",
        "stage_r_ms": "#f58518",
        "stage_c_ms": "#54a24b",
        "stage_m_ms": "#b279a2",
    }
    pipeline_detail_colors = {
        "r_log2phy_map": "#bc6c25",
        "r_build_dispatch_input": "#dda15e",
        "r_init_routing": "#f58518",
        "r_expert_tokens_cast": "#ffbe7d",
        "r_token_dispatch_total": "#8d5524",
        "r_residual_wait": "#d45087",
        "t_residual_wait": "#7a5195",
    }
    offload_colors = {
        "active_expert_normalize": "#72b7b2",
        "slot_cache_lookup": "#e45756",
        "slot_allocate": "#ff9da6",
        "host_bundle_lookup": "#9d755d",
        "expert_h2d_load_sync": "#4c78a8",
        "slot_mapping_build": "#59a14f",
        "prepared_slot_weights": "#b07aa1",
        "prepare_fixed_slot_plan": "#2f4b7c",
    }
    if source == "pipeline":
        return pipeline_colors.get(key, "#6b7280")
    if source == "pipeline_detail":
        return pipeline_detail_colors.get(key, "#6b7280")
    return offload_colors.get(key, "#6b7280")


def _short_label(label: str) -> str:
    if label == "T offload plan/load":
        return "T plan/load"
    if label == "logical->physical map":
        return "map"
    if label == "prepared_slot_weights":
        return "slot view"
    return label[:24]


def _human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{value} B"


def _svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _timeline_duration_ms(record: dict[str, Any]) -> float:
    duration_us = _float(record.get("duration_us"))
    if duration_us > 0:
        return duration_us / 1000.0
    if "start_ns" in record and "end_ns" in record:
        return max(0, _int(record.get("end_ns")) - _int(record.get("start_ns"))) / 1_000_000
    return _float(record.get("seconds")) * 1000.0


def _timeline_suffix(name: str, payload: dict[str, Any]) -> str:
    if name == "slot_cache_lookup":
        hit = "hit" if bool(payload.get("cache_hit")) else "miss"
        return f" e{payload.get('expert_id', '?')} {hit}"
    if name in {"slot_allocate", "host_bundle_lookup", "expert_h2d_load_sync"}:
        return f" e{payload.get('expert_id', '?')}"
    return ""


def _section_title(source: str) -> str:
    if source == "pipeline":
        return "Pipeline"
    if source == "offload_detail":
        return "Offload Detail"
    if source == "pipeline_detail":
        return "Pipeline Detail"
    return source


def _escape_mermaid(value: str) -> str:
    return value.replace(":", "-").replace(",", " ")


def _ns_to_us(value: int) -> float:
    return round(float(value) / 1000.0, 3)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SEW-MoE offload timing summary, Mermaid timeline, SVG image, and Chrome trace."
    )
    parser.add_argument("--profile-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--svg-output", type=Path)
    parser.add_argument("--decode-layer-svg-output", type=Path)
    parser.add_argument("--decode-wave-index", type=int)
    parser.add_argument("--layers-per-wave", type=int, default=48)
    parser.add_argument("--chrome-trace-output", type=Path)
    parser.add_argument("--step-id", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.profile_jsonl)
    summary = summarize_profile(records, step_id=args.step_id)
    markdown = render_markdown(records, step_id=args.step_id)
    svg = render_svg(records, step_id=args.step_id)
    decode_layer_svg = render_decode_layer_svg(
        records,
        wave_index=args.decode_wave_index,
        layers_per_wave=args.layers_per_wave,
    )
    chrome_trace = render_chrome_trace(records, step_id=args.step_id)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if args.svg_output:
        args.svg_output.parent.mkdir(parents=True, exist_ok=True)
        args.svg_output.write_text(svg, encoding="utf-8")
    if args.decode_layer_svg_output:
        args.decode_layer_svg_output.parent.mkdir(parents=True, exist_ok=True)
        args.decode_layer_svg_output.write_text(decode_layer_svg, encoding="utf-8")
    if args.chrome_trace_output:
        args.chrome_trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.chrome_trace_output.write_text(
            json.dumps(chrome_trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not (
        args.summary_json
        or args.markdown_output
        or args.svg_output
        or args.decode_layer_svg_output
        or args.chrome_trace_output
    ):
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
