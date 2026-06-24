# SPDX-License-Identifier: Apache-2.0
"""Measure one SEW-Offload expert host-to-device transfer breakdown.

This is a microbenchmark for the current fixed-slot miss path:

    slot.w13.copy_(cpu_w13)
    slot.w2.copy_(cpu_w2)

It intentionally does not load the full model.  By sweeping several payload
sizes and fitting ``time_ms = fixed_ms + bytes * slope_ms_per_byte``, the tool
separates payload movement time from per-transfer fixed overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/data/shared-models/Qwen3-30B-A3B"
DEFAULT_SIZE_FACTORS = "0.0625,0.125,0.25,0.5,1,2,4"
DEFAULT_PCIE_PEAK_GBPS = 64.0
DEFAULT_PROFILE_REPEATS = 200
DEFAULT_BATCH_EXPERT_COUNTS = "2,4,8,16"


@dataclass(frozen=True)
class ExpertShape:
    w13_shape: tuple[int, ...]
    w2_shape: tuple[int, ...]
    dtype_name: str

    @property
    def w13_elements(self) -> int:
        return _numel(self.w13_shape)

    @property
    def w2_elements(self) -> int:
        return _numel(self.w2_shape)


@dataclass(frozen=True)
class FitResult:
    slope_ms_per_byte: float
    intercept_ms: float
    points: int

    @property
    def payload_bandwidth_gbps(self) -> float:
        if self.slope_ms_per_byte <= 0:
            return 0.0
        return 1000.0 / self.slope_ms_per_byte / 1_000_000_000


@dataclass(frozen=True)
class CopyPatternSpec:
    name: str
    bytes_per_iteration: int
    copy_calls_per_iteration: int
    repeats: int
    experts_per_iteration: int = 1
    copies_per_sample: int = 1


def parse_size_factors(value: str) -> tuple[float, ...]:
    factors = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(factors) < 2:
        raise ValueError("at least two size factors are required for linear fitting")
    if any(factor <= 0 for factor in factors):
        raise ValueError("size factors must be positive")
    return factors


def parse_batch_expert_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not counts:
        raise ValueError("at least one batch expert count is required")
    if any(count <= 0 for count in counts):
        raise ValueError("batch expert counts must be positive")
    if len(set(counts)) != len(counts):
        raise ValueError("batch expert counts must be unique")
    return counts


def infer_expert_shape_from_config(model: str | Path, *, dtype_name: str | None = None) -> ExpertShape:
    config_path = Path(model)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config.get("moe_intermediate_size", config["intermediate_size"]))
    inferred_dtype = dtype_name or str(config.get("torch_dtype", "bfloat16"))
    return ExpertShape(
        w13_shape=(2 * intermediate_size, hidden_size),
        w2_shape=(hidden_size, intermediate_size),
        dtype_name=inferred_dtype,
    )


def linear_fit(points: list[tuple[int, float]]) -> FitResult:
    if len(points) < 2:
        raise ValueError("linear_fit requires at least two points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("linear_fit requires distinct byte sizes")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    return FitResult(
        slope_ms_per_byte=slope,
        intercept_ms=intercept,
        points=len(points),
    )


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "stdev": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": _percentile(values, 10),
        "p90": _percentile(values, 90),
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def make_breakdown(
    *,
    expert_bytes: int,
    expert_event_ms: float,
    expert_wall_ms: float,
    fit: FitResult,
    pcie_peak_gbps: float | None,
) -> dict[str, float | None]:
    payload_ms = max(0.0, fit.slope_ms_per_byte * float(expert_bytes))
    fixed_plus_residual_ms = expert_event_ms - payload_ms
    fixed_plus_residual_clipped_ms = max(0.0, fixed_plus_residual_ms)
    effective_gbps = _gbps(expert_bytes, expert_event_ms)
    payload_gbps = fit.payload_bandwidth_gbps
    wall_extra_ms = expert_wall_ms - expert_event_ms
    return {
        "expert_event_ms": expert_event_ms,
        "expert_wall_ms": expert_wall_ms,
        "payload_movement_ms_from_fit": payload_ms,
        "fixed_overhead_ms_from_fit_intercept": fit.intercept_ms,
        "fixed_plus_residual_ms": fixed_plus_residual_ms,
        "fixed_plus_residual_clipped_ms": fixed_plus_residual_clipped_ms,
        "wall_extra_ms": wall_extra_ms,
        "payload_fraction_of_event": payload_ms / expert_event_ms if expert_event_ms > 0 else None,
        "fixed_plus_residual_fraction_of_event": (
            fixed_plus_residual_clipped_ms / expert_event_ms if expert_event_ms > 0 else None
        ),
        "effective_bandwidth_gbps_including_overhead": effective_gbps,
        "payload_bandwidth_gbps_from_fit": payload_gbps,
        "pcie_peak_gbps": pcie_peak_gbps,
        "effective_pcie_utilization": effective_gbps / pcie_peak_gbps if pcie_peak_gbps else None,
        "payload_pcie_utilization": payload_gbps / pcie_peak_gbps if pcie_peak_gbps else None,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "batch_experts", None) is not None:
        args.batch_expert_counts = str(args.batch_experts)
    torch = _import_torch(args.device)
    device = torch.device(args.device)
    shape = _shape_from_args(args)
    dtype = _dtype_from_name(torch, args.dtype)
    element_size = _element_size(torch, dtype)
    expert_bytes = (shape.w13_elements + shape.w2_elements) * element_size

    _set_device(torch, device)
    size_factors = parse_size_factors(args.size_factors)
    ratio = shape.w13_elements / (shape.w13_elements + shape.w2_elements)

    sweep_results = []
    fit_points: list[tuple[int, float]] = []
    for factor in size_factors:
        total_elements = max(2, int(round((expert_bytes * factor) / element_size)))
        w13_elements = max(1, int(round(total_elements * ratio)))
        w2_elements = max(1, total_elements - w13_elements)
        src_w13, src_w2, dst_w13, dst_w2 = _allocate_pair(
            torch=torch,
            dtype=dtype,
            device=device,
            w13_shape=(w13_elements,),
            w2_shape=(w2_elements,),
            pin_memory=args.pin_memory,
        )
        observations = _measure_pair_copy(
            torch=torch,
            device=device,
            src_w13=src_w13,
            src_w2=src_w2,
            dst_w13=dst_w13,
            dst_w2=dst_w2,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        summary = _summarize_observations(observations)
        actual_bytes = (w13_elements + w2_elements) * element_size
        summary.update(
            {
                "factor": factor,
                "bytes": actual_bytes,
                "effective_bandwidth_gbps": _gbps(actual_bytes, summary["event_ms"]["median"]),
            }
        )
        sweep_results.append(summary)
        fit_points.append((actual_bytes, float(summary["event_ms"]["median"])))

    fit = linear_fit(fit_points)

    src_w13, src_w2, dst_w13, dst_w2 = _allocate_pair(
        torch=torch,
        dtype=dtype,
        device=device,
        w13_shape=shape.w13_shape,
        w2_shape=shape.w2_shape,
        pin_memory=args.pin_memory,
    )
    expert_observations = _measure_pair_copy(
        torch=torch,
        device=device,
        src_w13=src_w13,
        src_w2=src_w2,
        dst_w13=dst_w13,
        dst_w2=dst_w2,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    expert_summary = _summarize_observations(expert_observations)

    transfer_engine_summary = None
    if args.include_transfer_engine:
        transfer_engine_observations = _measure_transfer_engine(
            torch=torch,
            device=device,
            src_w13=src_w13,
            src_w2=src_w2,
            dst_w13=dst_w13,
            dst_w2=dst_w2,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        transfer_engine_summary = _summarize_observations(transfer_engine_observations)

    profiler_summary = None
    if args.profile_output_dir:
        profiler_summary = _run_profiler_copy_patterns(
            torch=torch,
            device=device,
            dtype=dtype,
            shape=shape,
            element_size=element_size,
            expert_bytes=expert_bytes,
            pin_memory=args.pin_memory,
            profile_output_dir=Path(args.profile_output_dir),
            profile_repeats=args.profile_repeats,
            profile_copies_per_sample=args.profile_copies_per_sample,
            profile_warmup=args.profile_warmup,
            batch_expert_counts=parse_batch_expert_counts(args.batch_expert_counts),
            pcie_peak_gbps=args.pcie_peak_gbps,
        )

    breakdown = make_breakdown(
        expert_bytes=expert_bytes,
        expert_event_ms=float(expert_summary["event_ms"]["median"]),
        expert_wall_ms=float(expert_summary["wall_ms"]["median"]),
        fit=fit,
        pcie_peak_gbps=args.pcie_peak_gbps,
    )

    result = {
        "experiment": "expert_transfer_breakdown",
        "device": str(device),
        "pin_memory": bool(args.pin_memory),
        "pin_memory_note": (
            "PyTorch CPU allocation flag for control experiments only; "
            "not evidence of true Ascend UVA or pinned-DMA support."
        ),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "pcie_peak_gbps_source": "cli_or_default_assumption",
        "expert": {
            "model": str(args.model) if args.model else None,
            "dtype": args.dtype,
            "element_size": element_size,
            "w13_shape": list(shape.w13_shape),
            "w2_shape": list(shape.w2_shape),
            "w13_bytes": shape.w13_elements * element_size,
            "w2_bytes": shape.w2_elements * element_size,
            "total_bytes": expert_bytes,
            "total_mib": expert_bytes / (1024**2),
        },
        "fit": {
            "points": fit.points,
            "slope_ms_per_byte": fit.slope_ms_per_byte,
            "intercept_ms": fit.intercept_ms,
            "payload_bandwidth_gbps": fit.payload_bandwidth_gbps,
        },
        "breakdown": breakdown,
        "expert_copy_summary": expert_summary,
        "transfer_engine_summary": transfer_engine_summary,
        "profiler_summary": profiler_summary,
        "sweep": sweep_results,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model directory or config.json used to infer expert shape.",
    )
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default=None, help="Override dtype. Defaults to model config torch_dtype.")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--moe-intermediate-size", type=int)
    parser.add_argument("--size-factors", default=DEFAULT_SIZE_FACTORS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "PyTorch CPU allocation flag used only for control experiments. "
            "Ascend NPU does not expose true UVA through this option."
        ),
    )
    parser.add_argument("--include-transfer-engine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--profile-output-dir",
        type=Path,
        help="Optional torch-npu/CANN profiler output directory for copy-pattern timeline evidence.",
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=DEFAULT_PROFILE_REPEATS,
        help="Profile samples per copy pattern.",
    )
    parser.add_argument(
        "--profile-copies-per-sample",
        type=int,
        default=1,
        help="Copy iterations grouped into each profiled sample window.",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=5,
        help="Warmup iterations per profiler copy pattern before profiling starts.",
    )
    parser.add_argument(
        "--batch-experts",
        type=int,
        default=None,
        help="Deprecated alias for --batch-expert-counts with a single value.",
    )
    parser.add_argument(
        "--batch-expert-counts",
        default=DEFAULT_BATCH_EXPERT_COUNTS,
        help="Comma-separated expert counts copied by contiguous profiler batch patterns.",
    )
    parser.add_argument("--pcie-peak-gbps", type=float, default=DEFAULT_PCIE_PEAK_GBPS)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    breakdown = result["breakdown"]
    print(
        "EXPERT_TRANSFER_BREAKDOWN "
        + json.dumps(
            {
                "expert_mib": result["expert"]["total_mib"],
                "event_ms": breakdown["expert_event_ms"],
                "payload_movement_ms": breakdown["payload_movement_ms_from_fit"],
                "fixed_plus_residual_ms": breakdown["fixed_plus_residual_ms"],
                "effective_bandwidth_gbps": breakdown["effective_bandwidth_gbps_including_overhead"],
                "effective_pcie_utilization": breakdown["effective_pcie_utilization"],
                "payload_bandwidth_gbps": breakdown["payload_bandwidth_gbps_from_fit"],
                "payload_pcie_utilization": breakdown["payload_pcie_utilization"],
                "profiler_output_dir": (
                    result["profiler_summary"]["profiler_output_dir"] if result["profiler_summary"] else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _shape_from_args(args: argparse.Namespace) -> ExpertShape:
    dtype_name = args.dtype
    if args.hidden_size is not None or args.moe_intermediate_size is not None:
        if args.hidden_size is None or args.moe_intermediate_size is None:
            raise ValueError("--hidden-size and --moe-intermediate-size must be set together")
        if dtype_name is None:
            dtype_name = "bfloat16"
        return ExpertShape(
            w13_shape=(2 * int(args.moe_intermediate_size), int(args.hidden_size)),
            w2_shape=(int(args.hidden_size), int(args.moe_intermediate_size)),
            dtype_name=dtype_name,
        )
    shape = infer_expert_shape_from_config(args.model, dtype_name=dtype_name)
    if args.dtype is None:
        args.dtype = shape.dtype_name
    return shape


def _measure_pair_copy(
    *,
    torch,
    device,
    src_w13,
    src_w2,
    dst_w13,
    dst_w2,
    warmup: int,
    repeats: int,
) -> list[dict[str, float]]:
    return _measure_callable(
        torch=torch,
        device=device,
        fn=lambda: _copy_pair(src_w13, src_w2, dst_w13, dst_w2),
        warmup=warmup,
        repeats=repeats,
    )


def _measure_transfer_engine(
    *,
    torch,
    device,
    src_w13,
    src_w2,
    dst_w13,
    dst_w2,
    warmup: int,
    repeats: int,
) -> list[dict[str, float]]:
    from vllm_ascend.moe_offload.expert_key import ExpertKey
    from vllm_ascend.moe_offload.host_store import ExpertWeightBundle
    from vllm_ascend.moe_offload.slot_bank import ExpertSlot
    from vllm_ascend.moe_offload.transfer_engine import TransferEngine

    bundle = ExpertWeightBundle(layer_id=0, expert_id=0, w13=src_w13, w2=src_w2)
    slot = ExpertSlot(slot_id=0, w13=dst_w13, w2=dst_w2, expert_key=ExpertKey(0, 0))
    engine = TransferEngine()
    return _measure_callable(
        torch=torch,
        device=device,
        fn=lambda: engine.load_sync(bundle, slot),
        warmup=warmup,
        repeats=repeats,
    )


def _run_profiler_copy_patterns(
    *,
    torch,
    device,
    dtype,
    shape: ExpertShape,
    element_size: int,
    expert_bytes: int,
    pin_memory: bool,
    profile_output_dir: Path,
    profile_repeats: int,
    profile_copies_per_sample: int,
    profile_warmup: int,
    batch_expert_counts: tuple[int, ...],
    pcie_peak_gbps: float | None,
) -> dict[str, Any]:
    if device.type != "npu":
        raise ValueError("torch-npu/CANN profiler patterns require an npu device")
    if profile_repeats <= 0:
        raise ValueError("--profile-repeats must be positive")
    if profile_copies_per_sample <= 0:
        raise ValueError("--profile-copies-per-sample must be positive")
    if profile_warmup < 0:
        raise ValueError("--profile-warmup must be non-negative")
    if not batch_expert_counts:
        raise ValueError("--batch-expert-counts must not be empty")
    if any(count <= 0 for count in batch_expert_counts):
        raise ValueError("--batch-expert-counts must be positive")

    import torch_npu
    from torch.profiler import record_function

    profile_output_dir.mkdir(parents=True, exist_ok=True)
    before = _profile_dirs(profile_output_dir)

    expert_elements = expert_bytes // element_size
    single_src, single_dst = _allocate_buffer(
        torch=torch,
        dtype=dtype,
        device=device,
        elements=expert_elements,
        pin_memory=pin_memory,
    )
    src_w13, src_w2, dst_w13, dst_w2 = _allocate_pair(
        torch=torch,
        dtype=dtype,
        device=device,
        w13_shape=shape.w13_shape,
        w2_shape=shape.w2_shape,
        pin_memory=pin_memory,
    )
    patterns = [
        {
            "spec": CopyPatternSpec(
                name="single_contiguous_expert",
                bytes_per_iteration=expert_bytes,
                copy_calls_per_iteration=1,
                repeats=profile_repeats,
                experts_per_iteration=1,
                copies_per_sample=profile_copies_per_sample,
            ),
            "fn": lambda src=single_src, dst=single_dst: _copy_buffer(src, dst),
        },
        {
            "spec": CopyPatternSpec(
                name="two_tensor_current",
                bytes_per_iteration=expert_bytes,
                copy_calls_per_iteration=2,
                repeats=profile_repeats,
                experts_per_iteration=1,
                copies_per_sample=profile_copies_per_sample,
            ),
            "fn": lambda: _copy_pair(src_w13, src_w2, dst_w13, dst_w2),
        },
    ]
    for batch_experts in batch_expert_counts:
        batched_src, batched_dst = _allocate_buffer(
            torch=torch,
            dtype=dtype,
            device=device,
            elements=expert_elements * batch_experts,
            pin_memory=pin_memory,
        )
        patterns.append(
            {
                "spec": CopyPatternSpec(
                    name=f"batch_{batch_experts}_expert_contiguous",
                    bytes_per_iteration=expert_bytes * batch_experts,
                    copy_calls_per_iteration=1,
                    repeats=profile_repeats,
                    experts_per_iteration=batch_experts,
                    copies_per_sample=profile_copies_per_sample,
                ),
                "fn": lambda src=batched_src, dst=batched_dst: _copy_buffer(src, dst),
            }
        )

    for pattern in patterns:
        fn = pattern["fn"]
        for _ in range(profile_warmup):
            fn()
        _synchronize(torch, device.type)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        msprof_tx=False,
        data_simplification=False,
        record_op_args=True,
        sys_interconnection=True,
    )
    schedule = torch_npu.profiler.schedule(wait=0, warmup=0, active=len(patterns), repeat=1)
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(profile_output_dir),
            worker_name="sew_transfer_patterns",
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for pattern in patterns:
            spec = pattern["spec"]
            for sample_index in range(profile_repeats):
                with record_function(f"sew_transfer_{spec.name}"):
                    for _ in range(profile_copies_per_sample):
                        pattern["fn"]()
                    _synchronize(torch, device.type)
            profiler.step()

    _synchronize(torch, device.type)
    profile_dir = _latest_new_profile_dir(before, profile_output_dir)
    profiler_output_dir = _resolve_profiler_output_dir(profile_dir)
    specs = [pattern["spec"] for pattern in patterns]
    summary = analyze_cann_profiler_output(
        profiler_output_dir=profiler_output_dir,
        pattern_specs=specs,
        pcie_peak_gbps=pcie_peak_gbps,
    )
    summary.update(
        {
            "profile_dir": str(profile_dir),
            "profiler_output_dir": str(profiler_output_dir),
            "profiler_kind": "torch_npu_cann_text",
            "evidence_note": (
                "AscendCL@aclrtMemcpy spans are CANN runtime memcpy timeline events; "
                "pcie.csv/trace PCIe counters are sampled hardware-link counters, not "
                "per-copy raw DMA engine cycles."
            ),
        }
    )
    return summary


def _measure_callable(*, torch, device, fn, warmup: int, repeats: int) -> list[dict[str, float]]:
    observations: list[dict[str, float]] = []
    event_api = _event_api(torch, device.type)
    for index in range(int(warmup) + int(repeats)):
        _synchronize(torch, device.type)
        start_event = event_api.Event(enable_timing=True)
        end_event = event_api.Event(enable_timing=True)
        wall_start = time.perf_counter()
        event_api.current_stream().record_event(start_event)
        fn()
        event_api.current_stream().record_event(end_event)
        _synchronize(torch, device.type)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        event_ms = float(start_event.elapsed_time(end_event))
        if index >= int(warmup):
            observations.append({"event_ms": event_ms, "wall_ms": wall_ms})
    return observations


def _copy_pair(src_w13, src_w2, dst_w13, dst_w2) -> None:
    dst_w13.copy_(src_w13)
    dst_w2.copy_(src_w2)


def _copy_buffer(src, dst) -> None:
    dst.copy_(src)


def _allocate_buffer(*, torch, dtype, device, elements: int, pin_memory: bool):
    src = torch.empty((elements,), dtype=dtype, device="cpu", pin_memory=pin_memory)
    src.fill_(1)
    dst = torch.empty((elements,), dtype=dtype, device=device)
    return src, dst


def _allocate_pair(*, torch, dtype, device, w13_shape, w2_shape, pin_memory: bool):
    src_w13 = torch.empty(w13_shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    src_w2 = torch.empty(w2_shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    src_w13.fill_(1)
    src_w2.fill_(1)
    dst_w13 = torch.empty(w13_shape, dtype=dtype, device=device)
    dst_w2 = torch.empty(w2_shape, dtype=dtype, device=device)
    return src_w13, src_w2, dst_w13, dst_w2


def _summarize_observations(observations: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "event_ms": summarize_values([obs["event_ms"] for obs in observations]),
        "wall_ms": summarize_values([obs["wall_ms"] for obs in observations]),
        "samples": observations,
    }


def analyze_cann_profiler_output(
    *,
    profiler_output_dir: Path,
    pattern_specs: list[CopyPatternSpec],
    pcie_peak_gbps: float | None,
) -> dict[str, Any]:
    trace_path = profiler_output_dir / "trace_view.json"
    pcie_path = profiler_output_dir / "pcie.csv"
    api_statistic_path = profiler_output_dir / "api_statistic.csv"
    return {
        "trace_view": str(trace_path),
        "pcie_csv": str(pcie_path),
        "api_statistic_csv": str(api_statistic_path),
        "patterns": summarize_trace_copy_patterns(
            trace_path=trace_path,
            pattern_specs=pattern_specs,
            pcie_peak_gbps=pcie_peak_gbps,
        ),
        "pcie_csv_summary": summarize_pcie_csv(pcie_path, pcie_peak_gbps=pcie_peak_gbps),
        "api_statistic": summarize_api_statistic_csv(api_statistic_path),
    }


def summarize_trace_copy_patterns(
    *,
    trace_path: Path,
    pattern_specs: list[CopyPatternSpec],
    pcie_peak_gbps: float | None,
) -> dict[str, Any]:
    if not trace_path.exists():
        return {"error": f"missing trace_view.json: {trace_path}"}
    with trace_path.open(encoding="utf-8") as f:
        events = json.load(f)
    if not isinstance(events, list):
        return {"error": f"unexpected trace_view.json payload: {type(events).__name__}"}

    pattern_by_name = {spec.name: spec for spec in pattern_specs}
    windows: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in pattern_specs}
    memcpy_events = []
    sync_events = []
    aten_copy_events = []
    pcie_counter_events = []
    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name", ""))
        if event.get("ph") == "X":
            if name.startswith("sew_transfer_"):
                pattern_name = name.removeprefix("sew_transfer_")
                if pattern_name in windows:
                    windows[pattern_name].append(event)
            elif name.startswith("AscendCL@aclrtMemcpy"):
                memcpy_events.append(event)
            elif name.startswith("AscendCL@aclrtSynchronize"):
                sync_events.append(event)
            elif name == "aten::copy_":
                aten_copy_events.append(event)
        elif event.get("ph") == "C" and name.startswith("PCIe_"):
            pcie_counter_events.append(event)

    results = {}
    for pattern_name, spec in pattern_by_name.items():
        pattern_windows = windows.get(pattern_name, [])
        window_ranges = [_event_range_us(event) for event in pattern_windows]
        window_ranges = [time_range for time_range in window_ranges if time_range is not None]
        in_memcpy = _events_in_ranges(memcpy_events, window_ranges)
        in_sync = _events_in_ranges(sync_events, window_ranges)
        in_aten_copy = _events_in_ranges(aten_copy_events, window_ranges)
        window_us = sum(stop - start for start, stop in window_ranges)
        memcpy_us = sum(_event_duration_us(event) for event in in_memcpy)
        sync_us = sum(_event_duration_us(event) for event in in_sync)
        aten_copy_us = sum(_event_duration_us(event) for event in in_aten_copy)
        actual_samples = len(window_ranges)
        total_iterations = actual_samples * spec.copies_per_sample
        total_bytes = spec.bytes_per_iteration * total_iterations
        memcpy_ms = memcpy_us / 1000.0
        window_ms = window_us / 1000.0
        expected_copy_calls = spec.copy_calls_per_iteration * total_iterations
        host_non_memcpy_us = window_us - memcpy_us
        host_other_us = window_us - memcpy_us - sync_us
        total_experts = spec.experts_per_iteration * total_iterations
        window_us_per_expert = _safe_div(window_us, total_experts)
        memcpy_us_per_expert = _safe_div(memcpy_us, total_experts)
        non_memcpy_us_per_expert = _safe_div(host_non_memcpy_us, total_experts)
        host_other_us_per_expert = _safe_div(host_other_us, total_experts)
        sync_us_per_expert = _safe_div(sync_us, total_experts)
        sample_summary = _summarize_copy_pattern_samples(
            spec=spec,
            window_ranges=window_ranges,
            memcpy_events=memcpy_events,
            sync_events=sync_events,
            aten_copy_events=aten_copy_events,
        )
        results[pattern_name] = {
            "bytes_per_iteration": spec.bytes_per_iteration,
            "experts_per_iteration": spec.experts_per_iteration,
            "copies_per_sample": spec.copies_per_sample,
            "total_bytes": total_bytes,
            "total_experts": total_experts,
            "total_mib": total_bytes / (1024**2),
            "repeats": spec.repeats,
            "copy_calls_per_iteration": spec.copy_calls_per_iteration,
            "expected_copy_calls": expected_copy_calls,
            "record_window_count": len(pattern_windows),
            "record_window_us": window_us,
            "record_window_us_per_iteration": _safe_div(window_us, spec.repeats),
            "record_window_us_per_expert": window_us_per_expert,
            "aclrt_memcpy_count": len(in_memcpy),
            "aclrt_memcpy_expected_count_delta": len(in_memcpy) - expected_copy_calls,
            "aclrt_memcpy_us": memcpy_us,
            "aclrt_memcpy_us_per_iteration": _safe_div(memcpy_us, spec.repeats),
            "aclrt_memcpy_us_per_expert": memcpy_us_per_expert,
            "aclrt_memcpy_us_per_call": _safe_div(memcpy_us, len(in_memcpy)),
            "aclrt_memcpy_us_summary": summarize_values([_event_duration_us(event) for event in in_memcpy]),
            "aclrt_memcpy_bandwidth_gbps": _gbps(total_bytes, memcpy_ms),
            "record_window_bandwidth_gbps": _gbps(total_bytes, window_ms),
            "aclrt_memcpy_pcie_utilization": (
                _gbps(total_bytes, memcpy_ms) / pcie_peak_gbps if pcie_peak_gbps else None
            ),
            "record_window_pcie_utilization": (
                _gbps(total_bytes, window_ms) / pcie_peak_gbps if pcie_peak_gbps else None
            ),
            "host_window_non_memcpy_us": host_non_memcpy_us,
            "host_window_non_memcpy_us_per_expert": non_memcpy_us_per_expert,
            "host_window_non_memcpy_fraction": (
                host_non_memcpy_us / window_us if window_us > 0 else None
            ),
            "aclrt_synchronize_count": len(in_sync),
            "aclrt_synchronize_us": sync_us,
            "aclrt_synchronize_us_per_expert": sync_us_per_expert,
            "aten_copy_count": len(in_aten_copy),
            "aten_copy_us": aten_copy_us,
            "sample_summary": sample_summary,
            "time_breakdown": {
                "record_window_us": window_us,
                "aclrt_memcpy_us": memcpy_us,
                "aclrt_synchronize_us": sync_us,
                "host_other_us": host_other_us,
                "record_window_us_per_expert": window_us_per_expert,
                "aclrt_memcpy_us_per_expert": memcpy_us_per_expert,
                "aclrt_synchronize_us_per_expert": sync_us_per_expert,
                "host_other_us_per_expert": host_other_us_per_expert,
                "aclrt_memcpy_fraction": memcpy_us / window_us if window_us > 0 else None,
                "aclrt_synchronize_fraction": sync_us / window_us if window_us > 0 else None,
                "host_other_fraction": host_other_us / window_us if window_us > 0 else None,
            },
            "pcie_trace_counters": _summarize_pcie_trace_counters(
                pcie_counter_events,
                window_ranges,
            ),
        }
    return results


def summarize_pcie_csv(path: Path, *, pcie_peak_gbps: float | None) -> dict[str, Any]:
    rows = _read_csv(path)
    summary = {}
    for row in rows:
        mode = str(row.get("Mode", "")).strip()
        if not mode:
            continue
        values = {
            "device_id": row.get("Device_id", ""),
            "min": _float(row.get("Min")),
            "max": _float(row.get("Max")),
            "avg": _float(row.get("Avg")),
        }
        if mode.endswith("(MB/s)"):
            values.update(
                {
                    "min_gbps": values["min"] / 1000.0,
                    "max_gbps": values["max"] / 1000.0,
                    "avg_gbps": values["avg"] / 1000.0,
                    "max_pcie_utilization": (
                        values["max"] / 1000.0 / pcie_peak_gbps if pcie_peak_gbps else None
                    ),
                    "avg_pcie_utilization": (
                        values["avg"] / 1000.0 / pcie_peak_gbps if pcie_peak_gbps else None
                    ),
                }
            )
        summary[mode] = values
    return summary


def _summarize_copy_pattern_samples(
    *,
    spec: CopyPatternSpec,
    window_ranges: list[tuple[float, float]],
    memcpy_events: list[dict[str, Any]],
    sync_events: list[dict[str, Any]],
    aten_copy_events: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_copy_calls_per_sample = spec.copy_calls_per_iteration * spec.copies_per_sample
    bytes_per_sample = spec.bytes_per_iteration * spec.copies_per_sample
    experts_per_sample = spec.experts_per_iteration * spec.copies_per_sample
    samples: list[dict[str, float | int]] = []
    for start, stop in window_ranges:
        sample_range = [(start, stop)]
        sample_memcpy_events = _events_in_ranges(memcpy_events, sample_range)
        sample_sync_events = _events_in_ranges(sync_events, sample_range)
        sample_aten_copy_events = _events_in_ranges(aten_copy_events, sample_range)
        window_us = stop - start
        memcpy_us = sum(_event_duration_us(event) for event in sample_memcpy_events)
        sync_us = sum(_event_duration_us(event) for event in sample_sync_events)
        aten_copy_us = sum(_event_duration_us(event) for event in sample_aten_copy_events)
        overhead_us = window_us - memcpy_us
        samples.append(
            {
                "record_window_ms_per_expert": _us_per_expert_to_ms(window_us, experts_per_sample),
                "aclrt_memcpy_ms_per_expert": _us_per_expert_to_ms(memcpy_us, experts_per_sample),
                "overhead_ms_per_expert": _us_per_expert_to_ms(overhead_us, experts_per_sample),
                "aclrt_synchronize_ms_per_expert": _us_per_expert_to_ms(sync_us, experts_per_sample),
                "aten_copy_ms_per_expert": _us_per_expert_to_ms(aten_copy_us, experts_per_sample),
                "aclrt_memcpy_bandwidth_gbps": _gbps(bytes_per_sample, memcpy_us / 1000.0),
                "record_window_bandwidth_gbps": _gbps(bytes_per_sample, window_us / 1000.0),
                "aclrt_memcpy_count": len(sample_memcpy_events),
                "aclrt_memcpy_expected_count_delta": len(sample_memcpy_events)
                - expected_copy_calls_per_sample,
            }
        )

    def summarize_field(field: str) -> dict[str, float]:
        return summarize_values([float(sample[field]) for sample in samples])

    deltas = [int(sample["aclrt_memcpy_expected_count_delta"]) for sample in samples]
    return {
        "samples": len(samples),
        "expected_copy_calls_per_sample": expected_copy_calls_per_sample,
        "bytes_per_sample": bytes_per_sample,
        "experts_per_sample": experts_per_sample,
        "record_window_ms_per_expert": summarize_field("record_window_ms_per_expert"),
        "aclrt_memcpy_ms_per_expert": summarize_field("aclrt_memcpy_ms_per_expert"),
        "overhead_ms_per_expert": summarize_field("overhead_ms_per_expert"),
        "aclrt_synchronize_ms_per_expert": summarize_field("aclrt_synchronize_ms_per_expert"),
        "aten_copy_ms_per_expert": summarize_field("aten_copy_ms_per_expert"),
        "aclrt_memcpy_bandwidth_gbps": summarize_field("aclrt_memcpy_bandwidth_gbps"),
        "record_window_bandwidth_gbps": summarize_field("record_window_bandwidth_gbps"),
        "aclrt_memcpy_expected_count_delta": {
            "min": min(deltas) if deltas else 0,
            "max": max(deltas) if deltas else 0,
            "nonzero_samples": sum(1 for delta in deltas if delta != 0),
        },
    }


def _us_per_expert_to_ms(us: float, experts: int) -> float:
    value = _safe_div(us, experts)
    if value is None:
        return 0.0
    return value / 1000.0


def summarize_api_statistic_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    summary = {}
    for row in rows:
        api_name = str(row.get("API Name", "")).strip()
        if not api_name:
            continue
        summary[api_name] = {
            "device_id": row.get("Device_id", ""),
            "level": row.get("Level", ""),
            "time_us": _float(row.get("Time(us)")),
            "count": _int(row.get("Count")),
            "avg_us": _float(row.get("Avg(us)")),
            "min_us": _float(row.get("Min(us)")),
            "max_us": _float(row.get("Max(us)")),
            "variance": _float(row.get("Variance")),
        }
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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


def _event_ts_us(event: dict[str, Any]) -> float | None:
    if "ts" not in event:
        return None
    return _float(event.get("ts"))


def _event_duration_us(event: dict[str, Any]) -> float:
    return _float(event.get("dur"))


def _event_range_us(event: dict[str, Any]) -> tuple[float, float] | None:
    start = _event_ts_us(event)
    if start is None:
        return None
    return start, start + _event_duration_us(event)


def _events_in_ranges(
    events: list[dict[str, Any]],
    ranges: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    if not ranges:
        return []
    result = []
    for event in events:
        ts = _event_ts_us(event)
        if ts is None:
            continue
        if any(start <= ts <= stop for start, stop in ranges):
            result.append(event)
    return result


def _summarize_pcie_trace_counters(
    events: list[dict[str, Any]],
    ranges: list[tuple[float, float]],
) -> dict[str, Any]:
    in_window = _events_in_ranges(events, ranges)
    values_by_counter: dict[str, dict[str, list[float]]] = {}
    for event in in_window:
        name = str(event.get("name", ""))
        args = event.get("args", {})
        if not isinstance(args, dict):
            continue
        counter_values = values_by_counter.setdefault(name, {})
        for direction in ("Rx", "Tx"):
            if direction in args:
                counter_values.setdefault(direction, []).append(_float(args[direction]))

    summary = {}
    for name, values_by_direction in values_by_counter.items():
        direction_summary = {}
        for direction, values in values_by_direction.items():
            stats = summarize_values(values)
            stats["min"] = min(values) if values else 0.0
            stats["max"] = max(values) if values else 0.0
            if "latency" in name.lower():
                stats["unit"] = "us"
            else:
                stats["unit"] = "MB/s"
                stats["mean_gbps"] = stats["mean"] / 1000.0
                stats["max_gbps"] = stats["max"] / 1000.0
            stats["sample_count"] = len(values)
            direction_summary[direction] = stats
        summary[name] = direction_summary
    return summary


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _profile_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path for path in root.iterdir() if path.is_dir() and path.name.endswith("_ascend_pt")}


def _latest_new_profile_dir(before: set[Path], root: Path) -> Path:
    after = _profile_dirs(root)
    candidates = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(after, key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"torch-npu profiler did not create an *_ascend_pt directory under {root}")
    return candidates[-1]


def _resolve_profiler_output_dir(path: Path) -> Path:
    if path.name == "ASCEND_PROFILER_OUTPUT":
        return path
    nested = path / "ASCEND_PROFILER_OUTPUT"
    if nested.exists():
        return nested
    return path


def _import_torch(device: str):
    import torch

    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401
    return torch


def _event_api(torch, device_type: str):
    if device_type == "npu":
        return torch.npu
    if device_type == "cuda":
        return torch.cuda
    raise ValueError(f"event timing requires npu or cuda device, got {device_type!r}")


def _set_device(torch, device) -> None:
    if device.type == "npu" and device.index is not None:
        torch.npu.set_device(device)
    elif device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)


def _synchronize(torch, device_type: str) -> None:
    if device_type == "npu":
        torch.npu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()
    else:
        raise ValueError(f"synchronize requires npu or cuda device, got {device_type!r}")


def _dtype_from_name(torch, dtype_name: str | None):
    normalized = (dtype_name or "bfloat16").lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "int8": "int8",
        "uint8": "uint8",
    }
    attr = aliases.get(normalized)
    if attr is None or not hasattr(torch, attr):
        raise ValueError(f"unsupported dtype: {dtype_name}")
    return getattr(torch, attr)


def _element_size(torch, dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _gbps(num_bytes: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return float(num_bytes) / (ms / 1000.0) / 1_000_000_000


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


if __name__ == "__main__":
    main()
