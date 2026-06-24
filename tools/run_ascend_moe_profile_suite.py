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
import importlib.util
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER_PATH = SCRIPT_DIR / "analyze_ascend_moe_profile.py"


@dataclass(frozen=True)
class Scenario:
    name: str
    phase: str
    description: str
    result_filename: str
    bench_args: list[str]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _post(url: str, timeout: float):
    req = request.Request(url, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        response.read()


def _profile_endpoint(profile_url: str, endpoint: str) -> str:
    return profile_url.rstrip("/") + "/" + endpoint.lstrip("/")


def _run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode


def _profile_dirs(profiler_dir: Path) -> set[Path]:
    if not profiler_dir.exists():
        return set()
    return {path for path in profiler_dir.glob("*_ascend_pt") if path.is_dir()}


def _new_profile_dirs(before: set[Path], profiler_dir: Path) -> list[Path]:
    after = _profile_dirs(profiler_dir)
    new_dirs = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if new_dirs:
        return new_dirs
    return sorted(after, key=lambda path: path.stat().st_mtime)[-1:]


def _jsonl_offset(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return path.stat().st_size


def _snapshot_jsonl_delta(source: Path | None, offset: int, dest: Path) -> bool:
    if source is None or not source.exists():
        return False
    current_size = source.stat().st_size
    if current_size <= offset:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src:
        src.seek(offset)
        payload = src.read()
    dest.write_bytes(payload)
    return bool(payload)


def _analyse_profile_dir(profile_dir: Path):
    from torch_npu.profiler.profiler import analyse

    analyse(str(profile_dir))


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_ascend_moe_profile", ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bench_base_args(args: argparse.Namespace, scenario_dir: Path, result_filename: str) -> list[str]:
    return [
        *shlex.split(args.bench_command),
        "--backend",
        args.backend,
        "--base-url",
        args.base_url,
        "--endpoint",
        args.endpoint,
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.tokenizer,
        "--request-rate",
        args.request_rate,
        "--max-concurrency",
        str(args.max_concurrency),
        "--num-warmups",
        str(args.num_warmups),
        "--save-result",
        "--result-dir",
        str(scenario_dir),
        "--result-filename",
        result_filename,
        "--ignore-eos",
    ]


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    mixed_result = "serving_qwen3_30b_a3b_mixed_torchprof.json"
    prefill_result = "serving_qwen3_30b_a3b_prefill_torchprof.json"
    decode_result = "serving_qwen3_30b_a3b_decode_torchprof.json"
    return [
        Scenario(
            name="mixed",
            phase="mixed",
            description="ShareGPT mixed workload for macro TTFT and TPOT view.",
            result_filename=mixed_result,
            bench_args=[
                "--dataset-name",
                "sharegpt",
                "--dataset-path",
                str(args.sharegpt_dataset),
                "--num-prompts",
                str(args.mixed_num_prompts),
                "--sharegpt-output-len",
                str(args.mixed_output_len),
            ],
        ),
        Scenario(
            name="prefill",
            phase="prefill",
            description="Long-prompt short-output window to amplify TTFT and prefill kernels.",
            result_filename=prefill_result,
            bench_args=[
                "--dataset-name",
                "random",
                "--random-input-len",
                str(args.prefill_input_len),
                "--random-output-len",
                str(args.prefill_output_len),
                "--num-prompts",
                str(args.prefill_num_prompts),
            ],
        ),
        Scenario(
            name="decode",
            phase="decode",
            description="Short-prompt long-output window to amplify TPOT and decode MoE kernels.",
            result_filename=decode_result,
            bench_args=[
                "--dataset-name",
                "random",
                "--random-input-len",
                str(args.decode_input_len),
                "--random-output-len",
                str(args.decode_output_len),
                "--num-prompts",
                str(args.decode_num_prompts),
            ],
        ),
    ]


def _scenario_manifest(
    scenario: Scenario,
    command: list[str],
    scenario_dir: Path,
    profile_dirs: list[Path],
    returncode: int | None,
    sew_moe_trace_source: Path | None = None,
    sew_moe_profile_source: Path | None = None,
    sew_moe_trace_jsonl: Path | None = None,
    sew_moe_profile_jsonl: Path | None = None,
) -> dict[str, Any]:
    benchmark_json = scenario_dir / scenario.result_filename
    payload = {
        "name": scenario.name,
        "phase": scenario.phase,
        "description": scenario.description,
        "command": command,
        "returncode": returncode,
        "benchmark_json": str(benchmark_json),
        "profile_dirs": [str(path) for path in profile_dirs],
        "profiler_outputs": [
            str(path / "ASCEND_PROFILER_OUTPUT") for path in profile_dirs if (path / "ASCEND_PROFILER_OUTPUT").exists()
        ],
    }
    if sew_moe_trace_source is not None:
        payload["sew_moe_source_trace_jsonl"] = str(sew_moe_trace_source)
        payload["sew_moe_trace_jsonl"] = str(sew_moe_trace_jsonl or scenario_dir / "moe_offload_trace.jsonl")
    if sew_moe_profile_source is not None:
        payload["sew_moe_source_profile_jsonl"] = str(sew_moe_profile_source)
        payload["sew_moe_profile_jsonl"] = str(sew_moe_profile_jsonl or scenario_dir / "sew_moe_profile.jsonl")
    return payload


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or Path("benchmarks/results") / f"qwen3_30b_a3b_ascend_pt_{_utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(args)
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "profiler_dir": str(args.profiler_dir),
        "base_url": args.base_url,
        "profile_url": args.profile_url,
        "scenarios": [],
    }
    trace_source = args.sew_moe_trace_path
    profile_source = args.sew_moe_profile_path
    if trace_source is not None:
        manifest["sew_moe_source_trace_jsonl"] = str(trace_source)
    if profile_source is not None:
        manifest["sew_moe_source_profile_jsonl"] = str(profile_source)

    for scenario in scenarios:
        scenario_dir = output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        command = _bench_base_args(args, scenario_dir, scenario.result_filename) + scenario.bench_args
        trace_dest = scenario_dir / "moe_offload_trace.jsonl" if trace_source is not None else None
        profile_dest = scenario_dir / "sew_moe_profile.jsonl" if profile_source is not None else None
        if args.dry_run:
            manifest["scenarios"].append(
                _scenario_manifest(
                    scenario,
                    command,
                    scenario_dir,
                    [],
                    None,
                    sew_moe_trace_source=trace_source,
                    sew_moe_profile_source=profile_source,
                    sew_moe_trace_jsonl=trace_dest,
                    sew_moe_profile_jsonl=profile_dest,
                ))
            continue

        before = _profile_dirs(args.profiler_dir)
        trace_offset = _jsonl_offset(trace_source)
        profile_offset = _jsonl_offset(profile_source)
        _post(_profile_endpoint(args.profile_url, "start_profile"), args.http_timeout)
        returncode = -1
        try:
            returncode = _run_command(command, scenario_dir / "benchmark.log")
        finally:
            _post(_profile_endpoint(args.profile_url, "stop_profile"), args.http_timeout)

        wrote_trace = _snapshot_jsonl_delta(trace_source, trace_offset, trace_dest) if trace_dest is not None else False
        wrote_profile = (
            _snapshot_jsonl_delta(profile_source, profile_offset, profile_dest) if profile_dest is not None else False
        )
        if args.require_sew_moe_artifacts and trace_source is not None and not wrote_trace:
            raise RuntimeError(f"Missing SEW-MoE trace delta for scenario {scenario.name}: {trace_source}")
        if args.require_sew_moe_artifacts and profile_source is not None and not wrote_profile:
            raise RuntimeError(f"Missing SEW-MoE profile delta for scenario {scenario.name}: {profile_source}")

        if args.stop_analyse_delay_sec:
            time.sleep(args.stop_analyse_delay_sec)

        profile_dirs = _new_profile_dirs(before, args.profiler_dir)
        if not args.skip_analyse:
            for profile_dir in profile_dirs:
                _analyse_profile_dir(profile_dir)

        manifest["scenarios"].append(
            _scenario_manifest(
                scenario,
                command,
                scenario_dir,
                profile_dirs,
                returncode,
                sew_moe_trace_source=trace_source,
                sew_moe_profile_source=profile_source,
                sew_moe_trace_jsonl=trace_dest if wrote_trace or trace_dest is not None else None,
                sew_moe_profile_jsonl=profile_dest if wrote_profile or profile_dest is not None else None,
            ))

    manifest_path = output_dir / "profile_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.dry_run:
        _write_analysis_report(
            manifest,
            output_dir,
            run_slot_sweep=args.run_slot_sweep,
            slot_sweep_range=args.slot_sweep_range,
            slot_sweep_policy=args.slot_sweep_policy,
            slot_sweep_expert_bytes=args.slot_sweep_expert_bytes,
            slot_sweep_bandwidth_gbps=args.slot_sweep_bandwidth_gbps,
        )
    return manifest


def _write_analysis_report(
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    run_slot_sweep: bool = False,
    slot_sweep_range: str = "8:64:8",
    slot_sweep_policy: str = "lru",
    slot_sweep_expert_bytes: int = 14_680_064,
    slot_sweep_bandwidth_gbps: float = 24.0,
):
    analyzer = _load_analyzer()
    reports = []
    report_scenarios = []
    for scenario in manifest["scenarios"]:
        outputs = scenario.get("profiler_outputs") or []
        if not outputs:
            continue
        reports.append(
            analyzer.analyze_profile(
                scenario["phase"],
                outputs[-1],
                scenario.get("benchmark_json"),
                scenario.get("sew_moe_trace_jsonl"),
                scenario.get("sew_moe_profile_jsonl"),
            ))
        report_scenarios.append(scenario)
    if not reports:
        return
    (output_dir / "ascend_moe_profile_report.json").write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ascend_moe_profile_report.md").write_text(
        analyzer.render_markdown(reports),
        encoding="utf-8",
    )
    slot_sweep_results = {}
    if run_slot_sweep:
        slot_sweep_results = _run_slot_sweeps(
            report_scenarios,
            output_dir,
            slot_sweep_range=slot_sweep_range,
            slot_sweep_policy=slot_sweep_policy,
            slot_sweep_expert_bytes=slot_sweep_expert_bytes,
            slot_sweep_bandwidth_gbps=slot_sweep_bandwidth_gbps,
        )
    p1_plan = _extract_p1_plan(reports, slot_sweep_results=slot_sweep_results)
    if p1_plan["plans"]:
        (output_dir / "sew_moe_p1_plan.json").write_text(
            json.dumps(p1_plan, indent=2),
            encoding="utf-8",
        )


def _run_slot_sweeps(
    scenarios: list[dict[str, Any]],
    output_dir: Path,
    *,
    slot_sweep_range: str,
    slot_sweep_policy: str,
    slot_sweep_expert_bytes: int,
    slot_sweep_bandwidth_gbps: float,
) -> dict[str, dict[str, Any]]:
    from tools.sew_offload.simulate_expert_slots import (
        load_trace,
        parse_slot_range,
    )
    from vllm_ascend.moe_offload.slot_simulator import ExpertSizeTable, SlotSimulator

    slot_values = parse_slot_range(slot_sweep_range)
    simulator = SlotSimulator(
        size_table=ExpertSizeTable(default_expert_bytes=slot_sweep_expert_bytes),
        host_to_hbm_bandwidth_gbps=slot_sweep_bandwidth_gbps,
    )
    results: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        trace_path = scenario.get("sew_moe_trace_jsonl")
        if not trace_path:
            continue
        trace = Path(trace_path)
        if not trace.exists():
            continue
        records = load_trace(trace)
        summaries = [
            simulator.replay(records, num_slots=num_slots, policy_name=slot_sweep_policy).to_jsonable()
            for num_slots in slot_values
        ]
        if not summaries:
            continue
        best = min(
            summaries,
            key=lambda item: (
                item["host_to_hbm_bytes"],
                item["miss_count"],
                item["num_slots"],
            ),
        )
        summary = {
            "trace": str(trace),
            "policy": slot_sweep_policy,
            "slot_range": slot_sweep_range,
            "recommended_num_slots": best["num_slots"],
            "recommended_host_to_hbm_bytes": best["host_to_hbm_bytes"],
            "recommended_miss_count": best["miss_count"],
            "recommended_prefetchable_miss_count": best["prefetchable_miss_count"],
            "recommended_exposed_miss_count": best["exposed_miss_count"],
            "recommended_prefetchable_host_to_hbm_bytes": best["prefetchable_host_to_hbm_bytes"],
            "recommended_exposed_host_to_hbm_bytes": best["exposed_host_to_hbm_bytes"],
            "sweep": summaries,
        }
        scenario_dir = output_dir / str(scenario.get("name") or scenario.get("phase") or "unknown")
        output_path = scenario_dir / f"slot_sweep_{slot_sweep_policy}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        phase = str(scenario.get("phase", ""))
        results[phase] = {
            "path": str(output_path),
            "summary": summary,
        }
    return results


def _extract_p1_plan(
    reports: list[dict[str, Any]],
    *,
    slot_sweep_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    slot_sweep_results = slot_sweep_results or {}
    plans = []
    for report in reports:
        decision = report.get("p1_decision") or {}
        plan: dict[str, Any] = {
            "phase": report.get("phase", ""),
            "target": decision.get("target", "UNKNOWN"),
        }
        if decision.get("compute_bucket_plan"):
            plan["compute_bucket_plan"] = decision["compute_bucket_plan"]
        if decision.get("slot_sweep_hint"):
            plan["slot_sweep_hint"] = decision["slot_sweep_hint"]
        sweep_result = slot_sweep_results.get(str(report.get("phase", "")))
        if sweep_result:
            plan["slot_sweep_result_json"] = sweep_result["path"]
            plan["slot_sweep_result"] = sweep_result["summary"]
        if len(plan) > 2:
            plans.append(plan)
    return {
        "version": 1,
        "plans": plans,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a three-window Ascend PyTorch Profiler suite for single-card non-offload "
            "Qwen3-30B-A3B MoE inference: mixed, prefill-heavy, and decode-heavy."
        ))
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--profile-url", default="http://127.0.0.1:8005")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--backend", default="openai-chat")
    parser.add_argument("--served-model-name", default="qwen3-30b-a3b")
    parser.add_argument("--tokenizer", default="/data/shared-models/Qwen3-30B-A3B")
    parser.add_argument(
        "--sharegpt-dataset",
        type=Path,
        default=Path("benchmarks/results/moe_offload_real_sharegpt_qwen3_30b_a3b/ShareGPT_prompt_le256_for_mlen512.json"),
    )
    parser.add_argument("--profiler-dir", type=Path, default=Path("vllm_profile"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bench-command", default="vllm bench serve")
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--num-warmups", type=int, default=0)
    parser.add_argument("--mixed-num-prompts", type=int, default=200)
    parser.add_argument("--mixed-output-len", type=int, default=64)
    parser.add_argument("--prefill-num-prompts", type=int, default=32)
    parser.add_argument("--prefill-input-len", type=int, default=2048)
    parser.add_argument("--prefill-output-len", type=int, default=1)
    parser.add_argument("--decode-num-prompts", type=int, default=32)
    parser.add_argument("--decode-input-len", type=int, default=128)
    parser.add_argument("--decode-output-len", type=int, default=256)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--stop-analyse-delay-sec", type=float, default=2.0)
    parser.add_argument("--skip-analyse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sew-moe-trace-path",
        type=Path,
        help=(
            "Optional JSONL path written by the running vLLM server via "
            "VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH. Per-scenario deltas are copied into the output directory."
        ),
    )
    parser.add_argument(
        "--sew-moe-profile-path",
        type=Path,
        help=(
            "Optional JSONL path written by the running vLLM server via "
            "VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH. Per-scenario deltas are copied into the output directory."
        ),
    )
    parser.add_argument(
        "--require-sew-moe-artifacts",
        action="store_true",
        help="Fail if configured SEW-MoE trace/profile paths do not produce per-scenario JSONL deltas.",
    )
    parser.add_argument(
        "--run-slot-sweep",
        action="store_true",
        help="After analysis, replay per-scenario SEW-MoE traces through the fixed-slot simulator.",
    )
    parser.add_argument(
        "--slot-sweep-range",
        default="8:64:8",
        help="Slot sweep range START:STOP[:STEP] used when --run-slot-sweep is set.",
    )
    parser.add_argument("--slot-sweep-policy", default="lru", choices=("lru", "sticky_layer_lru"))
    parser.add_argument("--slot-sweep-expert-bytes", type=int, default=14_680_064)
    parser.add_argument("--slot-sweep-bandwidth-gbps", type=float, default=24.0)
    return parser.parse_args()


def main() -> int:
    manifest = run_suite(parse_args())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
