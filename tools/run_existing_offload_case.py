# SPDX-License-Identifier: Apache-2.0
"""Run one SEW-Offload baseline case with the existing vLLM offload stack."""

from __future__ import annotations

import argparse
import os
import time
import traceback

import torch
import torch_npu
from vllm import LLM, SamplingParams


def _csv_set(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--model", default="/data/shared-models/Qwen3-30B-A3B")
    parser.add_argument("--offload-backend", default="auto")
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--cpu-offload-params", default="")
    parser.add_argument("--offload-group-size", type=int, default=0)
    parser.add_argument("--offload-num-in-group", type=int, default=1)
    parser.add_argument("--offload-prefetch-step", type=int, default=1)
    parser.add_argument("--offload-params", default="")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_offload_params = _csv_set(args.cpu_offload_params)
    offload_params = _csv_set(args.offload_params)

    print(f"CASE {args.case_name}", flush=True)
    print(f"model {args.model}", flush=True)
    print(f"ASCEND_RT_VISIBLE_DEVICES {os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)
    print(f"torch {torch.__version__} torch_npu {torch_npu.__version__}", flush=True)
    print(
        f"npu_available {torch.npu.is_available()} device_count {torch.npu.device_count()}",
        flush=True,
    )
    print(
        "offload "
        f"backend={args.offload_backend} cpu_gb={args.cpu_offload_gb} "
        f"cpu_params={sorted(cpu_offload_params)} group_size={args.offload_group_size} "
        f"num_in_group={args.offload_num_in_group} prefetch_step={args.offload_prefetch_step} "
        f"params={sorted(offload_params)}",
        flush=True,
    )

    t0 = time.time()
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            trust_remote_code=args.trust_remote_code,
            dtype="bfloat16",
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_mb * 1024 * 1024,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_expert_parallel=False,
            seed=0,
            offload_backend=args.offload_backend,
            cpu_offload_gb=args.cpu_offload_gb,
            cpu_offload_params=cpu_offload_params,
            offload_group_size=args.offload_group_size,
            offload_num_in_group=args.offload_num_in_group,
            offload_prefetch_step=args.offload_prefetch_step,
            offload_params=offload_params,
        )
        print(f"LOAD_OK seconds={time.time() - t0:.3f}", flush=True)

        params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        gen_t0 = time.time()
        outputs = llm.generate(["Briefly explain mixture-of-experts models."], params)
        print(f"GENERATE_OK seconds={time.time() - gen_t0:.3f}", flush=True)
        for output in outputs:
            print("OUTPUT " + output.outputs[0].text.replace("\n", "\\n"), flush=True)
    except BaseException as exc:
        print(f"CASE_FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
