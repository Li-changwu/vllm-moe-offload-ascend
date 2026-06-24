#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

MOE_OFFLOAD_GB_ENV = "VLLM_ASCEND_MOE_OFFLOAD_GB"
_BYTES_PER_GIB = 1024**3
_DEFAULT_PREFETCH_GROUP_SIZE = 4
_DEFAULT_QWEN3_30B_A3B_CONFIG = {
    "hidden_size": 2048,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_hidden_layers": 48,
    "torch_dtype": "bfloat16",
}

_DEFAULT_ENGINE_ARGS = {
    "offload_backend": "prefetch",
    "offload_prefetch_step": 1,
    "offload_params": {"experts"},
    "cpu_offload_gb": 0,
    "cpu_offload_params": set(),
}

_DEFAULT_ENV_VARS = {
    "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
    "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": "8",
    "VLLM_ASCEND_MOE_OFFLOAD_POLICY": "deadline",
    "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD": "0",
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD": "8",
}
_RESIDENT_LAYER_IDS_ENV = "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"


def get_moe_offload_gb() -> float:
    raw_value = os.getenv(MOE_OFFLOAD_GB_ENV)
    if raw_value is None:
        return 0.0
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{MOE_OFFLOAD_GB_ENV} must be a non-negative number, got {raw_value!r}.") from exc
    if value < 0:
        raise ValueError(f"{MOE_OFFLOAD_GB_ENV} must be a non-negative number, got {raw_value!r}.")
    return value


def is_moe_offload_autoconfig_enabled() -> bool:
    return get_moe_offload_gb() > 0


class _MoeOffloadGbAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            value = float(values)
        except ValueError as exc:
            raise argparse.ArgumentError(self, f"{MOE_OFFLOAD_GB_ENV} must be a non-negative number.") from exc
        if value < 0:
            raise argparse.ArgumentError(self, f"{MOE_OFFLOAD_GB_ENV} must be a non-negative number.")
        os.environ[MOE_OFFLOAD_GB_ENV] = str(values)
        setattr(namespace, self.dest, value)


def register_moe_offload_cli_arg(parser: argparse.ArgumentParser) -> None:
    if "--ascend-moe-offload-gb" in parser._option_string_actions:
        return
    group = parser.add_argument_group("Ascend MoE Offload")
    group.add_argument(
        "--ascend-moe-offload-gb",
        dest="ascend_moe_offload_gb",
        default=None,
        metavar="GB",
        action=_MoeOffloadGbAction,
        help=(
            "Enable Ascend MoE expert offload with the vLLM PrefetchOffloader "
            "and fixed-slot MoE runtime. The value is the target expert-weight "
            "offload budget in GiB and is used to derive the prefetch offload "
            "layer fraction. 0 or omitted keeps the normal non-offloaded path. "
            "Do not combine with cpu_offload_gb/UVA."
        ),
    )


def _dtype_size_bytes(torch_dtype: Any) -> int:
    dtype = str(torch_dtype).lower().replace("torch.", "")
    if dtype in ("float32", "fp32"):
        return 4
    if dtype in ("float16", "fp16", "bfloat16", "bf16"):
        return 2
    if dtype in ("float8", "fp8", "int8"):
        return 1
    return 2


def _expert_layer_gb(model_config: dict[str, Any]) -> float:
    hidden_size = int(model_config["hidden_size"])
    moe_intermediate_size = int(model_config.get("moe_intermediate_size") or model_config["intermediate_size"])
    num_experts = int(model_config["num_experts"])
    dtype_size = _dtype_size_bytes(model_config.get("torch_dtype", "bfloat16"))
    # Fused routed experts hold gate/up (2 * I * H) and down (I * H) weights.
    layer_bytes = 3 * hidden_size * moe_intermediate_size * num_experts * dtype_size
    return layer_bytes / _BYTES_PER_GIB


def derive_prefetch_defaults(
    target_offload_gb: float,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if target_offload_gb <= 0:
        raise ValueError("target_offload_gb must be greater than 0")

    model_config = model_config or _DEFAULT_QWEN3_30B_A3B_CONFIG
    num_layers = int(model_config["num_hidden_layers"])
    group_size = min(_DEFAULT_PREFETCH_GROUP_SIZE, num_layers)
    num_groups = math.ceil(num_layers / group_size)
    layer_gb = _expert_layer_gb(model_config)
    target_layers_per_group = target_offload_gb / (layer_gb * num_groups)
    offload_num_in_group = min(group_size, max(1, round(target_layers_per_group)))
    offloaded_layer_ids = tuple(
        layer_id
        for layer_id in range(num_layers)
        if layer_id % group_size >= group_size - offload_num_in_group
    )
    offloaded_layer_id_set = set(offloaded_layer_ids)
    resident_layer_ids = tuple(
        layer_id for layer_id in range(num_layers) if layer_id not in offloaded_layer_id_set
    )
    estimated_offloaded_layers = len(offloaded_layer_ids)
    estimated_offloaded_gb = estimated_offloaded_layers * layer_gb

    return {
        "offload_group_size": group_size,
        "offload_num_in_group": offload_num_in_group,
        "offloaded_layer_ids": offloaded_layer_ids,
        "resident_layer_ids": resident_layer_ids,
        "estimated_offloaded_layers": estimated_offloaded_layers,
        "estimated_offloaded_gb": estimated_offloaded_gb,
        "expert_layer_gb": layer_gb,
    }


def _load_model_config_dict(engine_args: Any) -> dict[str, Any] | None:
    test_config = getattr(engine_args, "_ascend_moe_offload_model_config", None)
    if isinstance(test_config, dict):
        return test_config

    model = getattr(engine_args, "model", None)
    if not model:
        return None
    config_path = Path(str(model)) / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    return config if isinstance(config, dict) else None


def _field_is_unset(current_value: Any, default_value: Any) -> bool:
    if current_value is None:
        return True
    if isinstance(default_value, set):
        return set(current_value) == set()
    return current_value == default_value


def _set_engine_default(engine_args: Any, field_name: str, value: Any) -> None:
    current_value = getattr(engine_args, field_name, None)
    upstream_defaults = {
        "offload_backend": "auto",
        "offload_group_size": 0,
        "offload_num_in_group": 1,
        "offload_prefetch_step": 1,
        "offload_params": set(),
        "cpu_offload_gb": 0,
        "cpu_offload_params": set(),
    }
    if not hasattr(engine_args, field_name) or _field_is_unset(current_value, upstream_defaults[field_name]):
        setattr(engine_args, field_name, value.copy() if isinstance(value, set) else value)


def _raise_on_uva_conflict(engine_args: Any) -> None:
    cpu_offload_gb = float(getattr(engine_args, "cpu_offload_gb", 0) or 0)
    offload_backend = getattr(engine_args, "offload_backend", "auto")
    if cpu_offload_gb > 0 or offload_backend == "uva":
        raise ValueError(
            f"{MOE_OFFLOAD_GB_ENV} enables Ascend MoE offload through vLLM "
            "PrefetchOffloader. Remove cpu_offload_gb/UVA settings; they select "
            "the UVA offload backend, which is not the Ascend MoE offload path."
        )


def apply_moe_offload_defaults(engine_args: Any) -> bool:
    target_offload_gb = get_moe_offload_gb()
    if target_offload_gb <= 0:
        return False

    _raise_on_uva_conflict(engine_args)
    for env_name, value in _DEFAULT_ENV_VARS.items():
        os.environ.setdefault(env_name, value)
    prefetch_defaults = derive_prefetch_defaults(
        target_offload_gb,
        _load_model_config_dict(engine_args),
    )
    if _RESIDENT_LAYER_IDS_ENV not in os.environ:
        os.environ[_RESIDENT_LAYER_IDS_ENV] = ",".join(
            str(layer_id) for layer_id in prefetch_defaults["resident_layer_ids"]
        )
    engine_defaults = {
        **_DEFAULT_ENGINE_ARGS,
        "offload_group_size": prefetch_defaults["offload_group_size"],
        "offload_num_in_group": prefetch_defaults["offload_num_in_group"],
    }
    for field_name, value in engine_defaults.items():
        _set_engine_default(engine_args, field_name, value)

    setattr(engine_args, "_ascend_moe_offload_autoconfig_plan", prefetch_defaults)
    setattr(engine_args, "_ascend_moe_offload_autoconfig_applied", True)
    return True
