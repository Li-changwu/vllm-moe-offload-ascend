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
"""Monkey-patches that restore MoE Offloading hooks into vllm-ascend.

Called once by vllm_moe_offload_ascend.register() at plugin load time.
Replaces the null stubs (from vllm_ascend._moe_offload_null) with the
real implementations from this plugin package.
"""

from __future__ import annotations


def apply_patches() -> None:
    from vllm_moe_offload_ascend.moe_offload.runtime import get_moe_offload_runtime, MoeOffloadDecisionPath
    from vllm_moe_offload_ascend.moe_offload.pipeline import get_moe_pipeline_profiler

    # Patch fused_moe.py
    import vllm_ascend.ops.fused_moe.fused_moe as _fused_moe
    _fused_moe.get_moe_offload_runtime = get_moe_offload_runtime

    # Patch moe_comm_method.py
    import vllm_ascend.ops.fused_moe.moe_comm_method as _comm
    _comm.get_moe_offload_runtime = get_moe_offload_runtime
    _comm.MoeOffloadDecisionPath = MoeOffloadDecisionPath
    _comm.get_moe_pipeline_profiler = get_moe_pipeline_profiler

    # Patch token_dispatcher.py
    import vllm_ascend.ops.fused_moe.token_dispatcher as _td
    _td.get_moe_pipeline_profiler = get_moe_pipeline_profiler

    # Patch platform.py: autoconfig CLI arg registration
    _patch_platform_autoconfig()


def _patch_platform_autoconfig() -> None:
    """Register the MoE Offload CLI args that platform.py skips when plugin absent."""
    try:
        import vllm_ascend.platform as _platform
        from vllm_moe_offload_ascend.moe_offload.autoconfig import register_moe_offload_cli_arg

        _original_fn = _platform.NPUPlatform.pre_register_and_update.__func__

        @classmethod  # type: ignore[misc]
        def _patched(cls, parser=None):
            _original_fn(cls, parser)
            if parser is not None:
                register_moe_offload_cli_arg(parser)

        _platform.NPUPlatform.pre_register_and_update = _patched
    except Exception:
        pass  # best-effort: CLI arg registration is non-critical
