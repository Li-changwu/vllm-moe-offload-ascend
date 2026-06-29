import sys
from types import SimpleNamespace

import vllm_moe_offload_ascend


def test_b2_wave_profile_summary_reports_overlap_and_stage_breakdown():
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _summarize_b2_wave_profiles,
    )

    summary = _summarize_b2_wave_profiles(
        [
            {
                "stage_mode": "main_slot_hit",
                "hits": 2,
                "misses": 0,
                "tokens": 17,
                "pairs": 17,
                "h2d_bytes": 0,
                "d2d_bytes": 0,
                "stage_issue_ms": 0.1,
                "stage_wait_ms": 0.0,
                "mlp_ms": 3.0,
                "gmm_ms": 2.0,
                "issued_before_compute": False,
                "issued_before_microbatch_materialize": False,
            },
            {
                "stage_mode": "async_double_buffer",
                "hits": 0,
                "misses": 2,
                "tokens": 23,
                "pairs": 23,
                "h2d_bytes": 1024,
                "d2d_bytes": 0,
                "stage_issue_ms": 0.2,
                "stage_wait_ms": 4.5,
                "mlp_ms": 5.0,
                "gmm_ms": 4.0,
                "prefetch_before_compute_count": 1,
                "prefetch_after_compute_count": 2,
                "issued_before_compute": True,
                "issued_before_microbatch_materialize": True,
                "issue_end_to_compute_ms": 6.0,
            },
            {
                "stage_mode": "async_double_buffer",
                "hits": 1,
                "misses": 1,
                "tokens": 11,
                "pairs": 11,
                "h2d_bytes": 512,
                "d2d_bytes": 256,
                "stage_issue_ms": 0.3,
                "stage_wait_ms": 1.5,
                "mlp_ms": 2.0,
                "gmm_ms": 1.0,
                "issued_before_compute": True,
            },
        ],
        layer_scatter_ms=12.75,
    )

    assert summary["wave_count"] == 3
    assert summary["hit_only_waves"] == 1
    assert summary["miss_only_waves"] == 1
    assert summary["mixed_waves"] == 1
    assert summary["main_slot_hit_waves"] == 1
    assert summary["staged_waves"] == 2
    assert summary["issued_before_compute_waves"] == 2
    assert summary["issued_before_microbatch_materialize_waves"] == 1
    assert summary["prefetch_before_compute_issues"] == 1
    assert summary["prefetch_after_compute_issues"] == 2
    assert summary["tokens"] == 51
    assert summary["pairs"] == 51
    assert summary["hits"] == 3
    assert summary["misses"] == 3
    assert summary["h2d_bytes"] == 1536
    assert summary["d2d_bytes"] == 256
    assert summary["stage_issue_ms"] == 0.6
    assert summary["stage_wait_ms"] == 6.0
    assert summary["mlp_ms"] == 10.0
    assert summary["gmm_ms"] == 7.0
    assert summary["layer_scatter_ms"] == 12.75
    assert summary["max_issue_end_to_compute_ms"] == 6.0
    assert summary["max_stage_wait_ms"] == 4.5


def test_register_aliases_plugin_modules_under_vllm_ascend_namespace(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")

    vllm_moe_offload_ascend.register()

    import vllm_ascend
    import vllm_moe_offload_ascend.moe_offload as plugin_pkg
    import vllm_moe_offload_ascend.moe_offload.prefill_residency as prefill_residency
    from vllm_ascend.moe_offload.runtime import get_moe_offload_runtime

    assert sys.modules["vllm_ascend.moe_offload"] is plugin_pkg
    assert sys.modules["vllm_ascend.moe_offload.prefill_residency"] is prefill_residency
    assert vllm_ascend.moe_offload is plugin_pkg
    assert get_moe_offload_runtime.__module__ == "vllm_moe_offload_ascend.moe_offload.runtime"


def test_register_aliases_sew_custom_op_modules(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")

    vllm_moe_offload_ascend.register()

    import vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op as stage_op
    import vllm_moe_offload_ascend.ops.fused_moe.moe_router_op as router_op
    import vllm_ascend.ops.fused_moe as ascend_fused_moe

    assert sys.modules["vllm_ascend.ops.fused_moe.moe_offload_stage_op"] is stage_op
    assert sys.modules["vllm_ascend.ops.fused_moe.moe_router_op"] is router_op
    assert ascend_fused_moe.moe_offload_stage_op is stage_op
    assert ascend_fused_moe.moe_router_op is router_op


def test_register_rebinds_already_imported_hook_globals(monkeypatch):
    import vllm_ascend.ops.fused_moe.fused_moe as fused_moe
    import vllm_ascend.ops.fused_moe.moe_comm_method as moe_comm_method
    import vllm_ascend.ops.fused_moe.token_dispatcher as token_dispatcher

    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")

    vllm_moe_offload_ascend.register()

    assert fused_moe.get_moe_offload_runtime.__module__ == (
        "vllm_moe_offload_ascend.moe_offload.runtime"
    )
    assert moe_comm_method.get_moe_offload_runtime.__module__ == (
        "vllm_moe_offload_ascend.moe_offload.runtime"
    )
    assert moe_comm_method.MoeOffloadDecisionPath.__module__ == (
        "vllm_moe_offload_ascend.moe_offload.runtime"
    )
    assert token_dispatcher.get_moe_pipeline_profiler.__module__ == (
        "vllm_moe_offload_ascend.moe_offload.pipeline"
    )


def test_seam_forward_prefill_resident_uses_native_fused_moe(monkeypatch):
    import torch

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeRunner:
        _ascend_moe_offload_seam_patch = False

        def _select_forward(self):
            raise AssertionError("original select_forward should not be used")

    fake_fused_moe = SimpleNamespace(AscendMoERunner=FakeRunner)
    patch_fused_moe._patch_ascend_moe_runner(fake_fused_moe)

    events = []

    class FakeRuntime:
        config = SimpleNamespace(
            offload_stage_seam=True,
            gmm_profile_path="/tmp/test-prefill-resident-native.jsonl",
        )

        def __init__(self):
            self.resident_checks = []

        def is_resident_layer(self, layer_id):
            self.resident_checks.append(int(layer_id))
            return int(layer_id) == 3

        def _record_profile_event(self, name, *, layer_id, start, payload):
            events.append(
                {
                    "name": name,
                    "layer_id": layer_id,
                    "payload": payload,
                    "start_type": type(start).__name__,
                }
            )

    runtime = FakeRuntime()
    vllm_moe_offload_ascend.register()
    import vllm_ascend.moe_offload.runtime as runtime_mod

    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: True)
    monkeypatch.setattr(
        runtime_mod,
        "get_moe_offload_runtime",
        lambda: runtime,
    )

    runner = FakeRunner()
    runner._seam_active = True
    runner._seam_layer_id = 3
    runner._seam_num_logical_experts = 128
    runner.layer_name = "model.layers.3.mlp.experts"

    hidden_states = torch.empty((4, 16), dtype=torch.float32)
    router_logits = torch.empty((4, 128), dtype=torch.float32)
    expected = torch.empty_like(hidden_states)
    calls = []

    def fake_moe_forward(hidden, logits, shared_experts_input, input_ids, layer_name):
        calls.append(("moe_forward", layer_name, tuple(hidden.shape)))
        return expected

    def forbidden_op(*args, **kwargs):
        raise AssertionError("resident Prefill must bypass router/stage/mlp seam")

    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_forward",
        fake_moe_forward,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_router_indirect",
        forbidden_op,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_offload_stage",
        forbidden_op,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_mlp",
        forbidden_op,
        raising=False,
    )

    out = runner._seam_forward_entry(
        hidden_states,
        router_logits,
        shared_experts_input=None,
        input_ids=None,
        layer_name="ignored.layer.name",
    )

    assert out is expected
    assert runtime.resident_checks == [3]
    assert calls == [("moe_forward", "ignored.layer.name", (4, 16))]
    assert events == [
        {
            "name": "prefill_resident_native",
            "layer_id": 3,
            "payload": {"n_tokens": 4, "path": "native_fused_moe"},
            "start_type": "float",
        }
    ]


def test_b2_prefill_skips_resident_layer_before_route_stats(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeCommMethod:
        _ascend_moe_offload_runtime_patch = False

        def fused_experts(self, fused_experts_input):
            return "native"

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            raise AssertionError("original offload plan should not be used")

    class FakeRuntime:
        config = SimpleNamespace(
            graph_compatible_offload=False,
            b2_wave_prefill=True,
            gmm_profile_path="/tmp/test-prefill-resident-native-comm.jsonl",
        )

        def __init__(self):
            self.route_stats_calls = 0
            self.fixed_slot_checks = 0
            self.events = []

        def is_resident_layer(self, layer_id):
            return int(layer_id) == 7

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            self.fixed_slot_checks += 1
            return True

        def consume_prefill_route_stats_record(self, **kwargs):
            self.route_stats_calls += 1
            raise AssertionError("resident layer must skip route stats")

        def _record_profile_event(self, name, *, layer_id, start, payload):
            self.events.append(
                {
                    "name": name,
                    "layer_id": int(layer_id),
                    "payload": payload,
                    "start_type": type(start).__name__,
                }
            )

    runtime = FakeRuntime()
    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: True)
    monkeypatch.setattr(
        runtime_impl,
        "get_moe_offload_runtime",
        lambda: runtime,
    )

    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=SimpleNamespace,
        MoEFusedExpertsInput=SimpleNamespace,
        MoEWeights=SimpleNamespace,
        MoEOffloadParams=SimpleNamespace,
        MoERoutingParams=SimpleNamespace,
        setup_moe_comm_method=None,
    )
    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    comm = FakeCommMethod()
    TokenDispatcherWithAllGather = type("TokenDispatcherWithAllGather", (), {})
    comm.token_dispatcher = TokenDispatcherWithAllGather()
    comm._run_b2_wave_prefill = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("resident layer must not run B2")
    )
    fused_experts_input = SimpleNamespace(
        hidden_states=torch.empty((4, 16)),
        topk_ids=torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]]),
        offload=SimpleNamespace(enabled=True, layer_id=7),
    )

    assert comm._maybe_run_b2_wave_prefill(
        fused_experts_input,
        before_dispatch_evt=None,
    ) is None
    assert comm.fused_experts(fused_experts_input) == "native"
    assert runtime.route_stats_calls == 0
    assert runtime.fixed_slot_checks == 0
    assert runtime.events == [
        {
            "name": "prefill_resident_native",
            "layer_id": 7,
            "payload": {
                "n_tokens": 4,
                "path": "native_fused_moe",
                "entry": "comm_method",
            },
            "start_type": "float",
        }
    ]


def test_b2_does_not_treat_multi_request_decode_as_prefill(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeCommMethod:
        _ascend_moe_offload_runtime_patch = False

        def fused_experts(self, fused_experts_input):
            return "native"

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            return fused_experts_input

    class FakeRuntime:
        config = SimpleNamespace(
            graph_compatible_offload=False,
            b2_wave_prefill=True,
            gmm_profile_path="",
        )

        def is_resident_layer(self, layer_id):
            raise AssertionError("decode path must not run prefill resident check")

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            raise AssertionError("decode path must not inspect B2 fixed-slot plan")

        def consume_prefill_route_stats_record(self, **kwargs):
            raise AssertionError("decode path must not consume prefill route stats")

    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: False)
    monkeypatch.setattr(
        runtime_impl,
        "get_moe_offload_runtime",
        lambda: FakeRuntime(),
    )

    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=SimpleNamespace,
        MoEFusedExpertsInput=SimpleNamespace,
        MoEWeights=SimpleNamespace,
        MoEOffloadParams=SimpleNamespace,
        MoERoutingParams=SimpleNamespace,
        setup_moe_comm_method=None,
    )
    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    comm = FakeCommMethod()
    TokenDispatcherWithAllGather = type("TokenDispatcherWithAllGather", (), {})
    comm.token_dispatcher = TokenDispatcherWithAllGather()
    comm._run_b2_wave_prefill = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("decode batch must not run B2")
    )
    fused_experts_input = SimpleNamespace(
        hidden_states=torch.empty((4, 16)),
        topk_ids=torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]]),
        offload=SimpleNamespace(enabled=True, layer_id=7),
    )

    assert comm._maybe_run_b2_wave_prefill(
        fused_experts_input,
        before_dispatch_evt=None,
    ) is None
