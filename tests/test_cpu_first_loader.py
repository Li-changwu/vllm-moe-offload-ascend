from types import SimpleNamespace

import torch

from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
from vllm_moe_offload_ascend.moe_offload.cpu_first_loader import (
    CPU_FIRST_MARKER,
    CPU_FIRST_PROCESSED_MARKER,
    maybe_create_unquantized_cpu_first_weights,
    maybe_process_unquantized_cpu_first_weights,
)
from vllm_moe_offload_ascend.moe_offload.host_store import HostExpertStore
from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime


class FakeMethod:
    def __init__(self):
        self.moe = SimpleNamespace(is_act_and_mul=True, has_bias=False)

    def _maybe_pad_weight(self, weight):
        return weight


class FakeRuntime:
    def __init__(self, config):
        self.config = config

    def should_use_fixed_slot_plan_for_layer(self, layer_id):
        return int(layer_id) == 7


class TinyLayer(torch.nn.Module):
    layer_id = 7


def test_cpu_first_create_weights_allocates_offloaded_experts_on_cpu():
    method = FakeMethod()
    layer = TinyLayer()
    config = MoeOffloadConfig(
        enabled=True,
        num_slots=2,
        cpu_first_load=True,
        pin_host_memory=False,
    )

    called = maybe_create_unquantized_cpu_first_weights(
        method,
        layer,
        runtime=FakeRuntime(config),
        num_experts=4,
        hidden_size=3,
        intermediate_size_per_partition=2,
        params_dtype=torch.float32,
        extra_weight_attrs={"weight_loader": lambda *args, **kwargs: None},
    )

    assert called is True
    assert getattr(layer, CPU_FIRST_MARKER) is True
    assert layer.w13_weight.device.type == "cpu"
    assert layer.w2_weight.device.type == "cpu"
    assert tuple(layer.w13_weight.shape) == (4, 4, 3)
    assert tuple(layer.w2_weight.shape) == (4, 3, 2)
    assert hasattr(layer.w13_weight, "weight_loader")
    assert hasattr(layer.w2_weight, "weight_loader")


def test_cpu_first_create_weights_skips_resident_or_disabled_layer():
    method = FakeMethod()
    layer = TinyLayer()
    disabled = MoeOffloadConfig(enabled=True, num_slots=2, cpu_first_load=False)

    called = maybe_create_unquantized_cpu_first_weights(
        method,
        layer,
        runtime=FakeRuntime(disabled),
        num_experts=4,
        hidden_size=3,
        intermediate_size_per_partition=2,
        params_dtype=torch.float32,
        extra_weight_attrs={},
    )

    assert called is False
    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")


def test_cpu_first_process_formats_weights_and_registers_without_host_clone():
    method = FakeMethod()
    layer = TinyLayer()
    config = MoeOffloadConfig(
        enabled=True,
        num_slots=2,
        cpu_first_load=True,
        pin_host_memory=False,
    )
    runtime = MoeOffloadRuntime(config)

    assert maybe_create_unquantized_cpu_first_weights(
        method,
        layer,
        runtime=runtime,
        num_experts=4,
        hidden_size=3,
        intermediate_size_per_partition=2,
        params_dtype=torch.float32,
        extra_weight_attrs={},
    )
    layer.w13_weight.data.copy_(
        torch.arange(layer.w13_weight.numel(), dtype=torch.float32).reshape(layer.w13_weight.shape)
    )
    layer.w2_weight.data.copy_(
        torch.arange(layer.w2_weight.numel(), dtype=torch.float32).reshape(layer.w2_weight.shape)
    )

    processed = maybe_process_unquantized_cpu_first_weights(
        method,
        layer,
        runtime=runtime,
    )

    assert processed is True
    assert getattr(layer, CPU_FIRST_PROCESSED_MARKER) is True
    assert tuple(layer.w13_weight.shape) == (4, 3, 4)
    assert tuple(layer.w2_weight.shape) == (4, 2, 3)
    assert runtime.is_layer_registered(7)
    bundle = runtime._host_store.get(7, 1)
    assert bundle.w13.data_ptr() == layer.w13_weight[1].data_ptr()
    assert bundle.w2.data_ptr() == layer.w2_weight[1].data_ptr()


def test_host_store_clone_tensors_false_adopts_cpu_parameter_views():
    layer = TinyLayer()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
        requires_grad=False,
    )
    store = HostExpertStore()

    store.register_layer(layer, clone_tensors=False)

    bundle = store.get(7, 2)
    assert bundle.w13.data_ptr() == layer.w13_weight[2].data_ptr()
    assert bundle.w2.data_ptr() == layer.w2_weight[2].data_ptr()
