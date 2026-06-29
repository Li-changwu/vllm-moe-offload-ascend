# vllm-moe-offload-ascend

MoE（Mixture-of-Experts）Expert Offloading 插件，适用于 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) / vllm-ascend-hust（Ascend NPU 后端）。

以独立插件形式提供，通过 vllm 的 `vllm.platform_plugins` 机制自动加载，无需修改 vllm 或 vllm-ascend 源码。

---

## 功能

- **Expert Weight Offloading**：将不活跃的 MoE expert 权重卸载到 CPU，按需加载回 NPU，降低显存占用
- **Fixed-Slot Plan**：预分配 NPU slot，按需加载 expert 权重，减少常驻 NPU 显存
- **Phase Split（MVP-D.11）**：将 MLP 计算按 hit/miss 拆分为多阶段，提升 overlap 效率
- **Trace & Profiling**：记录 routing 分布和 pipeline 耗时，用于调优
- **AutoConfig**：通过环境变量 `VLLM_ASCEND_MOE_OFFLOAD_GB` 自动配置 offload 参数

---

## 前提条件

| 依赖 | 说明 |
|------|------|
| Python ≥ 3.10 | |
| Ascend CANN | NPU 驱动环境，需在安装 vllm-ascend-hust 前配置好 |
| vllm-hust | vllm 主体（含 `vllm.platform_plugins` 支持） |
| vllm-ascend-hust | NPU 平台后端，需包含本插件的 hook 接缝（见下方说明） |

> **重要**：vllm-ascend-hust 需包含 `vllm_ascend/_moe_offload_null.py` 的改动（将 moe_offload 相关 import 改为 try/except 可选加载）。若使用上游官方 vllm-ascend，需先提 PR 合入该改动，或在本地手动 patch。

---

## 安装

### 方式一：从源码可编辑安装（研究开发推荐）

**步骤 1：安装 vllm-hust 和 vllm-ascend-hust**

通过 [vllm-hust-dev-hub](https://github.com/vLLM-HUST/vllm-hust-dev-hub) 一键初始化整个 workspace：

```bash
git clone git@github.com:vLLM-HUST/vllm-hust-dev-hub.git
```
这个命令会下载这个代码仓。然后运行：
```bash
cd vllm-hust-dev-hub
bash scripts/quickstart.sh
```
首次使用请选择菜单项1。脚本会自动下载整个workspace的代码。

**步骤 2：安装本插件**

```bash
git clone https://github.com/Li-changwu/vllm-moe-offload-ascend.git
pip install -e vllm-moe-offload-ascend
```

### 方式二：直接 pip 安装（vllm-hust 和 vllm-ascend-hust 已装）

```bash
pip install git+https://github.com/Li-changwu/vllm-moe-offload-ascend.git
```

安装后无需任何额外配置，vllm 启动时会自动发现并调用插件的 `register()` 函数。

---

## 使用

### 通过环境变量启用（AutoConfig，推荐）

```bash
# 设置目标 offload 显存大小（GiB），插件自动推导 resident layers 和 slot 容量
export VLLM_ASCEND_MOE_OFFLOAD_GB=14

vllm serve <model> --trust-remote-code ...
```

### 专家调试 override

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=32
export VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline
export VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1

vllm serve <model> --trust-remote-code ...
```

通常不需要设置 `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS`。当该变量未显式设置时，插件会根据 `VLLM_ASCEND_MOE_OFFLOAD_GB`、模型专家数/top-k、offloaded layer 数和服务配置在启动期自动推导 slot 容量；显式设置时会作为专家调试 override 保留。

以 Qwen3-30B-A3B 为例，`VLLM_ASCEND_MOE_OFFLOAD_GB=14` 通常会自动推导为 `32` slots；`28` 通常会自动推导为 `64` slots，用更多 slot 缩短 Prefill B2 wave 数。最终 slot 数还会受真实 HBM 可用预算和最小净显存收益约束。

不要和 vLLM 原生 weight offload 参数混用，例如 `--offload-backend prefetch`、`--offload-group-size`、`--cpu-offload-gb`。本插件通过 vllm-ascend-hust 的 MoE hooks 管理 expert offload，原生 offloader 是另一套路径。

如果设置 `VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1`，表示启用 graph-compatible 的 SEW fixed-slot 数据通路；这条路径会主动拒绝原生 prefetch offload 参数。未启用 SEW 数据通路时，AutoConfig 的普通分层路径仍可能通过 vLLM PrefetchOffloader 保留 high-fanout full-weight fallback，这是 legacy/layered 路径的一部分，不应和 SEW fixed-slot 实验混为同一组对比。

### CPU-first expert loading（实验）

```bash
export VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD=1
```

该开关用于大模型启动期：offloaded MoE 层的 expert 参数在 `create_weights` 阶段直接分配到 CPU host 内存，后续只按层短暂搬到 NPU 做 Ascend 格式化，再回落 CPU host store，避免“所有 expert 先完整加载到 NPU，再整体拷回 CPU”的启动峰值。当前第一阶段只覆盖 unquantized fixed-slot offloaded 层；resident 层、非 MoE 权重和暂未适配的量化 MoE 权重仍走原始加载路径。

### 验证插件已加载

启动日志中应出现：

```
Enabled Ascend MoE offload autoconfig from VLLM_ASCEND_MOE_OFFLOAD_GB. ...
```

### 禁用插件

不设置 `VLLM_ASCEND_MOE_OFFLOAD_GB`，或：

```bash
pip uninstall vllm-moe-offload-ascend
```

卸载后 vllm-ascend-hust 自动回落到 null stubs，功能不受影响。

---

## 环境变量参考

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `VLLM_ASCEND_MOE_OFFLOAD_GB` | 未设置（禁用） | 目标 offload 大小（GiB），设置即启用 AutoConfig |
| `VLLM_ASCEND_MOE_OFFLOAD_ENABLED` | `1` | 是否启用 offload |
| `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS` | 自动推导 | NPU 上预分配的 expert slot 数量；仅建议作为专家 override |
| `VLLM_ASCEND_MOE_OFFLOAD_POLICY` | `deadline` | 调度策略（`deadline` / `lru`） |
| `VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY` | `0` | 仅收集 routing trace，不做实际 offload |
| `VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME` | 普通路径 `1`，SEW 路径 `0` | 启用分层运行时（resident + offload 混合） |
| `VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD` | 跟随 `NUM_SLOTS` | 切换 slot cache / full weight 路径的 expert 数阈值 |
| `VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES` | `1` | Phase split 最大阶段数 |
| `VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS` | `0` | host store 注册后是否释放 offloaded layer 的原始 NPU expert 权重 |
| `VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS` | 自动推导 | 逗号分隔的常驻层 ID（不 offload） |
| `VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB` | 自动读取 | slot bank 可使用的 HBM 预算；通常由 `torch.npu.mem_get_info()` 和 `gpu_memory_utilization` 推导，设置该变量可显式覆盖 |
| `VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_RATIO` | `0.25` | 自动推导 slots 时至少保留的净显存收益比例 |
| `VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_GB` | 未设置 | 自动推导 slots 时至少保留的净显存收益 GiB 下限 |
| `VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE` | `0` | 启用 SEW router-stage-MLP graph-compatible fixed-slot 数据通路 |
| `VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD` | `0` | 实验开关；offloaded unquantized MoE expert 在初始化时直接落到 CPU host store，降低启动期 NPU 峰值 |
| `VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH` | `1` | SEW B2 Prefill 软件流水预取深度 |
| `VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT` | `2` | SEW B2 Prefill stage buffer 数 |
| `VLLM_ASCEND_MOE_GMM_PROFILE_PATH` | 未设置 | MoE/GMM profile JSONL 输出路径 |
| `VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH` | 未设置 | offload/stage profile JSONL 输出路径 |

---

## 与 vllm-ascend-hust 的关系

```
vllm
 └── vllm-ascend-hust (NPU 平台后端)
      ├── vllm_ascend/_moe_offload_null.py  ← 无插件时的空实现
      └── ops/fused_moe/*.py                ← try/except 导入 hook 点

vllm-moe-offload-ascend (本插件，可选)
 └── 注册 vllm.platform_plugins
      └── register() → apply_patches()
           └── 将 null stubs 替换为真实实现
```

插件通过 Python 包的 `vllm.platform_plugins` entry point 注册，vllm 在平台初始化时自动调用 `register()`，将 null stubs monkey-patch 为本包提供的真实实现。

---

## 工具脚本

`tools/` 目录包含研究用工具：

| 脚本 | 用途 |
|------|------|
| `collect_moe_trace.py` | 收集 routing 分布 trace |
| `run_minimal_offload_benchmark.py` | 最小化 offload benchmark |
| `run_fixed_slot_smoke.py` | fixed-slot 冒烟测试 |
| `simulate_expert_slots.py` | 模拟 slot 命中率 |
| `estimate_fixed_slot_memory.py` | 估算 slot 显存占用 |
| `analyze_layered_strategy.py` | 分析分层 offload 策略 |
| `moe_offload_timeline.py` | 可视化 offload timeline |

---

## License

Apache 2.0
