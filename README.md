# vllm-moe-offload-ascend

MoE（Mixture-of-Experts）Expert Offloading 插件，适用于 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) / vllm-ascend-hust（Ascend NPU 后端）。

以独立插件形式提供，通过 vllm 的 `vllm.platform_plugins` 机制自动加载，无需修改 vllm 或 vllm-ascend 源码。

---

## 功能

- **Expert Weight Offloading**：将不活跃的 MoE expert 权重卸载到 CPU，按需加载回 NPU，降低显存占用
- **Fixed-Slot Plan**：预分配 NPU slot，异步预取 expert 权重，减少传输等待
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

```bash
# 1. 克隆 vllm-hust（vllm 主体）
git clone https://github.com/vLLM-HUST/vllm-hust.git
pip install -e vllm-hust

# 2. 克隆并安装 vllm-ascend-hust（NPU 后端，含插件 hook 接缝）
#    需提前配置好 CANN 环境并设置 SOC_VERSION，例如：
#    export SOC_VERSION=ascend910b1  # Atlas A2
#    export SOC_VERSION=ascend910_9392  # Atlas A3
git clone https://github.com/vLLM-HUST/vllm-ascend-hust.git
pip install -e vllm-ascend-hust

# 3. 克隆并安装本插件
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
# 设置目标 offload 显存大小（GB），插件自动推导其余参数
export VLLM_ASCEND_MOE_OFFLOAD_GB=13.5

vllm serve <model> --trust-remote-code ...
```

### 手动指定参数

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=8
export VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline

vllm serve <model> \
  --offload-backend prefetch \
  --offload-prefetch-step 1 \
  ...
```

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
| `VLLM_ASCEND_MOE_OFFLOAD_GB` | 未设置（禁用） | 目标 offload 大小（GB），设置即启用 AutoConfig |
| `VLLM_ASCEND_MOE_OFFLOAD_ENABLED` | `1` | 是否启用 offload |
| `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS` | `8` | NPU 上预分配的 expert slot 数量 |
| `VLLM_ASCEND_MOE_OFFLOAD_POLICY` | `deadline` | 调度策略（`deadline` / `lru`） |
| `VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY` | `0` | 仅收集 routing trace，不做实际 offload |
| `VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME` | `1` | 启用分层运行时（resident + offload 混合） |
| `VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD` | `8` | 切换 slot cache / full weight 路径的 expert 数阈值 |
| `VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES` | `1` | Phase split 最大阶段数 |
| `VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS` | 自动推导 | 逗号分隔的常驻层 ID（不 offload） |

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
