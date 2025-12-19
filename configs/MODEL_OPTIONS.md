# 模型选择指南

## 推荐模型（按优先级）

### 1. Qwen1.5-MoE-A2.7B-Chat（推荐）⭐

**配置文件**: `configs/rtx4050_config.json`

**特点**:
- ✅ 较小的 MoE 模型（2.7B 参数）
- ✅ 适合 RTX 4050 (6-8GB 显存)
- ✅ 使用 4-bit 量化后约 2-3GB 显存
- ✅ 支持中文和多语言
- ✅ 在 HuggingFace 上可用

**模型路径**: `Qwen/Qwen1.5-MoE-A2.7B-Chat`

**使用**:
```bash
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_config.json
```

### 2. Mixtral-8x7B-Instruct（备选）

**配置文件**: `configs/rtx4050_mixtral_config.json`

**特点**:
- ✅ 广泛使用的 MoE 模型
- ✅ 性能较好
- ⚠️ 需要 4-bit 量化（约 5-6GB 显存）
- ⚠️ 可能接近 RTX 4050 显存上限

**模型路径**: `mistralai/Mixtral-8x7B-Instruct-v0.1`

**使用**:
```bash
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_mixtral_config.json
```

### 3. Qwen1.5-1.5B-Instruct（对比实验）

**配置文件**: `configs/rtx4050_qwen_dense_config.json`

**特点**:
- ✅ 小模型，显存占用低
- ⚠️ **注意**: 这是密集模型（非 MoE）
- ⚠️ 仅用于对比实验，验证 MoE 的必要性
- ⚠️ MoE 水印方法需要 MoE 架构

**模型路径**: `Qwen/Qwen1.5-1.5B-Instruct`

## 已废弃的模型

- ❌ `Qwen/Qwen1.5-MoE-A14B-Chat` - 已从 HuggingFace 移除

## 关于更小的 MoE 模型

**重要说明**: `Qwen1.5-MoE-A2.7B-Chat` 是目前**最小的常用 MoE 模型**。虽然理论上存在一些更小的 MoE 模型（如 MiniMind-MoE-640-120M 149M参数），但它们通常：

1. ❌ 不支持中文或功能受限
2. ❌ 可能不兼容当前的代码实现
3. ❌ 在 HuggingFace 上可能不可用或未维护

**建议**: 如果仍然遇到 OOM 问题，请使用 `rtx4050_ultra_low_memory_config.json` 配置，它会通过以下方式进一步降低显存占用：
- 限制 GPU 显存使用上限
- 减少使用的 MoE 层数
- 减少专家激活数量
- 减少 LSH 位数和池大小
- 减少生成长度

这些优化可以显著降低显存占用，同时保持模型的 MoE 架构特性。

## 如何选择模型

### 对于 RTX 4050 (6-8GB 显存)

**首选**: Qwen1.5-MoE-A2.7B-Chat
- 显存占用小
- 性能足够验证论文思路
- 下载速度快

**如果显存充足**: Mixtral-8x7B-Instruct
- 性能更好
- 更接近论文中的实验设置

### 显存优化建议

如果遇到 OOM（显存不足）：

### 方案 1: 使用超低显存配置（推荐）⭐

**配置文件**: `configs/rtx4050_ultra_low_memory_config.json`

这个配置已经做了最激进的优化：
- ✅ 限制 GPU 显存为 5GB（通过 `max_memory`）
- ✅ 只使用 2 层 MoE（layer 6-8）
- ✅ 减少 top-k 专家数为 2
- ✅ 减少 LSH 位数为 64
- ✅ 减少生成长度为 64 tokens
- ✅ 减少 pool_size 为 4

**使用**:
```bash
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_ultra_low_memory_config.json \
    --num_seeds 10
```

### 方案 2: 手动优化配置

1. **使用 4-bit 量化**（已在配置中）
2. **限制最大显存**:
   ```json
   {
     "max_memory": {
       "0": "5GiB"
     }
   }
   ```
3. **减少 MoE 层数**:
   ```json
   {
     "extractor_config": {
       "start_layer_idx": 6,
       "end_layer_idx": 8,  // 只使用 2 层
       "k": 2  // 减少 top-k 专家数
     }
   }
   ```
4. **减少生成长度**:
   ```json
   {
     "generation_config": {
       "max_new_tokens": 64  // 从 128 减少到 64
     }
   }
   ```
5. **减少 LSH 位数**:
   ```json
   {
     "encoder_config": {
       "num_bits": 64,  // 从 128 减少到 64
       "pool_size": 4  // 从 8 减少到 4
     }
   }
   ```

## 验证模型可用性

在运行实验前，可以验证模型是否可用：

```python
from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")
    print("✅ 模型可用")
except Exception as e:
    print(f"❌ 模型不可用: {e}")
```

## 使用国内镜像下载

确保设置了 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或在代码中（已自动设置）：
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

## 模型信息对比

| 模型 | 参数量 | 量化后显存 | 推荐度 | 备注 |
|------|--------|-----------|--------|------|
| Qwen1.5-MoE-A2.7B-Chat | 2.7B | ~2-3GB | ⭐⭐⭐⭐⭐ | 最适合 RTX 4050 |
| Qwen1.5-MoE-A2.7B-Chat (超低显存) | 2.7B | ~1.5-2GB | ⭐⭐⭐⭐⭐ | 使用超低显存配置，适合 4-6GB 显存 |
| Mixtral-8x7B-Instruct | 47B (激活~13B) | ~5-6GB | ⭐⭐⭐⭐ | 性能好，但显存紧张 |
| Qwen1.5-1.5B-Instruct | 1.5B | ~1GB | ⭐⭐ | 非 MoE，仅对比用 |

**注意**: `Qwen1.5-MoE-A2.7B-Chat` 是当前最小的常用 MoE 模型。如果仍然 OOM，请使用 `rtx4050_ultra_low_memory_config.json` 配置。

## 更新日志

- 2025-01-XX: 移除 Qwen1.5-MoE-A14B-Chat（已不可用）
- 2025-01-XX: 添加 Qwen1.5-MoE-A2.7B-Chat 作为推荐模型
- 2025-01-XX: 添加 Mixtral-8x7B-Instruct 作为备选
- 2025-01-XX: 添加 `rtx4050_ultra_low_memory_config.json` 超低显存配置，适合 4-6GB 显存

