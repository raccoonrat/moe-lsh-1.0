# RTX 4050 实验环境配置指南

## 硬件限制

RTX 4050 通常具有：
- **显存**: 6GB 或 8GB
- **计算能力**: 适合推理，不适合大规模训练

## 推荐的模型选择

### 选项 1: 量化 MoE 模型（推荐）

**Qwen2.5-MoE-A2.7B-Chat (4-bit 量化)**
- 原始大小: ~2.7B 参数
- 量化后显存: ~2-3GB
- 优点: 保持 MoE 架构，适合 RTX 4050
- 配置: 使用 `configs/rtx4050_config.json`
- **注意**: Qwen2.5-MoE-A14B-Chat 已不可用，已更换为 A2.7B 版本

**Mixtral-8x7B-Instruct (4-bit 量化)**
- 原始大小: ~47B 参数（但只有部分激活）
- 量化后显存: ~5-6GB
- 优点: 广泛使用的 MoE 模型
- 注意: 可能需要进一步优化

### 选项 2: 较小的 MoE 模型

**OpenMoE-8B** (如果可用)
- 参数: 8B
- 显存需求: ~4GB (FP16)
- 优点: 专为小显存设计

### 选项 3: 密集模型（作为对比）

**Llama-3-8B-Instruct**
- 参数: 8B
- 显存需求: ~4GB (FP16)
- 用途: 对比实验，验证 MoE 的必要性

## 环境搭建步骤

### Windows

```cmd
# 1. 运行环境搭建脚本
scripts\setup_environment.bat

# 2. 激活虚拟环境
venv\Scripts\activate

# 3. 运行实验
python experiments/memory_optimized_experiment.py --config configs/rtx4050_config.json --num_seeds 10
```

### Linux/Mac

```bash
# 1. 运行环境搭建脚本
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 运行实验
python experiments/memory_optimized_experiment.py --config configs/rtx4050_config.json --num_seeds 10
```

## 显存优化策略

### 1. 使用 4-bit 量化

```json
{
  "load_in_4bit": true,
  "torch_dtype": "float16"
}
```

### 2. 减少 MoE 层数

```json
{
  "extractor_config": {
    "start_layer_idx": 6,
    "end_layer_idx": 12,  // 只使用 6 层，而不是 17 层
    "k": 4  // 减少 top-k 专家数
  }
}
```

### 3. 减少 LSH 位数

```json
{
  "encoder_config": {
    "num_bits": 128,  // 从 256 减少到 128
    "pool_size": 8    // 从 10 减少到 8
  }
}
```

### 4. 限制生成长度

```json
{
  "generation_config": {
    "max_new_tokens": 128  // 限制生成长度
  }
}
```

### 5. 使用 Flash Attention

```json
{
  "use_flash_attention": true
}
```

## 实验规模建议

### 最小规模（验证思路）

- **种子数量**: 10-20
- **测试提示**: 3-5 个
- **每个种子生成次数**: 1-2 次
- **预计时间**: 30-60 分钟
- **显存使用**: ~5-6GB

### 中等规模（论文实验）

- **种子数量**: 50
- **测试提示**: 10-20 个
- **每个种子生成次数**: 3-5 次
- **预计时间**: 2-4 小时
- **显存使用**: ~6-7GB

### 完整规模（需要更多显存或时间）

- **种子数量**: 100+
- **测试提示**: 50+
- **每个种子生成次数**: 10+
- **预计时间**: 8+ 小时
- **建议**: 分批运行，或使用更大显存的 GPU

## 常见问题

### Q1: 显存不足错误 (OOM)

**解决方案**:
1. 减少 `num_seeds` 参数
2. 减少 `max_new_tokens`
3. 使用更小的模型
4. 启用 `load_in_4bit`
5. 减少 `start_layer_idx` 到 `end_layer_idx` 的范围

### Q2: 模型下载失败

**解决方案**:
1. 使用 HuggingFace 镜像站
2. 手动下载模型到本地
3. 使用 `local_files_only=True`

### Q3: 生成速度慢

**解决方案**:
1. 减少 `max_new_tokens`
2. 使用 Flash Attention
3. 减少 LSH 位数
4. 使用更少的 MoE 层

### Q4: 量化模型性能下降

**解决方案**:
1. 尝试 8-bit 量化（如果显存允许）
2. 使用更高质量的量化方法（GPTQ）
3. 只量化部分层

## 监控显存使用

在实验过程中监控显存：

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"显存使用: {allocated:.2f} GB / {reserved:.2f} GB")

# 在关键位置调用
print_gpu_memory()
```

## 实验脚本示例

### 快速验证（10 分钟）

```bash
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_config.json \
    --num_seeds 5 \
    --output_dir results/quick_test
```

### 标准实验（1-2 小时）

```bash
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_config.json \
    --num_seeds 20 \
    --output_dir results/standard
```

### 完整实验（分批运行）

```bash
# 第一批
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_config.json \
    --num_seeds 20 \
    --output_dir results/batch1

# 第二批（修改种子范围）
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_config.json \
    --num_seeds 20 \
    --output_dir results/batch2
```

## 结果分析

实验完成后，查看结果：

```python
import json
from pathlib import Path

# 加载分析结果
with open("results/seed_sensitivity_analysis.json", 'r') as f:
    analysis = json.load(f)

print(f"评估种子数: {analysis['num_seeds']}")
print(f"平均检测分数: {analysis['detection_score_stats']['mean']:.2f}")
print(f"高质量种子: {len(analysis['high_quality_seeds'])}")
print(f"低质量种子: {len(analysis['low_quality_seeds'])}")
```

## 下一步

1. **验证思路**: 先用最小规模验证代码和思路
2. **优化参数**: 根据初步结果调整配置
3. **扩展实验**: 逐步增加实验规模
4. **分析结果**: 使用几何指标分析种子质量

## 参考资源

- [HuggingFace Transformers 量化指南](https://huggingface.co/docs/transformers/quantization)
- [BitsAndBytes 文档](https://github.com/TimDettmers/bitsandbytes)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

