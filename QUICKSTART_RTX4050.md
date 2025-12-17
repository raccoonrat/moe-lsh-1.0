# RTX 4050 快速开始指南

## 🚀 5 分钟快速验证

### 步骤 1: 环境搭建

**Windows:**
```cmd
scripts\setup_environment.bat
venv\Scripts\activate
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
source venv/bin/activate
```

### 步骤 2: 快速测试

```bash
python scripts/quick_test.py
```

这会测试：
- ✅ GPU 可用性
- ✅ 模型加载（使用量化）
- ✅ 水印功能

### 步骤 3: 运行最小实验

```bash
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_config.json \
    --num_seeds 5 \
    --output_dir results/quick_test
```

**预计时间**: 10-15 分钟  
**显存使用**: ~5-6GB

## 📋 完整实验流程

### 1. 准备配置文件

配置文件已创建：`configs/rtx4050_config.json`

**关键配置说明**:
- `load_in_4bit: true` - 使用 4-bit 量化节省显存
- `start_layer_idx: 6, end_layer_idx: 12` - 只使用 6 层 MoE（减少显存）
- `num_bits: 128` - LSH 位数（从 256 减少到 128）
- `max_new_tokens: 128` - 限制生成长度

### 2. 运行标准实验

```bash
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_config.json \
    --num_seeds 20 \
    --output_dir results/standard
```

**预计时间**: 1-2 小时  
**显存使用**: ~6-7GB

### 3. 查看结果

```python
import json

# 查看种子敏感性分析
with open("results/standard/seed_sensitivity_analysis.json", 'r') as f:
    analysis = json.load(f)

print(f"评估种子数: {analysis['num_seeds']}")
print(f"平均检测分数: {analysis['detection_score_stats']['mean']:.2f}")
print(f"高质量种子: {len(analysis['high_quality_seeds'])}")
```

## 🔧 显存优化技巧

### 如果遇到 OOM（显存不足）

1. **减少种子数量**
   ```bash
   --num_seeds 5  # 从 20 减少到 5
   ```

2. **减少 MoE 层数**（修改 config）
   ```json
   {
     "extractor_config": {
       "start_layer_idx": 8,
       "end_layer_idx": 10  // 只使用 2 层
     }
   }
   ```

3. **减少生成长度**
   ```json
   {
     "generation_config": {
       "max_new_tokens": 64  // 从 128 减少到 64
     }
   }
   ```

4. **使用更小的模型**
   - 尝试 `Qwen/Qwen2.5-1.5B`（密集模型，用于对比）
   - 或使用 `OpenMoE-8B`（如果可用）

## 📊 实验规模建议

| 规模 | 种子数 | 提示数 | 时间 | 显存 | 用途 |
|------|--------|--------|------|------|------|
| 快速验证 | 5 | 3 | 10分钟 | 5GB | 验证环境 |
| 标准实验 | 20 | 10 | 1-2小时 | 6GB | 论文实验 |
| 完整实验 | 50+ | 20+ | 4+小时 | 7GB | 深度分析 |

## 🎯 验证论文思路的关键实验

### 实验 1: 种子敏感性验证

**目标**: 验证不同种子导致性能差异

```bash
python experiments/memory_optimized_experiment.py \
    --config configs/rtx4050_config.json \
    --num_seeds 20 \
    --output_dir results/seed_sensitivity
```

**预期结果**:
- 检测分数方差较大（std > 1.0）
- 高质量种子和低质量种子有明显差异

### 实验 2: 几何指标验证

**目标**: 验证几何指标能预测种子质量

查看 `seed_evaluation_results.json`，检查：
- `split_entropy` 高的种子 → 检测分数是否也高？
- `pca_alignment` 高的种子 → 是否更稳定？

### 实验 3: 攻击鲁棒性（可选）

如果需要测试攻击鲁棒性：

```python
from experiments.attack_methods import BigramParaphraseAttack, AttackEvaluator

# 加载带水印文本
# 执行攻击
# 评估检测率
```

## 🐛 常见问题

### Q: 模型下载很慢

**A**: 使用 HuggingFace 镜像
```python
# 在代码中设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Q: 量化模型性能差

**A**: 
1. 尝试 8-bit（如果显存允许）
2. 使用 GPTQ 量化（更高质量）
3. 只量化部分层

### Q: 实验太慢

**A**:
1. 减少 `num_seeds`
2. 减少 `max_new_tokens`
3. 使用更少的 MoE 层
4. 减少 LSH 位数

## 📈 结果分析示例

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# 加载结果
with open("results/standard/seed_evaluation_results.json", 'r') as f:
    results = json.load(f)

# 提取数据
scores = [r['avg_detection_score'] for r in results.values()]
entropies = [r['geometric_metrics'].get('split_entropy', 0) 
             for r in results.values()]

# 绘制散点图
plt.scatter(entropies, scores)
plt.xlabel('Split Entropy')
plt.ylabel('Detection Score')
plt.title('Seed Quality: Entropy vs Detection Score')
plt.savefig('results/seed_quality_analysis.png')
```

## 🎓 下一步

1. **验证核心思路**: 先用 5-10 个种子快速验证
2. **分析结果**: 查看几何指标与检测性能的相关性
3. **优化参数**: 根据结果调整配置
4. **扩展实验**: 逐步增加实验规模

## 📚 相关文档

- `configs/README_RTX4050.md` - 详细配置说明
- `experiments/README.md` - 实验代码说明
- `EXPERIMENT_STRUCTURE.md` - 整体架构

## 💡 提示

- **首次运行**: 建议先用 `quick_test.py` 验证环境
- **监控显存**: 使用 `nvidia-smi` 或代码中的 `print_gpu_memory()`
- **保存中间结果**: 实验可能运行较久，定期保存结果
- **分批运行**: 如果显存不足，可以分批运行不同种子

---

**祝实验顺利！** 🚀

如有问题，请检查：
1. GPU 驱动和 CUDA 版本
2. 模型是否正确下载
3. 显存是否足够
4. 配置文件是否正确

