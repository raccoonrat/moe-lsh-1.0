# MoE LSH 水印种子敏感性实验代码

本目录包含用于评估 MoE LSH 水印种子敏感性的完整实验代码，基于论文《混合专家模型与语义哈希水印中的随机种子敏感性：基于高维几何与各向异性的数学机理深度分析》。

## 目录结构

```
experiments/
├── __init__.py                 # 模块初始化
├── geometric_metrics.py         # 几何质量指标模块
├── seed_evaluator.py            # 种子敏感性评估模块
├── attack_methods.py            # 攻击方法实现
├── baseline_comparison.py       # 基线方法对比模块
├── main_experiment.py           # 主实验运行脚本
└── README.md                    # 本文件
```

## 核心模块说明

### 1. geometric_metrics.py
实现论文中提出的三类几何质量指标：

- **语义簇分割熵 (Semantic Cluster Split Entropy)**: 衡量水印划分是否均匀切分语义聚类
- **Wasserstein 距离**: 量化水印对原始 Logits 分布的几何扭曲程度
- **投影方差与 PCA 对齐度**: 衡量投影向量是否捕捉到数据的主要变化方向

### 2. seed_evaluator.py
评估不同随机种子对水印性能的影响：

- 生成多个随机种子
- 评估每个种子的检测性能和几何质量
- 分析种子敏感性统计特征
- 识别高质量和低质量种子

### 3. attack_methods.py
实现各种攻击方法：

- **HumanParaphraseAttack**: 人类改写攻击
- **LLMParaphraseAttack**: 使用 LLM 进行改写
- **BigramParaphraseAttack**: Bigram 改写攻击
- **B4ScrubbingAttack**: B4 黑盒清洗
- **MixedAttack**: 混合攻击（结合多种方法）

### 4. baseline_comparison.py
与基线方法进行对比：

- **KirchenbauerWatermark**: 实现 Kirchenbauer et al. (2023) 的绿色词表水印
- 对比生成质量和检测鲁棒性

### 5. main_experiment.py
主实验脚本，整合所有模块：

- 种子敏感性实验
- 攻击鲁棒性实验
- 基线方法对比实验

## 使用方法

### 1. 准备配置文件

首先需要准备水印配置文件（JSON 格式），例如：

```json
{
  "extractor_config": {
    "start_layer_idx": 6,
    "end_layer_idx": 23,
    "use_top_k": true,
    "k": 6,
    "token_idx": -1
  },
  "encoder_config": {
    "num_bits": 256,
    "pool_size": 10,
    "secret_key": "default_key",
    "semantic_pools": true
  },
  "watermark_strength_delta": 2.0,
  "detection_z_threshold": 4.0
}
```

### 2. 运行完整实验

```bash
python experiments/main_experiment.py \
    --config path/to/watermark_config.json \
    --num_seeds 50 \
    --output_dir results \
    --experiment all
```

### 3. 运行特定实验

只运行种子敏感性实验：
```bash
python experiments/main_experiment.py \
    --config path/to/watermark_config.json \
    --experiment seed
```

只运行攻击鲁棒性实验：
```bash
python experiments/main_experiment.py \
    --config path/to/watermark_config.json \
    --experiment attack
```

只运行基线对比实验：
```bash
python experiments/main_experiment.py \
    --config path/to/watermark_config.json \
    --experiment baseline
```

### 4. 使用自定义测试提示

```bash
python experiments/main_experiment.py \
    --config path/to/watermark_config.json \
    --test_prompts_file data/test_prompts.txt \
    --output_dir results
```

## 输出结果

实验会在 `output_dir` 目录下生成以下文件：

1. **seed_evaluation_results.json**: 所有种子的详细评估结果
2. **seed_sensitivity_analysis.json**: 种子敏感性统计分析
3. **attack_robustness_results.json**: 攻击鲁棒性评估结果
4. **baseline_comparison_results.json**: 基线方法对比结果

## 示例输出

### seed_sensitivity_analysis.json
```json
{
  "num_seeds": 50,
  "detection_score_stats": {
    "mean": 5.23,
    "std": 2.15,
    "min": 0.45,
    "max": 8.92,
    "median": 5.10
  },
  "split_entropy_stats": {
    "mean": 0.65,
    "std": 0.12,
    "min": 0.23,
    "max": 0.89
  },
  "high_quality_seeds": [
    {
      "seed": "seed_42_15",
      "detection_score": 8.92,
      "split_entropy": 0.89
    }
  ],
  "low_quality_seeds": [...]
}
```

## 依赖项

- torch
- numpy
- scipy
- scikit-learn
- transformers
- tqdm
- openai (可选，用于 LLM 攻击)

## 注意事项

1. **API 密钥**: 如果使用 LLM 攻击方法（LLMParaphraseAttack, B4ScrubbingAttack），需要设置 `OPENAI_API_KEY` 环境变量。

2. **计算资源**: 种子敏感性实验需要评估多个种子，计算量较大。建议使用 GPU 加速。

3. **校准数据**: 几何质量指标需要校准数据集。如果没有提供，代码会使用默认示例文本。

4. **模型兼容性**: 确保使用的模型支持 MoE 架构，并且能够提取路由权重。

## 扩展

### 添加新的几何指标

在 `geometric_metrics.py` 的 `GeometricQualityMetrics` 类中添加新方法：

```python
def compute_new_metric(self, ...) -> float:
    """计算新的几何指标"""
    # 实现逻辑
    return metric_value
```

### 添加新的攻击方法

在 `attack_methods.py` 中继承 `ParaphraseAttack` 类：

```python
class NewAttack(ParaphraseAttack):
    def attack(self, text: str) -> str:
        # 实现攻击逻辑
        return attacked_text
```

### 添加新的基线方法

在 `baseline_comparison.py` 中继承 `BaselineWatermark` 类：

```python
class NewBaseline(BaselineWatermark):
    def generate_watermarked_text(self, prompt: str) -> str:
        # 实现生成逻辑
        return watermarked_text
    
    def detect_watermark(self, text: str) -> Dict:
        # 实现检测逻辑
        return detection_result
```

## 引用

如果使用本代码，请引用相关论文：

```bibtex
@article{moe_lsh_seed_sensitivity,
  title={混合专家模型与语义哈希水印中的随机种子敏感性：基于高维几何与各向异性的数学机理深度分析},
  author={...},
  journal={ICML 2025},
  year={2025}
}
```

