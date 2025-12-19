"""
主实验脚本
运行完整的种子敏感性分析实验
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import argparse
import json
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

from moe_lsh.MoELSHWatermark import MoELSHWatermark, MoELSHWatermarkConfig
from experiments.geometric_metrics import GeometricQualityMetrics
from experiments.seed_evaluator import SeedEvaluator
from experiments.attack_methods import (
    HumanParaphraseAttack,
    LLMParaphraseAttack,
    BigramParaphraseAttack,
    B4ScrubbingAttack,
    MixedAttack,
    AttackEvaluator
)
from experiments.baseline_comparison import (
    KirchenbauerWatermark,
    BaselineComparator
)


def load_calibration_texts(data_path: Path, max_texts: int = 100) -> List[str]:
    """加载校准文本"""
    texts = []
    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
                if len(texts) >= max_texts:
                    break
    else:
        # 使用默认示例文本
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Watermarking techniques help identify AI-generated content.",
        ] * (max_texts // 3 + 1)
        texts = texts[:max_texts]
    
    return texts


def load_test_prompts(data_path: Optional[Path] = None) -> List[str]:
    """加载测试提示"""
    if data_path and data_path.exists():
        prompts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                prompts.append(line.strip())
        return prompts
    else:
        # 默认测试提示
        return [
            "Explain the concept of machine learning.",
            "What is the difference between supervised and unsupervised learning?",
            "Describe the process of neural network training.",
            "How does backpropagation work?",
            "What are the applications of deep learning?",
        ]


def run_seed_sensitivity_experiment(
    watermark: MoELSHWatermark,
    test_prompts: List[str],
    num_seeds: int = 50,
    output_dir: Path = Path("results")
):
    """运行种子敏感性实验"""
    print("=" * 60)
    print("Running Seed Sensitivity Experiment")
    print("=" * 60)
    
    # 初始化几何质量指标
    embedding_matrix = watermark.config.generation_model.get_input_embeddings().weight.detach().float()
    geometric_metrics = GeometricQualityMetrics(
        embedding_matrix=embedding_matrix,
        vocab_size=watermark.encoder.vocab_size,
        num_clusters=100,
        device=watermark.config.device
    )
    
    # 加载校准文本
    calibration_texts = load_calibration_texts(Path("data/calibration.txt"), max_texts=50)
    
    # 初始化种子评估器
    seed_evaluator = SeedEvaluator(
        watermark=watermark,
        geometric_metrics=geometric_metrics,
        calibration_texts=calibration_texts,
        device=watermark.config.device
    )
    
    # 生成种子列表
    seeds = seed_evaluator.generate_seeds(num_seeds=num_seeds)
    
    # 评估所有种子
    seed_results = seed_evaluator.evaluate_multiple_seeds(
        seeds=seeds,
        test_prompts=test_prompts,
        output_dir=output_dir / "seed_evaluation"
    )
    
    # 分析种子敏感性
    sensitivity_analysis = seed_evaluator.analyze_seed_sensitivity(seed_results)
    
    # 保存分析结果
    analysis_path = output_dir / "seed_sensitivity_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(sensitivity_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\nSeed Sensitivity Analysis saved to {analysis_path}")
    print(f"High quality seeds: {len(sensitivity_analysis['high_quality_seeds'])}")
    print(f"Low quality seeds: {len(sensitivity_analysis['low_quality_seeds'])}")
    
    return seed_results, sensitivity_analysis


def run_attack_robustness_experiment(
    watermark: MoELSHWatermark,
    watermarked_texts: List[str],
    output_dir: Path = Path("results")
):
    """运行攻击鲁棒性实验"""
    print("=" * 60)
    print("Running Attack Robustness Experiment")
    print("=" * 60)
    
    # 初始化攻击方法
    attack_methods = [
        BigramParaphraseAttack(replacement_prob=0.3),
        # 注意：LLM 和 B4 攻击需要 API 密钥
        # LLMParaphraseAttack(model_name="gpt-4", api_key=os.getenv("OPENAI_API_KEY")),
        # B4ScrubbingAttack(api_key=os.getenv("OPENAI_API_KEY")),
    ]
    
    # 混合攻击
    if len(attack_methods) > 1:
        mixed_attack = MixedAttack(attacks=attack_methods)
        attack_methods.append(mixed_attack)
    
    # 初始化攻击评估器
    attack_evaluator = AttackEvaluator(
        watermark_detector=watermark,
        attack_methods=attack_methods
    )
    
    # 评估鲁棒性
    robustness_results = attack_evaluator.evaluate_robustness(
        watermarked_texts=watermarked_texts,
        attack_names=[method.__class__.__name__ for method in attack_methods]
    )
    
    # 保存结果
    robustness_path = output_dir / "attack_robustness_results.json"
    with open(robustness_path, 'w', encoding='utf-8') as f:
        json.dump(robustness_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAttack Robustness Results saved to {robustness_path}")
    for attack_name, result in robustness_results.items():
        print(f"{attack_name}: Detection Rate = {result['detection_rate']:.3f}")
    
    return robustness_results


def run_baseline_comparison_experiment(
    watermark: MoELSHWatermark,
    test_prompts: List[str],
    output_dir: Path = Path("results")
):
    """运行基线方法对比实验"""
    print("=" * 60)
    print("Running Baseline Comparison Experiment")
    print("=" * 60)
    
    # 初始化基线方法
    baseline_methods = [
        KirchenbauerWatermark(
            model=watermark.config.generation_model,
            tokenizer=watermark.config.generation_tokenizer,
            delta=2.0,
            gamma=0.5
        )
    ]
    
    # 初始化对比器
    comparator = BaselineComparator(
        our_method=watermark,
        baseline_methods=baseline_methods
    )
    
    # 对比生成质量
    quality_results = comparator.compare_generation_quality(
        prompts=test_prompts
    )
    
    # 生成带水印文本用于鲁棒性对比
    watermarked_texts = []
    for prompt in test_prompts:
        try:
            watermarked_text = watermark.generate_watermarked_text(prompt)
            watermarked_texts.append(watermarked_text)
        except Exception as e:
            print(f"Error generating watermarked text: {e}")
    
    # 初始化攻击方法
    attack_methods = [BigramParaphraseAttack(replacement_prob=0.3)]
    
    # 对比检测鲁棒性
    robustness_results = comparator.compare_detection_robustness(
        watermarked_texts=watermarked_texts,
        attack_methods=attack_methods
    )
    
    # 保存结果
    comparison_path = output_dir / "baseline_comparison_results.json"
    comparison_results = {
        'generation_quality': quality_results,
        'detection_robustness': robustness_results
    }
    
    comparator.save_comparison_results(
        results=comparison_results,
        output_path=comparison_path
    )
    
    print(f"\nBaseline Comparison Results saved to {comparison_path}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="MoE LSH Watermark Seed Sensitivity Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to watermark config file")
    parser.add_argument("--num_seeds", type=int, default=50, help="Number of seeds to evaluate")
    parser.add_argument("--test_prompts_file", type=str, default=None, help="Path to test prompts file")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--experiment", type=str, choices=["all", "seed", "attack", "baseline"], 
                       default="all", help="Which experiment to run")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载水印配置
    print(f"Loading watermark from config: {args.config}")
    watermark = MoELSHWatermark(algorithm_config=args.config)
    
    # 加载测试提示
    test_prompts = load_test_prompts(
        Path(args.test_prompts_file) if args.test_prompts_file else None
    )
    
    # 运行实验
    if args.experiment in ["all", "seed"]:
        seed_results, sensitivity_analysis = run_seed_sensitivity_experiment(
            watermark=watermark,
            test_prompts=test_prompts,
            num_seeds=args.num_seeds,
            output_dir=output_dir
        )
    
    if args.experiment in ["all", "attack"]:
        # 生成一些带水印文本用于攻击测试
        watermarked_texts = []
        for prompt in test_prompts[:10]:  # 使用前10个提示
            try:
                watermarked_text = watermark.generate_watermarked_text(prompt)
                watermarked_texts.append(watermarked_text)
            except Exception as e:
                print(f"Error generating watermarked text: {e}")
        
        if watermarked_texts:
            robustness_results = run_attack_robustness_experiment(
                watermark=watermark,
                watermarked_texts=watermarked_texts,
                output_dir=output_dir
            )
    
    if args.experiment in ["all", "baseline"]:
        comparison_results = run_baseline_comparison_experiment(
            watermark=watermark,
            test_prompts=test_prompts,
            output_dir=output_dir
        )
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

