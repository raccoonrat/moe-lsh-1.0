"""
针对 RTX 4050 (6-8GB 显存) 优化的实验脚本
使用量化、梯度检查点等技术节省显存
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import gc
import argparse
import os
from typing import List, Dict, Optional
import json
from tqdm import tqdm

# 配置 HuggingFace 镜像（如果未设置）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
import accelerate

from moe_lsh.MoELSHWatermark import MoELSHWatermark, MoELSHWatermarkConfig
from experiments.geometric_metrics import GeometricQualityMetrics
from experiments.seed_evaluator import SeedEvaluator


def clear_gpu_cache():
    """清理 GPU 缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_with_quantization(
    model_name: str,
    config: Dict,
    device: str = "cuda"
) -> tuple:
    """
    使用量化加载模型以节省显存
    
    Args:
        model_name: 模型名称
        config: 配置字典
        device: 设备
        
    Returns:
        (model, tokenizer) 元组
    """
    print(f"加载模型: {model_name}")
    print(f"设备: {device}")
    
    # 清理显存
    clear_gpu_cache()
    
    # 配置量化
    quantization_config = None
    if config.get("load_in_4bit", False):
        print("使用 4-bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8  # 使用更紧凑的存储
        )
    elif config.get("load_in_8bit", False):
        print("使用 8-bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # 设置数据类型
    torch_dtype = getattr(torch, config.get("torch_dtype", "float16"), torch.float16)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 处理 max_memory 配置：将字符串键转换为整数键（GPU设备号）
    max_memory = {}
    use_max_memory = False
    if "max_memory" in config and config["max_memory"]:
        for key, value in config["max_memory"].items():
            # 尝试将键转换为整数（用于GPU设备号）
            try:
                max_memory[int(key)] = value
                use_max_memory = True
            except ValueError:
                # 保留非数字键（如 'cpu', 'disk'）
                max_memory[key] = value
                use_max_memory = True
    
    # 加载模型
    # 对于 4-bit 量化模型，不使用 device_map，让 transformers 自动处理
    # 这样可以避免 accelerate 库的显存管理问题
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch_dtype,  # 使用 dtype 代替已废弃的 torch_dtype
        "low_cpu_mem_usage": True,  # 减少 CPU 内存使用
    }
    
    # 对于 4-bit 量化，不设置 device_map，让量化库自动处理设备分配
    if not quantization_config:
        model_kwargs["device_map"] = "auto"
        print("使用自动设备映射")
    else:
        print("使用默认设备映射（4-bit 量化模式）")
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        # 4-bit 量化时不使用 max_memory，让 accelerate 自动管理
        print("使用 4-bit 量化，自动管理显存")
    
    # 只在非量化模式时设置 max_memory
    if use_max_memory and not config.get("load_in_4bit", False):
        model_kwargs["max_memory"] = max_memory
        print(f"显存限制配置: {max_memory}")
    
    # 使用 Flash Attention (如果可用)
    if config.get("use_flash_attention", False):
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("使用 Flash Attention 2")
        except:
            print("Flash Attention 不可用，使用标准注意力")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            print("❌ 显存不足，清理缓存后重试...")
            clear_gpu_cache()
            # 移除 device_map，使用默认方式
            if "device_map" in model_kwargs:
                del model_kwargs["device_map"]
                print("使用默认设备映射重试...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                raise
        else:
            raise
    
    # 启用梯度检查点以节省显存
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("启用梯度检查点")
    
    # 设置为评估模式
    model.eval()
    
    # 清理缓存
    clear_gpu_cache()
    
    print(f"模型加载完成，显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def create_watermark_config(
    model,
    tokenizer,
    base_config: Dict
) -> MoELSHWatermarkConfig:
    """创建水印配置对象"""
    
    # 构建配置字典
    watermark_config_dict = {
        "generation_model": model,
        "generation_tokenizer": tokenizer,
        "device": base_config.get("device", "cuda"),
        "vocab_size": len(tokenizer),
        "extractor_config": base_config.get("extractor_config", {}),
        "encoder_config": base_config.get("encoder_config", {}),
        "watermark_strength_delta": base_config.get("watermark_strength_delta", 2.0),
        "detection_z_threshold": base_config.get("detection_z_threshold", 4.0),
        "gen_kwargs": base_config.get("generation_config", {})
    }
    
    # 创建配置对象（直接传递字典）
    config = MoELSHWatermarkConfig(config_dict=watermark_config_dict)
    
    return config


def run_minimal_seed_experiment(
    watermark: MoELSHWatermark,
    test_prompts: List[str],
    num_seeds: int = 10,
    output_dir: Path = Path("results")
):
    """
    运行最小规模的种子敏感性实验（适配小显存）
    """
    print("=" * 60)
    print("运行种子敏感性实验（优化版）")
    print(f"评估种子数量: {num_seeds}")
    print("=" * 60)
    
    # 初始化几何质量指标
    embedding_matrix = watermark.config.generation_model.get_input_embeddings().weight.detach().float()
    
    # 使用较小的聚类数量以节省内存
    geometric_metrics = GeometricQualityMetrics(
        embedding_matrix=embedding_matrix,
        vocab_size=watermark.encoder.vocab_size,
        num_clusters=50,  # 减少聚类数量
        device=watermark.config.device
    )
    
    # 加载校准文本（少量即可）
    calibration_texts = test_prompts[:5]  # 只使用前5个作为校准
    
    # 初始化种子评估器
    seed_evaluator = SeedEvaluator(
        watermark=watermark,
        geometric_metrics=geometric_metrics,
        calibration_texts=calibration_texts,
        device=watermark.config.device
    )
    
    # 生成种子列表
    seeds = seed_evaluator.generate_seeds(num_seeds=num_seeds)
    
    # 逐个评估种子（避免内存积累）
    all_results = {}
    for seed in tqdm(seeds, desc="评估种子"):
        try:
            clear_gpu_cache()
            result = seed_evaluator.evaluate_single_seed(
                seed=seed,
                test_prompts=test_prompts[:3],  # 每个种子只用3个提示
                num_generations=1  # 每个提示只生成1次
            )
            all_results[seed] = result
        except Exception as e:
            print(f"评估种子 {seed} 时出错: {e}")
            continue
        finally:
            clear_gpu_cache()
    
    # 分析结果
    sensitivity_analysis = seed_evaluator.analyze_seed_sensitivity(all_results)
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "seed_evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    analysis_path = output_dir / "seed_sensitivity_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(sensitivity_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存:")
    print(f"  - {results_path}")
    print(f"  - {analysis_path}")
    print(f"\n高质量种子数: {len(sensitivity_analysis['high_quality_seeds'])}")
    print(f"低质量种子数: {len(sensitivity_analysis['low_quality_seeds'])}")
    
    return all_results, sensitivity_analysis


def main():
    parser = argparse.ArgumentParser(
        description="RTX 4050 优化的种子敏感性实验"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rtx4050_config.json",
        help="配置文件路径（默认使用 Qwen1.5-MoE-A2.7B-Chat）"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="评估的种子数量（建议 10-20，显存有限）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="输出目录"
    )
    parser.add_argument(
        "--test_prompts_file",
        type=str,
        default=None,
        help="测试提示文件（可选）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"使用配置: {config_path}")
    print(f"模型: {config.get('model_name', 'N/A')}")
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("错误: 未检测到 CUDA")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载模型
    try:
        model, tokenizer = load_model_with_quantization(
            model_name=config["model_name"],
            config=config,
            device=config.get("device", "cuda")
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("\n建议:")
        print("1. 检查模型名称是否正确")
        print("2. 尝试使用更小的模型或量化版本")
        print("3. 检查网络连接（需要下载模型）")
        return
    
    # 创建水印配置
    try:
        watermark_config = create_watermark_config(model, tokenizer, config)
        watermark = MoELSHWatermark(algorithm_config=watermark_config)
    except Exception as e:
        print(f"水印初始化失败: {e}")
        return
    
    # 加载测试提示
    if args.test_prompts_file and Path(args.test_prompts_file).exists():
        with open(args.test_prompts_file, 'r', encoding='utf-8') as f:
            test_prompts = [line.strip() for line in f if line.strip()]
    else:
        # 默认测试提示（简短，节省显存）
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks.",
            "Describe deep learning.",
        ]
    
    print(f"\n测试提示数量: {len(test_prompts)}")
    
    # 运行实验
    output_dir = Path(args.output_dir)
    try:
        results, analysis = run_minimal_seed_experiment(
            watermark=watermark,
            test_prompts=test_prompts,
            num_seeds=args.num_seeds,
            output_dir=output_dir
        )
        
        print("\n" + "=" * 60)
        print("实验完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        clear_gpu_cache()
        del model, tokenizer, watermark
        clear_gpu_cache()


if __name__ == "__main__":
    main()

