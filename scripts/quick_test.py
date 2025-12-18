"""
快速测试脚本 - 验证环境配置是否正确
适用于 RTX 4050
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def test_gpu():
    """测试 GPU 可用性"""
    print("=" * 60)
    print("GPU 测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"✅ CUDA 版本: {torch.version.cuda}")
    print(f"✅ PyTorch 版本: {torch.__version__}")
    
    return True

def test_model_loading(config_path: str):
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("模型加载测试")
    print("=" * 60)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_name = config.get("model_name", "Qwen/Qwen1.5-MoE-A2.7B-Chat")
    print(f"尝试加载模型: {model_name}")
    
    try:
        # 配置量化
        quantization_config = None
        if config.get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("使用 4-bit 量化")
        
        # 处理 max_memory 配置：将字符串键转换为整数键（GPU设备号）
        max_memory = {}
        for key, value in config.get("max_memory", {}).items():
            # 尝试将键转换为整数（用于GPU设备号）
            try:
                max_memory[int(key)] = value
            except ValueError:
                # 保留非数字键（如 'cpu', 'disk'）
                max_memory[key] = value
        
        if max_memory:
            print(f"显存限制配置: {max_memory}")
        
        # 加载 tokenizer
        print("加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✅ Tokenizer 加载成功")
        
        # 加载模型
        print("加载模型（这可能需要几分钟）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            max_memory=max_memory if max_memory else None
        )
        
        print("✅ 模型加载成功")
        print(f"显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 测试生成
        print("\n测试生成...")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 生成测试成功")
        print(f"生成文本: {generated_text}")
        
        # 清理
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_watermark(config_path: str):
    """测试水印功能"""
    print("\n" + "=" * 60)
    print("水印功能测试")
    print("=" * 60)
    
    try:
        from experiments.memory_optimized_experiment import (
            load_model_with_quantization,
            create_watermark_config
        )
        from moe_lsh.MoELSHWatermark import MoELSHWatermark
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载模型
        model, tokenizer = load_model_with_quantization(
            model_name=config["model_name"],
            config=config
        )
        
        # 创建水印
        watermark_config = create_watermark_config(model, tokenizer, config)
        watermark = MoELSHWatermark(algorithm_config=watermark_config)
        
        print("✅ 水印初始化成功")
        
        # 测试生成
        test_prompt = "What is machine learning?"
        print(f"\n测试提示: {test_prompt}")
        
        watermarked_text = watermark.generate_watermarked_text(test_prompt)
        print(f"✅ 生成带水印文本成功")
        print(f"生成文本: {watermarked_text[:100]}...")
        
        # 测试检测
        detection_result = watermark.detect_watermark(watermarked_text)
        print(f"✅ 检测成功")
        print(f"检测分数: {detection_result.get('score', 0):.2f}")
        print(f"是否带水印: {detection_result.get('is_watermarked', False)}")
        
        # 清理
        del model, tokenizer, watermark
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 水印测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("MoE LSH Watermark - 快速测试")
    print("适用于 RTX 4050")
    print("=" * 60)
    
    config_path = "configs/rtx4050_config.json"
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print("请先创建配置文件")
        return
    
    # 测试 GPU
    if not test_gpu():
        return
    
    # 测试模型加载
    if not test_model_loading(config_path):
        print("\n建议:")
        print("1. 检查网络连接（需要下载模型）")
        print("2. 尝试使用更小的模型")
        print("3. 检查显存是否足够")
        return
    
    # 测试水印
    if not test_watermark(config_path):
        print("\n建议:")
        print("1. 检查 moe_lsh 模块是否正确安装")
        print("2. 检查模型是否支持 MoE 架构")
        return
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("可以开始运行完整实验了")
    print("=" * 60)
    print("\n运行命令:")
    print("python experiments/memory_optimized_experiment.py \\")
    print("    --config configs/rtx4050_config.json \\")
    print("    --num_seeds 10")

if __name__ == "__main__":
    main()

