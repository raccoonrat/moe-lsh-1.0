#!/bin/bash
# RTX 4050 实验环境搭建脚本

echo "=========================================="
echo "MoE LSH Watermark 实验环境搭建"
echo "适用于 RTX 4050 (6-8GB 显存)"
echo "=========================================="

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到 NVIDIA GPU"
    exit 1
fi

echo "检测到的 GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 创建虚拟环境
echo ""
echo "创建 Python 虚拟环境..."
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装 PyTorch (CUDA 12.1)
echo ""
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖
echo ""
echo "安装基础依赖..."
pip install transformers accelerate bitsandbytes
pip install scipy scikit-learn numpy
pip install tqdm

# 安装可选依赖（用于实验）
echo ""
echo "安装实验依赖..."
pip install datasets  # 用于加载数据集
pip install rouge-score  # 用于评估
pip install nltk  # 用于文本处理

# 安装 Flash Attention (可选，节省显存)
echo ""
echo "尝试安装 Flash Attention..."
pip install flash-attn --no-build-isolation || echo "Flash Attention 安装失败，将使用标准注意力"

echo ""
echo "=========================================="
echo "环境搭建完成！"
echo "=========================================="
echo ""
echo "使用说明:"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 运行实验: python experiments/main_experiment.py --config configs/rtx4050_config.json"
echo ""

