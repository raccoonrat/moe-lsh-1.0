#!/bin/bash
# RTX 4050 实验环境搭建脚本 (使用 Conda + 国内镜像)

set -e  # 遇到错误立即退出

echo "=========================================="
echo "MoE LSH Watermark 实验环境搭建"
echo "适用于 RTX 4050 (6-8GB 显存)"
echo "使用 Conda + 国内镜像源"
echo "=========================================="

# 检查 Conda
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 Conda"
    echo "请先安装 Miniconda 或 Anaconda"
    echo "下载地址: https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/"
    exit 1
fi

echo "检测到的 Conda 版本:"
conda --version

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: 未检测到 NVIDIA GPU，将使用 CPU 模式"
else
    echo "检测到的 GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# 配置 Conda 国内镜像源
echo ""
echo "配置 Conda 国内镜像源（清华镜像）..."
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

# 配置 pip 国内镜像源
echo ""
echo "配置 pip 国内镜像源..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
EOF
echo "✅ pip 镜像源已配置（清华镜像）"

# 创建 Conda 环境
ENV_NAME="moe-lsh-watermark"
PYTHON_VERSION="3.10"

echo ""
echo "创建 Conda 环境: $ENV_NAME (Python $PYTHON_VERSION)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "环境 $ENV_NAME 已存在，是否删除并重新创建? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    else
        echo "使用现有环境"
    fi
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# 激活环境
echo ""
echo "激活 Conda 环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 升级 pip
echo ""
echo "升级 pip..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 PyTorch (CUDA 12.1) - 使用清华镜像
echo ""
echo "安装 PyTorch (CUDA 12.1)..."
echo "注意: PyTorch 将从官方源下载，但会使用国内镜像加速"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖（使用国内镜像）
echo ""
echo "安装基础依赖（使用清华镜像）..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers>=4.35.0 \
    accelerate>=0.24.0 \
    bitsandbytes>=0.41.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    numpy>=1.24.0 \
    tqdm>=4.66.0

# 安装实验依赖
echo ""
echo "安装实验依赖..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    datasets>=2.14.0 \
    rouge-score>=0.1.2 \
    nltk>=3.8.0

# 尝试安装 Flash Attention (可选，可能失败)
echo ""
echo "尝试安装 Flash Attention (可选)..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn --no-build-isolation || \
    echo "⚠️  Flash Attention 安装失败，将使用标准注意力（不影响主要功能）"

# 配置 HuggingFace 镜像（用于下载模型）
echo ""
echo "配置 HuggingFace 镜像源..."
mkdir -p ~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
echo "✅ HuggingFace 镜像已配置（hf-mirror.com）"

# 创建环境变量文件
echo ""
echo "创建环境变量配置文件..."
cat > .env_rtx4050 << EOF
# RTX 4050 实验环境变量
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
EOF
echo "✅ 环境变量文件已创建: .env_rtx4050"

echo ""
echo "=========================================="
echo "环境搭建完成！"
echo "=========================================="
echo ""
echo "使用说明:"
echo "1. 激活环境: conda activate $ENV_NAME"
echo "2. 加载环境变量: source .env_rtx4050"
echo "3. 运行快速测试: python scripts/quick_test.py"
echo "4. 运行实验: python experiments/memory_optimized_experiment.py --config configs/rtx4050_config.json"
echo ""
echo "镜像源配置:"
echo "  - Conda: 清华镜像"
echo "  - pip: 清华镜像"
echo "  - HuggingFace: hf-mirror.com"
echo ""
