#!/bin/bash
# 跳过 flash-attn 安装的环境搭建脚本

set -e

echo "=========================================="
echo "MoE LSH Watermark 环境搭建（跳过 flash-attn）"
echo "=========================================="

# 检查 Conda
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 Conda"
    exit 1
fi

# 配置镜像源（如果未配置）
if [ ! -f ~/.pip/pip.conf ]; then
    echo "配置 pip 镜像源..."
    mkdir -p ~/.pip
    cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
fi

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 创建/激活环境
ENV_NAME="moe-lsh-watermark"
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "创建 Conda 环境..."
    conda create -n $ENV_NAME python=3.10 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 升级 pip
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 PyTorch
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖（跳过 flash-attn）
echo "安装基础依赖（跳过 flash-attn）..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers>=4.35.0 \
    accelerate>=0.24.0 \
    bitsandbytes>=0.41.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    numpy>=1.24.0 \
    tqdm>=4.66.0 \
    datasets>=2.14.0 \
    rouge-score>=0.1.2 \
    nltk>=3.8.0

echo ""
echo "=========================================="
echo "✅ 环境搭建完成（未安装 flash-attn）"
echo "=========================================="
echo ""
echo "注意:"
echo "  - flash-attn 已跳过（编译时间过长）"
echo "  - 代码会自动使用标准注意力机制"
echo "  - 功能完全正常，只是可能稍慢一些"
echo ""

