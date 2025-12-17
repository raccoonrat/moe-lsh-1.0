#!/bin/bash
# 使用 environment.yml 创建 Conda 环境（推荐方式）

set -e

echo "=========================================="
echo "使用 Conda 环境文件创建环境"
echo "=========================================="

# 检查 Conda
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 Conda"
    exit 1
fi

# 配置 pip 国内镜像
echo "配置 pip 国内镜像源..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
EOF

# 配置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

# 创建环境
echo "从 environment.yml 创建环境..."
conda env create -f environment.yml

echo ""
echo "=========================================="
echo "环境创建完成！"
echo "=========================================="
echo ""
echo "激活环境: conda activate moe-lsh-watermark"
echo "注意: PyTorch 需要单独安装 CUDA 版本"
echo "运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""

