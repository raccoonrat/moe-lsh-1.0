#!/bin/bash
# 配置所有国内镜像源

echo "=========================================="
echo "配置国内镜像源"
echo "=========================================="

# 配置 Conda 镜像
echo "配置 Conda 镜像源（清华）..."
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

# 配置 pip 镜像
echo "配置 pip 镜像源（清华）..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 配置 HuggingFace 镜像
echo "配置 HuggingFace 镜像源（hf-mirror.com）..."
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

# 创建缓存目录
mkdir -p ~/.cache/huggingface

echo ""
echo "=========================================="
echo "镜像源配置完成！"
echo "=========================================="
echo ""
echo "已配置的镜像源:"
echo "  - Conda: 清华镜像"
echo "  - pip: 清华镜像"
echo "  - HuggingFace: hf-mirror.com"
echo ""
echo "注意: 需要重新打开终端或运行 'source ~/.bashrc' 使 HuggingFace 镜像生效"
echo ""

