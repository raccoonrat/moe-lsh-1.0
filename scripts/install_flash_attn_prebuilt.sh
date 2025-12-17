#!/bin/bash
# 尝试安装 flash-attn 预编译版本

set -e

echo "=========================================="
echo "尝试安装 flash-attn 预编译版本"
echo "=========================================="

# 检查环境
if ! conda env list | grep -q "moe-lsh-watermark"; then
    echo "错误: 请先创建 Conda 环境"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moe-lsh-watermark

# 检查 CUDA 版本
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION="12.1"  # 默认
    echo "未检测到 CUDA 版本，使用默认: $CUDA_VERSION"
else
    echo "检测到 CUDA 版本: $CUDA_VERSION"
fi

# 检查 PyTorch 版本
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
echo "PyTorch 版本: $PYTORCH_VERSION"

# 方法 1: 尝试从预编译 wheel 安装
echo ""
echo "方法 1: 尝试安装预编译 wheel..."
pip install flash-attn --no-build-isolation --no-cache-dir || {
    echo "预编译版本不可用，尝试其他方法..."
    
    # 方法 2: 使用 conda-forge（如果有）
    echo ""
    echo "方法 2: 尝试从 conda-forge 安装..."
    conda install -c conda-forge flash-attn -y || {
        echo "conda-forge 版本不可用"
        
        # 方法 3: 从源码编译（设置超时）
        echo ""
        echo "方法 3: 从源码编译（设置 30 分钟超时）..."
        echo "警告: 这可能需要 20-30 分钟"
        echo "按 Ctrl+C 可以随时中断"
        
        timeout 1800 pip install flash-attn --no-build-isolation || {
            echo ""
            echo "=========================================="
            echo "⚠️  flash-attn 安装失败或超时"
            echo "=========================================="
            echo ""
            echo "建议:"
            echo "1. flash-attn 不是必需的，可以跳过"
            echo "2. 代码会自动使用标准注意力机制"
            echo "3. 功能完全正常，只是可能稍慢一些"
            echo ""
            exit 1
        }
    }
}

echo ""
echo "=========================================="
echo "✅ flash-attn 安装成功！"
echo "=========================================="

# 验证安装
python -c "import flash_attn; print('flash-attn 版本:', flash_attn.__version__)" 2>/dev/null || \
    python -c "import flash_attn; print('flash-attn 已安装')" || \
    echo "⚠️  安装可能不完整，但可以继续使用"

