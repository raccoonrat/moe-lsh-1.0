@echo off
REM RTX 4050 实验环境搭建脚本 (Windows)

echo ==========================================
echo MoE LSH Watermark 实验环境搭建
echo 适用于 RTX 4050 (6-8GB 显存)
echo ==========================================

REM 检查 CUDA
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到 NVIDIA GPU
    exit /b 1
)

echo 检测到的 GPU:
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

REM 创建虚拟环境
echo.
echo 创建 Python 虚拟环境...
python -m venv venv
call venv\Scripts\activate.bat

REM 安装 PyTorch (CUDA 12.1)
echo.
echo 安装 PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM 安装基础依赖
echo.
echo 安装基础依赖...
pip install transformers accelerate bitsandbytes
pip install scipy scikit-learn numpy
pip install tqdm

REM 安装可选依赖
echo.
echo 安装实验依赖...
pip install datasets
pip install rouge-score
pip install nltk

REM 安装 Flash Attention (可选)
echo.
echo 尝试安装 Flash Attention...
pip install flash-attn --no-build-isolation || echo Flash Attention 安装失败，将使用标准注意力

echo.
echo ==========================================
echo 环境搭建完成！
echo ==========================================
echo.
echo 使用说明:
echo 1. 激活虚拟环境: venv\Scripts\activate
echo 2. 运行实验: python experiments/main_experiment.py --config configs/rtx4050_config.json
echo.

pause

