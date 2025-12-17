@echo off
REM RTX 4050 实验环境搭建脚本 (Windows, 使用 Conda + 国内镜像)

echo ==========================================
echo MoE LSH Watermark 实验环境搭建
echo 适用于 RTX 4050 (6-8GB 显存)
echo 使用 Conda + 国内镜像源
echo ==========================================

REM 检查 Conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到 Conda
    echo 请先安装 Miniconda 或 Anaconda
    echo 下载地址: https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/
    pause
    exit /b 1
)

echo 检测到的 Conda 版本:
conda --version

REM 检查 CUDA
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: 未检测到 NVIDIA GPU，将使用 CPU 模式
) else (
    echo 检测到的 GPU:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

REM 配置 Conda 国内镜像源
echo.
echo 配置 Conda 国内镜像源（清华镜像）...
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

REM 配置 pip 国内镜像源
echo.
echo 配置 pip 国内镜像源...
if not exist "%APPDATA%\pip" mkdir "%APPDATA%\pip"
(
    echo [global]
    echo index-url = https://pypi.tuna.tsinghua.edu.cn/simple
    echo trusted-host = pypi.tuna.tsinghua.edu.cn
    echo timeout = 120
) > "%APPDATA%\pip\pip.ini"
echo ✅ pip 镜像源已配置（清华镜像）

REM 创建 Conda 环境
set ENV_NAME=moe-lsh-watermark
set PYTHON_VERSION=3.10

echo.
echo 创建 Conda 环境: %ENV_NAME% (Python %PYTHON_VERSION%)...
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo 环境 %ENV_NAME% 已存在，是否删除并重新创建? (Y/N)
    set /p response=
    if /i "%response%"=="Y" (
        conda env remove -n %ENV_NAME% -y
        conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
    ) else (
        echo 使用现有环境
    )
) else (
    conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
)

REM 激活环境
echo.
echo 激活 Conda 环境...
call conda activate %ENV_NAME%

REM 升级 pip
echo.
echo 升级 pip...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 安装 PyTorch (CUDA 12.1)
echo.
echo 安装 PyTorch (CUDA 12.1)...
echo 注意: PyTorch 将从官方源下载，但会使用国内镜像加速
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM 安装基础依赖（使用国内镜像）
echo.
echo 安装基础依赖（使用清华镜像）...
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers>=4.35.0 accelerate>=0.24.0 bitsandbytes>=0.41.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy>=1.11.0 scikit-learn>=1.3.0 numpy>=1.24.0 tqdm>=4.66.0

REM 安装实验依赖
echo.
echo 安装实验依赖...
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets>=2.14.0 rouge-score>=0.1.2 nltk>=3.8.0

REM 尝试安装 Flash Attention (可选)
echo.
echo 尝试安装 Flash Attention (可选)...
echo 注意: flash-attn 编译可能需要 20-30 分钟，可以按 Ctrl+C 跳过
echo flash-attn 不是必需的，代码会自动使用标准注意力机制
echo.
set /p install_flash="是否安装 flash-attn? (直接回车跳过) [N]: "
if /i "%install_flash%"=="Y" (
    echo 开始安装 flash-attn（这可能需要较长时间）...
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn --no-build-isolation || (
        echo ⚠️  Flash Attention 安装失败，将使用标准注意力（不影响主要功能）
    )
) else (
    echo 跳过 flash-attn 安装
)

REM 配置 HuggingFace 镜像
echo.
echo 配置 HuggingFace 镜像源...
if not exist "%USERPROFILE%\.cache\huggingface" mkdir "%USERPROFILE%\.cache\huggingface"
setx HF_ENDPOINT "https://hf-mirror.com" >nul 2>&1
set HF_ENDPOINT=https://hf-mirror.com
echo ✅ HuggingFace 镜像已配置（hf-mirror.com）

REM 创建环境变量文件
echo.
echo 创建环境变量配置文件...
(
    echo # RTX 4050 实验环境变量
    echo set HF_ENDPOINT=https://hf-mirror.com
    echo set HF_HOME=%USERPROFILE%\.cache\huggingface
    echo set TRANSFORMERS_CACHE=%USERPROFILE%\.cache\huggingface\transformers
    echo set HF_DATASETS_CACHE=%USERPROFILE%\.cache\huggingface\datasets
) > .env_rtx4050.bat
echo ✅ 环境变量文件已创建: .env_rtx4050.bat

echo.
echo ==========================================
echo 环境搭建完成！
echo ==========================================
echo.
echo 使用说明:
echo 1. 激活环境: conda activate %ENV_NAME%
echo 2. 加载环境变量: call .env_rtx4050.bat
echo 3. 运行快速测试: python scripts/quick_test.py
echo 4. 运行实验: python experiments/memory_optimized_experiment.py --config configs/rtx4050_config.json
echo.
echo 镜像源配置:
echo   - Conda: 清华镜像
echo   - pip: 清华镜像
echo   - HuggingFace: hf-mirror.com
echo.

pause
