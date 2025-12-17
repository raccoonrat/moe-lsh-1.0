@echo off
REM 使用 environment.yml 创建 Conda 环境（推荐方式）

echo ==========================================
echo 使用 Conda 环境文件创建环境
echo ==========================================

REM 检查 Conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到 Conda
    pause
    exit /b 1
)

REM 配置 pip 国内镜像
echo 配置 pip 国内镜像源...
if not exist "%APPDATA%\pip" mkdir "%APPDATA%\pip"
(
    echo [global]
    echo index-url = https://pypi.tuna.tsinghua.edu.cn/simple
    echo trusted-host = pypi.tuna.tsinghua.edu.cn
    echo timeout = 120
) > "%APPDATA%\pip\pip.ini"

REM 配置 HuggingFace 镜像
setx HF_ENDPOINT "https://hf-mirror.com" >nul 2>&1
set HF_ENDPOINT=https://hf-mirror.com

REM 创建环境
echo 从 environment.yml 创建环境...
conda env create -f environment.yml

echo.
echo ==========================================
echo 环境创建完成！
echo ==========================================
echo.
echo 激活环境: conda activate moe-lsh-watermark
echo 注意: PyTorch 需要单独安装 CUDA 版本
echo 运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

pause

