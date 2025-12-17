# 国内镜像源配置说明

本项目已配置使用国内镜像源，加速依赖包和模型的下载。

## 已配置的镜像源

### 1. Conda 镜像（清华）
- 主频道: `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main`
- Free 频道: `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free`
- R 频道: `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r`
- Conda-forge: `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge`

### 2. pip 镜像（清华）
- 镜像地址: `https://pypi.tuna.tsinghua.edu.cn/simple`
- 配置文件: `~/.pip/pip.conf` (Linux/Mac) 或 `%APPDATA%\pip\pip.ini` (Windows)

### 3. HuggingFace 镜像（hf-mirror.com）
- 镜像地址: `https://hf-mirror.com`
- 环境变量: `HF_ENDPOINT=https://hf-mirror.com`

## 使用方法

### 方法 1: 使用自动配置脚本（推荐）

**Linux/Mac:**
```bash
chmod +x scripts/configure_mirrors.sh
./scripts/configure_mirrors.sh
```

**Windows:**
```cmd
REM 运行 setup_environment.bat 会自动配置所有镜像源
scripts\setup_environment.bat
```

### 方法 2: 手动配置

#### Conda 镜像
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

#### pip 镜像

**Linux/Mac:**
```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

**Windows:**
```cmd
mkdir %APPDATA%\pip
(
    echo [global]
    echo index-url = https://pypi.tuna.tsinghua.edu.cn/simple
    echo trusted-host = pypi.tuna.tsinghua.edu.cn
) > %APPDATA%\pip\pip.ini
```

#### HuggingFace 镜像

**Linux/Mac:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
```

**Windows:**
```cmd
setx HF_ENDPOINT "https://hf-mirror.com"
set HF_ENDPOINT=https://hf-mirror.com
```

## 验证配置

### 验证 Conda 镜像
```bash
conda config --show channels
```

### 验证 pip 镜像
```bash
pip config list
# 或
pip install --dry-run numpy  # 查看使用的源
```

### 验证 HuggingFace 镜像
```bash
echo $HF_ENDPOINT  # Linux/Mac
echo %HF_ENDPOINT%  # Windows
```

## 使用镜像下载模型

代码中已自动配置 HuggingFace 镜像。如果环境变量未设置，代码会自动设置：

```python
import os
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

下载模型时会自动使用镜像：
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-MoE-A14B-Chat")
# 会自动从 hf-mirror.com 下载
```

## 其他可用镜像源

如果清华镜像速度慢，可以尝试：

### pip 镜像备选
- 阿里云: `https://mirrors.aliyun.com/pypi/simple/`
- 中科大: `https://pypi.mirrors.ustc.edu.cn/simple/`
- 豆瓣: `https://pypi.douban.com/simple/`

### HuggingFace 镜像备选
- 官方: `https://huggingface.co` (默认，可能较慢)

## 常见问题

### Q: 为什么 PyTorch 还是从官方源下载？

A: PyTorch 的 CUDA 版本需要从官方源下载，但安装脚本会使用官方索引。如果速度慢，可以考虑：
1. 使用代理
2. 下载 wheel 文件后本地安装
3. 使用 conda 安装（但可能版本较旧）

### Q: HuggingFace 镜像不生效？

A: 确保：
1. 环境变量已设置: `echo $HF_ENDPOINT`
2. 代码中已设置环境变量（`experiments/memory_optimized_experiment.py` 已包含）
3. 重新启动终端/IDE

### Q: 如何临时使用其他镜像？

A: 可以在命令中指定：
```bash
# pip
pip install -i https://mirrors.aliyun.com/pypi/simple/ package_name

# HuggingFace
HF_ENDPOINT=https://huggingface.co python script.py
```

## 参考链接

- [清华 Conda 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)
- [清华 pip 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [HuggingFace 镜像站](https://hf-mirror.com)

