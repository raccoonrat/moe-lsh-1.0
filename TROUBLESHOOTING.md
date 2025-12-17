# 故障排除指南

## flash-attn 编译问题

### 问题：Building wheel for flash-attn 时间过长

**原因**:
- flash-attn 需要从源码编译 CUDA 代码
- 编译时间通常需要 20-30 分钟甚至更久
- 取决于 CPU 性能、CUDA 版本、系统配置

### 解决方案

#### 方案 1: 跳过 flash-attn（推荐）

flash-attn **不是必需的**，代码会自动使用标准注意力机制。

**快速跳过安装:**
```bash
# 中断当前安装 (Ctrl+C)
# 然后运行：
chmod +x scripts/install_without_flash_attn.sh
./scripts/install_without_flash_attn.sh
```

**或者手动跳过:**
```bash
# 在安装脚本中，flash-attn 安装行已经设置了 || echo，会自动跳过
# 如果卡住，直接 Ctrl+C 中断即可
```

#### 方案 2: 检查编译是否正常进行

```bash
# 运行检查脚本
chmod +x scripts/check_flash_attn_build.sh
./scripts/check_flash_attn_build.sh

# 或者手动检查
# 1. 检查 CPU 使用率
top

# 2. 检查编译进程
ps aux | grep flash-attn

# 3. 检查是否有错误日志
tail -f ~/.cache/pip/log/*.log
```

**正常编译的特征:**
- CPU 使用率较高（接近 100%）
- 有 `gcc` 或 `nvcc` 进程在运行
- 有磁盘 I/O 活动
- 没有错误信息

**异常情况:**
- CPU 使用率为 0
- 进程无响应
- 出现编译错误

#### 方案 3: 使用预编译版本

```bash
chmod +x scripts/install_flash_attn_prebuilt.sh
./scripts/install_flash_attn_prebuilt.sh
```

**注意**: 预编译版本可能不适用于所有 CUDA/PyTorch 版本组合。

#### 方案 4: 设置编译超时

如果决定编译，可以设置超时：

```bash
# 设置 30 分钟超时
timeout 1800 pip install flash-attn --no-build-isolation
```

### 验证 flash-attn 是否安装成功

```python
try:
    import flash_attn
    print("✅ flash-attn 已安装")
except ImportError:
    print("⚠️  flash-attn 未安装，将使用标准注意力")
```

### 使用标准注意力（无 flash-attn）

代码会自动检测并使用标准注意力：

```python
# 在 configs/rtx4050_config.json 中
{
  "use_flash_attention": false  # 设置为 false 或删除此选项
}
```

## 其他常见问题

### 问题 1: Conda 环境创建失败

**解决方案:**
```bash
# 清理 conda 缓存
conda clean --all

# 使用国内镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

# 重新创建
conda create -n moe-lsh-watermark python=3.10 -y
```

### 问题 2: PyTorch 安装失败

**解决方案:**
```bash
# 检查 CUDA 版本
nvidia-smi
nvcc --version

# 安装对应版本的 PyTorch
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题 3: 模型下载很慢

**解决方案:**
```bash
# 确保设置了 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或者在代码中设置（已自动设置）
# experiments/memory_optimized_experiment.py 已包含
```

### 问题 4: 显存不足 (OOM)

**解决方案:**
1. 使用 4-bit 量化（已在配置中）
2. 减少 MoE 层数
3. 减少生成长度
4. 减少 batch size

### 问题 5: bitsandbytes 安装失败

**解决方案:**
```bash
# 确保安装了正确的 CUDA 工具包
# 然后重新安装
pip install bitsandbytes --no-cache-dir

# 或者从 conda 安装
conda install -c conda-forge bitsandbytes
```

## 调试技巧

### 1. 检查环境

```bash
# 检查 Python 版本
python --version

# 检查 CUDA
nvidia-smi
nvcc --version

# 检查已安装的包
pip list | grep -E "(torch|transformers|bitsandbytes)"
```

### 2. 检查日志

```bash
# pip 安装日志
cat ~/.cache/pip/log/*.log | tail -50

# Conda 日志
conda info
```

### 3. 清理缓存

```bash
# 清理 pip 缓存
pip cache purge

# 清理 conda 缓存
conda clean --all

# 清理 HuggingFace 缓存
rm -rf ~/.cache/huggingface
```

### 4. 逐步安装

如果批量安装失败，可以逐步安装：

```bash
# 1. PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Transformers
pip install transformers accelerate

# 3. 其他依赖
pip install scipy scikit-learn numpy tqdm
```

## 获取帮助

如果问题仍未解决：

1. **检查错误信息**: 复制完整的错误信息
2. **检查环境**: 运行 `scripts/quick_test.py`
3. **查看日志**: 检查安装日志文件
4. **简化配置**: 使用最小配置测试

## 快速恢复

如果环境完全损坏，可以快速重建：

```bash
# 删除环境
conda env remove -n moe-lsh-watermark

# 重新运行安装脚本（跳过 flash-attn）
./scripts/install_without_flash_attn.sh
```

