#!/bin/bash
# 检查 flash-attn 编译状态

echo "=========================================="
echo "检查 flash-attn 编译状态"
echo "=========================================="

# 检查是否有编译进程在运行
echo "检查编译进程..."
if pgrep -f "flash-attn" > /dev/null; then
    echo "✅ 发现 flash-attn 编译进程正在运行"
    ps aux | grep -i flash-attn | grep -v grep
else
    echo "⚠️  未发现编译进程，可能已卡住或完成"
fi

# 检查 CPU 使用率
echo ""
echo "检查 CPU 使用率..."
if command -v top &> /dev/null; then
    echo "当前 CPU 使用率（前5个进程）:"
    top -b -n 1 | head -n 12
fi

# 检查磁盘 I/O
echo ""
echo "检查磁盘 I/O..."
if command -v iostat &> /dev/null; then
    iostat -x 1 2
fi

# 检查编译日志位置
echo ""
echo "检查可能的编译日志..."
if [ -d ~/.cache/pip ]; then
    echo "pip 缓存目录: ~/.cache/pip"
    ls -lh ~/.cache/pip/wheels/ 2>/dev/null | tail -5 || echo "无缓存文件"
fi

# 检查临时文件
echo ""
echo "检查临时编译文件..."
TMP_DIRS=("/tmp" "$TMPDIR" "$TEMP")
for dir in "${TMP_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "检查 $dir:"
        find "$dir" -name "*flash*" -type f -mmin -30 2>/dev/null | head -5 || echo "  无相关文件"
    fi
done

echo ""
echo "=========================================="
echo "建议操作:"
echo "1. 如果编译超过 30 分钟，建议中断（Ctrl+C）"
echo "2. flash-attn 不是必需的，可以跳过"
echo "3. 使用预编译版本或跳过安装"
echo "=========================================="

