# ICML 2025 论文：Endogenous Semantic Watermarking for MoE LLMs

本文档包含按照ICML 2025模板生成的论文LaTeX源文件。

## 文件说明

- `moe_lsh_watermark_paper.tex`: 英文版主论文LaTeX文件
- `moe_lsh_watermark_paper_zh.tex`: 中文版主论文LaTeX文件
- `moe_lsh_watermark_refs.bib`: 参考文献BibTeX文件
- `ICML2025_Template/`: ICML 2025官方模板文件

## 编译说明

### 前置要求

1. 安装LaTeX发行版（如TeX Live或MiKTeX）
2. **中文版本额外要求**：
   - 需要安装 `ctex` 包（通常包含在TeX Live完整版中）
   - 或使用XeLaTeX编译中文版本（推荐）
3. 确保包含以下包：
   - `icml2025.sty` (在模板目录中)
   - `algorithm.sty` 和 `algorithmic.sty` (在模板目录中)
   - 标准LaTeX包：`amsmath`, `amssymb`, `graphicx`, `hyperref` 等

### 编译步骤

1. 将论文文件复制到模板目录，或确保模板文件在LaTeX搜索路径中：

```bash
cd docs/ICML2025_Template
cp ../moe_lsh_watermark_paper.tex .
cp ../moe_lsh_watermark_refs.bib .
```

2. 编译LaTeX文件：

```bash
pdflatex moe_lsh_watermark_paper.tex
bibtex moe_lsh_watermark_paper
pdflatex moe_lsh_watermark_paper.tex
pdflatex moe_lsh_watermark_paper.tex
```

或者使用latexmk：

```bash
latexmk -pdf moe_lsh_watermark_paper.tex
```

3. 生成的PDF文件：`moe_lsh_watermark_paper.pdf`

### 编译中文版本

中文版本需要使用XeLaTeX或支持中文的LaTeX引擎：

```bash
# 方法1：使用XeLaTeX（推荐）
xelatex moe_lsh_watermark_paper_zh.tex
bibtex moe_lsh_watermark_paper_zh
xelatex moe_lsh_watermark_paper_zh.tex
xelatex moe_lsh_watermark_paper_zh.tex

# 方法2：使用pdflatex（需要ctex包）
pdflatex moe_lsh_watermark_paper_zh.tex
bibtex moe_lsh_watermark_paper_zh
pdflatex moe_lsh_watermark_paper_zh.tex
pdflatex moe_lsh_watermark_paper_zh.tex

# 方法3：使用latexmk
latexmk -xelatex moe_lsh_watermark_paper_zh.tex
```

注意：中文版本使用 `ctex` 包进行中文支持，编译时可能需要额外的时间来处理中文字体。

## 论文结构

论文按照review意见组织，包含以下主要部分：

1. **摘要**: 强调MoE生态、释义攻击问题、内生语义信号、LSH稳定化、主要结果
2. **引言**: MoE趋势、现有方法局限性、我们的贡献
3. **相关工作**: 
   - Token级水印（Kirchenbauer等）
   - 语义级水印（SemStamp）
   - MoE路由与稳定性
   - 鲁棒性评估
4. **方法**:
   - 路由权重提取
   - LSH签名生成（SimHash）
   - 绿色词表选择与偏置
   - 检测方法（白盒与窗口化）
   - 多层融合策略
5. **理论分析**:
   - LSH碰撞概率分析
   - 检测功效分析
6. **实验**:
   - 模型：Mixtral-8x7B, OpenMoE-8B/34B
   - 数据集：QA、摘要、数据到文本
   - 基线：Kirchenbauer, SemStamp, Unigram, Multi-bit
   - 攻击：人类释义、GPT-4释义、B4黑盒清洗、混合攻击
   - 指标：检测率、质量、鲁棒性曲线
7. **安全模型与局限性**:
   - 威胁模型（白盒vs公验）
   - 密钥管理
   - 局限性讨论
8. **结论**: 总结与未来方向

## 根据Review意见的改进

论文已根据review意见进行了以下调整：

1. ✅ 题目改为："Endogenous Semantic Watermarking for MoE LLMs via LSH-Stabilized Routing Weights"
2. ✅ 摘要结构：痛点→直觉→技术→结果→适用性
3. ✅ 明确"生成端近零开销；检测端需模型重构"（而非"零成本"）
4. ✅ 详细说明LSH家族（SimHash）、比特数、碰撞概率公式
5. ✅ 包含统计检测公式（z-score）和窗口化检测
6. ✅ 系统对比相关工作（表格形式）
7. ✅ 明确安全模型（白盒vs公验）
8. ✅ 包含实验设计：模型、数据集、基线、攻击方法、指标

## 待补充内容

以下内容需要根据实际实验数据补充：

1. **实验结果表格**：需要实际运行实验后填入数据
   - Table 1: 鲁棒性结果（检测率）
   - Table 2: 质量指标（困惑度、BLEU、ROUGE）
   - Figure 1: 检测曲线（z-score vs 文本长度）

2. **作者信息**：在提交前需要填写真实作者和单位

3. **致谢**：需要添加实际资助信息

4. **图表**：需要创建实际的实验图表

## 注意事项

- 当前使用 `\usepackage{icml2025}` 用于盲审版本
- 如果论文被接受，需要改为 `\usepackage[accepted]{icml2025}` 并添加作者信息
- 确保所有引用在.bib文件中都有对应条目
- 检查图表路径是否正确（当前为占位符）

## 参考文献

论文引用了以下主要工作：

- Kirchenbauer et al. (2024): 绿色词表水印
- Zhao et al. (2024): SemStamp语义水印
- Kuditipudi et al. (2024): Unigram水印
- Qu et al. (2025): 多比特水印
- Fedus et al. (2022): Switch Transformers (MoE)
- Zhou et al. (2024): MoE路由分析
- Wang et al. (2024): WaterPark评估框架
- Liu et al. (2025): B4黑盒攻击

所有引用已在 `moe_lsh_watermark_refs.bib` 中定义。

