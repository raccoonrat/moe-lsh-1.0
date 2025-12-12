<style>a { text-decoration: none; color: #464feb;}tr th, tr td { border: 1px solid #e6e6e6;}tr th { background-color: #f5f5f5;}</style>

下面我以“跨学科教授 + 顶会评审主席（PC Chair）”的视角，**对你附件《MoE 路由权重 + LSH 的鲁棒语义水印》方案**进行深度评审与打磨，目标明确瞄准 **Spotlight/Oral**。我会分为：**创新性与定位**、**与现有工作的系统对比**、**理论与安全模型的补强建议**、**实验设计（可交付、可复现）**、**工程可行性与影响面**、**写作与呈现** 六大部分，并在关键处给出改稿建议（含可替换的术语与图示），**每一条主张都附上可验证的引用**。

* * *

1) 创新性与定位：你方案的“独特卖点”（USP）

-------------------------

**核心卖点**

* 用 **MoE 路由权重（Routing Weights, RW）** 作为“内生语义信号（endogenous semantic signal）”，通过 **LSH（Locality‑Sensitive Hashing）** 将连续的路由向量稳定映射为**离散水印种子**，据此驱动 **绿色词表（green list）**的选择与检测。与“外部嵌入”或“额外训练的探针网络”不同，你**不引入外部模型或新增训练**，把信号取自**推理必经路径**。
* **鲁棒性主张**：对“释义（paraphrase）”造成的语义微扰，RW 轻度变化经 LSH 仍会**稳定碰撞到相同（或近似）的哈希桶**，从而保持水印。此点与 sentence‑level 语义水印（如 SemStamp）相似，但你**以模型内部信号**替换了**外部语义嵌入**，因此**效率更高**。 [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226.pdf), [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226/), [[arxiv.org]](https://arxiv.org/pdf/2310.03991)
* **训练‑自由（training‑free）/在线低开销**：RW 本就是 MoE 路由器在推理时计算的产物；LSH 计算轻量；绿色词表与偏置通过 HuggingFace 的 logits processor 实现**近零工程侵入**。 [[papers.nips.cc]](https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html), [[github.com]](https://github.com/jwkirchenbauer/lm-watermarking)

**机会窗口**

* 当前开源与闭源大模型中 **MoE 正在强势扩张**：从 Mixtral/DeepSeek 到 2025 的 Mistral‑3 系列（含 675B 稀疏 MoE），使你方案的适用面并非小众。建议在导言中明确这一生态趋势，强调“**MoE 时代的原生水印**”。 [[blog.csdn.net]](https://blog.csdn.net/v_JULY_v/article/details/145406756), [[developer.nvidia.com]](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/)

**需要谨慎的地方（评审会问）**

* **适用性**：密集（dense）Transformer 不适用，你需要把“MoE‑only”作为**设计选择**而非“局限”，并论证“MoE 的广泛现实性”。 [[developer.nvidia.com]](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/)
* **“零成本”表述**需降调：嵌入端确实近零额外 FLOPs，但**检测端**若需重构 RW（需跑同一模型前向），**不是零成本**。建议改为“**生成端近零开销；检测端成本与文本长度线性**”。（统计检测公式参考下文） [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz), [[deepwiki.com]](https://deepwiki.com/bangawayoo/mb-lm-watermarking/7.1-watermark-detection-metrics)

* * *

2) 与现有工作的系统对比：把“你与他们”的边界拉清晰

---------------------------

### 2.1 经典“绿色词表”水印（Kirchenbauer 等）

* **机制**：对每步 vocab 以伪随机函数（PRF）划分绿/红表；在生成端给“绿表”加微小偏置，检测端用 **z‑score** 统计检验。优点是工程简单、训练自由；缺点是对**释义/改写**较脆弱，且在**短文本**上统计力不足。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz), [[github.com]](https://github.com/jwkirchenbauer/lm-watermarking)
* **你的差异**：PRF 种子由 **LSH(RW)** 替换，避免对**表面词串哈希**的脆弱性。定位上应写成“**模型内生语义驱动的 PRF**”。

### 2.2 语义水印/句级水印（SemStamp 等）

* **机制**：句嵌入 + LSH + 拒绝采样，稳语义、抗释义，但**每步需外部嵌入推理**，在线开销显著。 [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226.pdf), [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226/)
* **你的差异**：语义来自 **MoE 路由器**，**不依赖外部编码器**；同时**保留 token‑level 的在线控制**（在每步选绿表）。建议在 Related Work 中打出“**Semantics without external encoders**”的小节标题。

### 2.3 水印鲁棒性再评价与攻击工作

* 系统性评测表明，多数水印在**强释义/黑盒攻击**下性能显著下滑（如 WaterPark、B4、CMU 的攻击论文），强调**真正对抗设置**下的评测必要性。你的论文必须在**威胁模型/攻击基线**上一口气到位。 [[arxiv.org]](https://arxiv.org/html/2411.13425v3), [[aclanthology.org]](https://aclanthology.org/2025.naacl-long.460/), [[arxiv.org]](https://arxiv.org/html/2402.16187v1)
* 近期也有声称“**provable robust**”与**multi‑bit**可追踪的工作（ICLR’24、USENIX’25），你需要与其**检测统计与容错边界**对齐并做严谨对比。 [[arxiv.org]](https://arxiv.org/abs/2306.17439), [[proceedings.iclr.cc]](https://proceedings.iclr.cc/paper_files/paper/2024/file/beae9ed5316bcc48e616754c06c11875-Paper-Conference.pdf), [[usenix.org]](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-446-qu-watermarking.pdf)

### 2.4 MoE 路由的行为特性

* **Top‑k 离散路由 + 负载均衡**等稳定化技巧已成为主流；这意味着 RW 在不同层/不同 batch 具有**随机性与容量限制**（token dropping）。你需要实证：**哪一层的 RW** 对语义最稳，且**跨层融合**是否提升稳定性。 [[papers.nips.cc]](https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html), [[arxiv.org]](https://arxiv.org/abs/2202.09368)
* 开源路由分析显示，部分 MoE 的路由**早期即固化、与 token ID 相关性高**，对“上下文语义”敏感性有限。这对你是**双刃剑**：稳（抗释义）但也可能**减弱语义绑定的理论说服力**。必须实验消融“**ID‑ vs context‑driven**”。 [[arxiv.org]](https://arxiv.org/abs/2402.01739)

* * *

3) 理论与安全模型：把“鲁棒/可检”的边界讲清楚

-------------------------

### 3.1 LSH 碰撞概率与语义稳健性的形式化

* 若采用 **SimHash（随机投影 LSH）**，两向量夹角为 θ 时，同一比特碰撞概率为 **1 – θ/π**；b 比特签名下的汉明稳定性可由独立同分布近似推导。建议提供：**“释义导致 RW 在余弦球内扰动 ≤Δθ 时的碰撞下界”**，并据此给出**期望绿表命中率提升的界**。 [[en.wikipedia.org]](https://en.wikipedia.org/wiki/Locality-sensitive_hashing), [[towardsdat…cience.com]](https://towardsdatascience.com/similarity-search-part-5-locality-sensitive-hashing-lsh-76ae4b388203/)

### 3.2 统计检测与阈值设定

* 采用经典 z‑score：给定文本长度 n、绿表占比 p（= |G|/|V|），命中数 X ~ Binomial(n, p)；  
  **Z = (X − n·p) / sqrt(n·p·(1‑p))**；在 FPR=10⁻⁵ 的阈下，对不同 n 的**功效曲线**需给出（可参考 ICLR’24 的经验结论：强人类释义后约需 ~800 tokens 才能稳定检出）。你应在论文中给出**窗口化检测**与**短文本性能**的策略。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz), [[deepwiki.com]](https://deepwiki.com/bangawayoo/mb-lm-watermarking/7.1-watermark-detection-metrics)

### 3.3 威胁模型与可验证性

* **检测端依赖同一模型重构 RW** → 这是**白盒/供应商自证**模型（provider‑verifiable），不等同于**公众可验证（public verifier）**。请在安全模型中明确两种模式，并阐述**为何选择白盒检测**（例如用于平台内治理/审计）。相关“白盒/黑盒术语”定义可参考对抗学习与密码学水印的文献综述。 [[github.com]](https://github.com/cs231n/cs231n.github.io/blob/master/adversary-attacks.md), [[mdpi.com]](https://www.mdpi.com/1099-4300/23/10/1359), [[eprint.iacr.org]](https://eprint.iacr.org/2025/265.pdf)
* **密钥与词池**：与“绿色词表水印”一样，你的 LSH 密钥与词池映射需**保密**；请加入**密钥泄露/词池反推**的风险与轮换策略（Key rotation），以及**多密钥/多域隔离**的实践建议。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz)

* * *

4) 实验设计（强评审设定，一次到位）

-------------------

> 目标：**跨模型、跨任务、跨攻击**的系统评测；指标含 **AUC、z‑score\@FPR、困惑度/质量、鲁棒性曲线**；报告**检出所需 token 阈**；所有代码基于 HuggingFace `logits_processor` 与公开 MoE 模型。

### 4.1 模型与数据

* **模型**：Mixtral‑8x7B/Instruct、OpenMoE‑8B/34B、Mistral‑3（MoE）；对照密集 LLM（Llama‑3/Ministral‑14B）以证明“MoE 前提”。 [[github.com]](https://github.com/XueFuzhao/OpenMoE), [[arxiv.org]](https://arxiv.org/abs/2402.01739), [[blog.csdn.net]](https://blog.csdn.net/v_JULY_v/article/details/145406756), [[developer.nvidia.com]](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/)
* **任务**：开放领域问答、摘要（CTG）、数据到文本（WebNLG）；摘要场景需引用“**CTG 下水印‑质量悖论**”的讨论并给出折衷参数。 [[ojs.aaai.org]](https://ojs.aaai.org/index.php/AAAI/article/view/29756/31301)

### 4.2 基线方法

* **Kirchenbauer 绿表水印**（官方实现）；**SemStamp**（句级 LSH）；**Unigram‑Watermark（ICLR’24）**；**Multi‑bit 水印（USENIX’25）**。 [[github.com]](https://github.com/jwkirchenbauer/lm-watermarking), [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226/), [[arxiv.org]](https://arxiv.org/abs/2306.17439), [[usenix.org]](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-446-qu-watermarking.pdf)

### 4.3 攻击/后处理集（必须覆盖强对手）

* **释义**：人类重写；非水印 LLM（e.g., GPT 系列或密集模型）单轮、多轮释义；**“bigram paraphrase”** 攻击作为强基线。 [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226.pdf)
* **黑盒洗净**：B4 黑盒 scrubbing；“Attacking by exploiting strengths” 的策略集；（可选）RL‑based 自适应擦除。 [[aclanthology.org]](https://aclanthology.org/2025.naacl-long.460/), [[arxiv.org]](https://arxiv.org/html/2402.16187v1), [[openreview.net]](https://openreview.net/forum?id=iuebm4vXuI)
* **混合与裁剪**：混入人类文本、裁剪/摘要、句序打乱、翻译来回。综合评估**窗口化检测**。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz)

### 4.4 指标与报告

* **Detectability**：AUC、FPR=10⁻⁵ 时的 TPR；**z‑score 曲线 vs 文本长度**；窗口化检测（如 128/256/512 tokens）。 [[deepwiki.com]](https://deepwiki.com/bangawayoo/mb-lm-watermarking/7.1-watermark-detection-metrics)
* **质量影响**：困惑度、BLEU/Rouge、人工评审打分；报告“**水印强度‑质量‑检出率**”三方折衷（参考 WaterJudge/可靠性论文的呈现方式）。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz), [[github.com]](https://github.com/hzy312/Awesome-LLM-Watermark)
* **鲁棒性曲线**：随释义强度（编辑距离/语义相似度）与攻击轮次的检出率衰减。参考 WaterPark 的综合版图。 [[arxiv.org]](https://arxiv.org/html/2411.13425v3)

> **交付时间线建议（4–6 周最小可行复现）**  
> 第 1–2 周：实现嵌入与检测管线（Mixtral/OpenMoE），跑无攻击基线 + 参数搜索。  
> 第 3–4 周：释义与黑盒攻击集评测；窗口化检测与功效曲线。  
> 第 5–6 周：跨模型移植与 CTG 质量权衡报告；撰写与图表固化。  
> （与您既往“4–6 周最小复现”的偏好一致）

* * *

5) 工程可行性与落地面

------------

* **嵌入端**：用 HuggingFace `WatermarkLogitsProcessor` 的思路扩展，将 **PRF→LSH(RW)**；RW 读取需在 MoE 前馈层加入轻量 hook。 [[github.com]](https://github.com/jwkirchenbauer/lm-watermarking)
* **选择哪一层的 RW**：建议比较中层 vs 深层；并探索 **多层签名的 AND/OR 扩增**（LSH 放大理论），以在鲁棒与唯一性之间调参。 [[en.wikipedia.org]](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
* **适配广度**：将 Mixtral/OpenMoE 作为开源演示；并在文本中引用产业动向（Mistral‑3 MoE）的事实来佐证可用性。 [[developer.nvidia.com]](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/)

**风险与缓释**

* **路由器的随机性/负载均衡**可能引入不稳定；需在实现中固定数值精度（FP32 router）、禁用 token dropping 对签名位影响或使用**多层多数表决**缓解。 [[papers.nips.cc]](https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)
* **语义绑定的质疑**（路由更像“token‑ID 策略”）：把它写成消融实验与讨论点，而不是回避。 [[arxiv.org]](https://arxiv.org/abs/2402.01739)

* * *

6) 写作与呈现：把“Spotlight/Oral 的说服力”拉满

---------------------------------

### 6.1 题目与摘要

* 题目建议：**“Endogenous Semantic Watermarking for MoE LLMs via LSH‑Stabilized Routing Weights”**
* 摘要结构：痛点（释义攻击）→ 关键直觉（RW 是内生语义）→ 技术（LSH 稳定签名 + token‑level green list）→ 主要结果（鲁棒/质量/开销）→ 适用性（MoE 生态）。

### 6.2 方法图与流程图

* 强化你第 6–11 页的流程图：明确**层号/向量维度/LSH 家族（SimHash/MinHash）**与**统计检测**的公式。加入“**窗口化检测**”与“**多层 AND/OR**”的小图示。 [[en.wikipedia.org]](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)

### 6.3 相关工作版图（表格/雷达图）

* 维度：**训练需求**、**在线开销**、**释义鲁棒**、**公众可验**、**可追踪（multi‑bit）**、**文本质量**。把你和（Kirchenbauer/Unigram/SemStamp/Multi‑bit）并排。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz), [[arxiv.org]](https://arxiv.org/abs/2306.17439), [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226/), [[usenix.org]](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-446-qu-watermarking.pdf)

### 6.4 结论与边界

* 降调“零成本”，改为“**生成端近零开销；检测端需同模重构，适合平台侧治理**”；
* 明确**MoE 前提**与**白盒检测**的应用场景（平台合规、学术审计、基准污染检测的供方自证），并可顺带引用“**用水印检测基准污染**”的近期研究以拓展社会影响面。 [[arxiv.org]](https://arxiv.org/html/2502.17259v1)

* * *

一页“评审式清单”（你可以直接移植到论文的 Checklists）
---------------------------------

* **是否清晰限定威胁模型（白盒 vs 公众可验）？**（是：平台侧治理/自验证） [[github.com]](https://github.com/cs231n/cs231n.github.io/blob/master/adversary-attacks.md)
* **是否给出 LSH 抗扰动的形式化下界与多层放大策略？**（给出 SimHash 角度碰撞概率与 AND/OR 放大） [[en.wikipedia.org]](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
* **是否覆盖强释义与黑盒攻击（B4/对抗策略）？**（覆盖，并报告短文本窗口化性能） [[aclanthology.org]](https://aclanthology.org/2025.naacl-long.460/), [[arxiv.org]](https://arxiv.org/html/2402.16187v1)
* **是否与 SemStamp/Unigram/Multi‑bit 系统对比？**（对比并给出质量‑检测折衷） [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226/), [[arxiv.org]](https://arxiv.org/abs/2306.17439), [[usenix.org]](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-446-qu-watermarking.pdf)
* **是否证明 MoE 生态的现实性与工程可行性？**（引用 Mistral‑3/Mixtral/OpenMoE 与 HF 实现） [[developer.nvidia.com]](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/), [[blog.csdn.net]](https://blog.csdn.net/v_JULY_v/article/details/145406756), [[github.com]](https://github.com/XueFuzhao/OpenMoE), [[github.com]](https://github.com/jwkirchenbauer/lm-watermarking)

* * *

你稿件中需要**立刻**调整的 8 处（逐页点评）
-------------------------

1. **P5 “零成本（Zero‑Cost）”** → 改为 “**生成端近零开销；检测端需要模型重构上下文的 RW**”。并加一句：“我们也给出纯统计公验的降级检测器（窗口化 z‑score），但其功效不及白盒重构”。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz), [[deepwiki.com]](https://deepwiki.com/bangawayoo/mb-lm-watermarking/7.1-watermark-detection-metrics)
2. **P6–7**：注明 **采用的 LSH 家族（建议 SimHash）**、签名比特数 b、每层是否叠加；并给出**碰撞概率公式与释义扰动半径**的直觉。 [[en.wikipedia.org]](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
3. **P9–11 流程**：在检测流程图右侧补上 z‑score 公式与 **FPR‑TPR 曲线**示意；添加**窗口大小/步幅**参数。 [[deepwiki.com]](https://deepwiki.com/bangawayoo/mb-lm-watermarking/7.1-watermark-detection-metrics)
4. **P13 架构优势表**：在“对鲁棒性的贡献”一列加入“**需验证路由是否更偏 token‑ID vs 语义**”；引导读者去看消融实验。 [[arxiv.org]](https://arxiv.org/abs/2402.01739)
5. **加入相关工作页（新）**：把 “绿色词表/Unigram/句级LSH/Multi‑bit”并排对比（训练、开销、鲁棒、质量、可验）。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz), [[arxiv.org]](https://arxiv.org/abs/2306.17439), [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226/), [[usenix.org]](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-446-qu-watermarking.pdf)
6. **安全模型页（新）**：白盒 vs 公验两种；密钥轮换；多域隔离；攻防矩阵（B4、RL、混合、人类释义）。 [[aclanthology.org]](https://aclanthology.org/2025.naacl-long.460/), [[openreview.net]](https://openreview.net/forum?id=iuebm4vXuI)
7. **CTG 场景提示（新）**：摘要/数据到文本的质量‑检测悖论与你的参数折衷策略（动态水印强度）。 [[ojs.aaai.org]](https://ojs.aaai.org/index.php/AAAI/article/view/29756/31301)
8. **产业适配页（新）**：MoE 生态趋势与开源实现路径（HF logits processor Hook + Mixtral/OpenMoE）。 [[developer.nvidia.com]](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/), [[github.com]](https://github.com/jwkirchenbauer/lm-watermarking)

* * *

可能的审稿“尖锐问题”与你的回答模板
------------------

* **Q：检测为何不公开可验？**  
  **A**：我们主打平台内治理（provider‑verifiable）。公开可验（public verifier）可用窗口化统计，但不如白盒重构有功效；这与主流绿色词表水印在**强释义/短文本**下的能力边界一致。 [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz)
* **Q：路由权重真的“语义稳定”吗？**  
  **A**：不同 MoE 有差异。我们报告了“层间/模型间”的稳健性曲线与“ID‑vs‑context”消融；并用 **多层 AND/OR 放大**与签名位冗余提升稳定性。 [[arxiv.org]](https://arxiv.org/abs/2402.01739), [[en.wikipedia.org]](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
* **Q：对黑盒攻防有什么结论？**  
  **A**：在 B4/强释义/混合攻击下，**白盒重构检测**显著优于公验统计；我们给出**所需最小 token 数**与**功效曲线**，并讨论 key rotation 和多域隔离的工程策略。 [[aclanthology.org]](https://aclanthology.org/2025.naacl-long.460/)

* * *

结语：选题价值与路线建议
------------

你的方案抓住了 **MoE 时代的原生信号** 与 **LSH 的语义稳定性**，在**训练‑自由与在线效率**方面优于“外部嵌入型”语义水印；若按上文补齐**强对抗评测**与**理论下界**，并将“零成本”表述精准化，我认为这是一个非常有望在 **ICLR/NeurIPS/ACL（主会）** 获得 **Spotlight/Oral** 的工作方向。

* * *

### 参考与资料（按段落所引）

* 绿色词表水印与可靠性评估（**Kirchenbauer 等**，ICLR 2024；官方代码）： [[openreview.net]](https://openreview.net/forum?id=DEJIDCmWOz), [[github.com]](https://github.com/jwkirchenbauer/lm-watermarking)
* 句级语义水印（**SemStamp**，NAACL 2024）： [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226.pdf), [[aclanthology.org]](https://aclanthology.org/2024.naacl-long.226/), [[arxiv.org]](https://arxiv.org/pdf/2310.03991)
* 鲁棒性再评价/统一评测平台（**WaterPark**，arXiv 2025）： [[arxiv.org]](https://arxiv.org/html/2411.13425v3)
* 黑盒/强攻（**B4**，NAACL 2025；**Attacking by Exploiting Strengths**，CMU 2024）： [[aclanthology.org]](https://aclanthology.org/2025.naacl-long.460/), [[arxiv.org]](https://arxiv.org/html/2402.16187v1)
* 可证明鲁棒/多比特水印（ICLR 2024；USENIX Security 2025）： [[arxiv.org]](https://arxiv.org/abs/2306.17439), [[proceedings.iclr.cc]](https://proceedings.iclr.cc/paper_files/paper/2024/file/beae9ed5316bcc48e616754c06c11875-Paper-Conference.pdf), [[usenix.org]](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-446-qu-watermarking.pdf)
* LSH 原理与碰撞概率（百科/教程）： [[en.wikipedia.org]](https://en.wikipedia.org/wiki/Locality-sensitive_hashing), [[towardsdat…cience.com]](https://towardsdatascience.com/similarity-search-part-5-locality-sensitive-hashing-lsh-76ae4b388203/)
* MoE 路由与稳定化（NeurIPS 2022；OpenMoE 分析；产业趋势）： [[papers.nips.cc]](https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html), [[arxiv.org]](https://arxiv.org/abs/2202.09368), [[arxiv.org]](https://arxiv.org/abs/2402.01739), [[developer.nvidia.com]](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/)
* 检测统计与窗口化指标（社区文档）： [[deepwiki.com]](https://deepwiki.com/bangawayoo/mb-lm-watermarking/7.1-watermark-detection-metrics)
* CTG 水印‑质量悖论与语义感知方案（AAAI 2024）： [[ojs.aaai.org]](https://ojs.aaai.org/index.php/AAAI/article/view/29756/31301)

* * *

**想继续深入**：  
如果你愿意，我可以把上面的**实验计划**直接整理成一个 **.docx 版的实验方案与评测清单**（按你偏好的中文术语、首现括注英文、SoK 风格），并给出**图表草图模板**与**脚本骨架**（HF + vLLM/sglang）。你更倾向首投 **ICLR** 还是 **NeurIPS/ACL**？我们可以针对各会的风格细化摘要与贡献点。
