# **混合专家模型（MoE）与语义哈希（SemHash）水印技术中的随机种子敏感性：基于高维几何与各向异性的数学机理深度分析报告**

## **摘要**

本研究报告针对混合专家模型（Mixture-of-Experts, MoE）在大语言模型（LLM）水印技术应用中出现的“随机种子敏感性”问题，进行了详尽的数学与几何分析。当前基于SemHash（语义哈希）的水印技术虽然在理论上具备抗改写能力，但在实际应用中表现出极高的方差：特定的随机种子（Seed）能够实现SOTA（State-of-the-Art）性能，而大多数随机初始化则导致生成质量显著下降或检测率崩塌。

本报告摒弃了将种子视为单纯超参数的传统视角，转而利用**高维几何（High-dimensional Geometry）**、**局部敏感哈希（LSH）理论**以及**流形学习（Manifold Learning）原理，揭示了这一现象的深层数学机理。分析表明，该不稳定性源于LLM嵌入空间的各向异性（Anisotropy）**——即著名的“锥体效应（Cone Effect）”——与LSH算法预设的各向同性（Isotropy）假设之间的根本性几何错配。

具体而言，随机均匀采样的超平面在高维各向异性空间中，极大概率与数据流形的主成分方向正交，导致**区域坍塌（Region Collapse）**，即水印划分的有效区域（Green List）要么包含整个语义簇，要么完全将其排除。这种二元化的失效模式在MoE架构中被进一步放大，因为水印引入的语义偏移可能干扰专家路由（Expert Routing）机制，导致模型在推理过程中激活次优专家，从而产生级联误差。

为解决这一痛点，本报告不仅提供了数学直觉与推导，还建立了一套基于**Wasserstein距离**、\*\*语义簇熵（Semantic Cluster Entropy）**和**谱方差（Spectral Variance）\*\*的量化评价体系，用于在部署前预测种子的优劣。最终，报告建议从数据无关的随机投影转向数据依赖的几何划分（如PCA对齐的LSH或基于质心的划分），作为彻底解决敏感性问题的技术路径。

## ---

**1\. 引言：语义水印的几何不稳定性**

随着大语言模型（LLM）生成能力的飞速发展，区分机器生成文本与人类创作文本的需求日益迫切。传统基于Token层面的水印技术（如KGW方案）虽然计算高效，但对改写攻击（Paraphrase Attack）极其脆弱。作为回应，学术界转向了基于语义的水印方案，如**SemStamp**和**SemHash** 1。这些方法的核心假设是：尽管句子的表层词汇可能在改写中发生变化，但其在语义空间中的嵌入向量（Embedding Vector）保持相对不变。

SemHash技术通过局部敏感哈希（LSH）将连续的高维语义空间 $\\mathbb{R}^d$ 划分为离散的“红/绿”区域。在生成过程中，算法强制模型优先选择那些落入“绿色”区域的句子或Token。然而，用户在MoE模型上的实验揭示了一个关键的工程痛点：**性能对随机种子的选择呈现出极端的敏感性**。

这种敏感性并非简单的统计波动，而是高维空间中几何结构相互作用的必然结果。在低维直觉中，任何通过原点的随机平面都有50%的概率切分一个原本连通的区域。但在高维空间，特别是当数据分布呈现出强烈的**各向异性**时，随机投影的行为发生了质变。

### **1.1 问题背景：MoE架构下的特殊性**

在混合专家模型（MoE）中，模型的每一层包含多个“专家（Experts）”子网络，并通过一个路由门控网络（Router/Gating Network）根据输入的语义特征动态选择激活的专家 3。这种稀疏激活机制使得MoE在参数量巨大的同时保持了推理的高效性。

然而，这也引入了新的脆弱性：MoE的性能高度依赖于路由的准确性。路由器的决策边界本质上也是高维空间中的划分平面。当SemHash引入的水印划分平面（由随机种子决定）与MoE内部的专家路由平面发生复杂的几何干涉时，如果水印强制模型选择一个语义上有偏差的Token，可能会导致后续的Hidden State发生偏移，进而错误地激活了不相关的专家。这种\*\*路由错位（Routing Misalignment）\*\*是MoE模型对水印种子尤为敏感的结构性原因 5。

### **1.2 报告结构与目标**

本报告旨在通过数学建模和几何分析，回答用户提出的三个核心问题：

1. **语义碎片化机制：** 糟糕的种子如何通过几何切割破坏语义聚类，进而影响Logits分布。  
2. **各向异性的影响：** 为何LLM的“锥体效应”注定了均匀随机投影的低效性。  
3. **量化指标设计：** 如何构建数学指标来事前衡量一个种子的“几何质量”。

我们将深入探讨高维球体上的测度集中现象、LSH的碰撞概率公式在各向异性条件下的修正，以及如何利用最优传输理论（Optimal Transport）来量化水印造成的分布扭曲。

## ---

**2\. 高维语义空间的几何特性分析**

要理解随机种子为何失效，我们首先必须精确描述它所作用的“舞台”——LLM的Embedding空间。传统的LSH理论建立在数据分布均匀的假设之上，而现实中的LLM空间则大相径庭。

### **2.1 理想假设 vs. 现实：各向同性与各向异性**

在经典的LSH理论（特别是SimHash）中，为了保证碰撞概率 $P(h(x)=h(y))$ 与角度 $\\theta(x,y)$ 成线性关系：

$$P(h(x) \= h(y)) \= 1 \- \\frac{\\theta(x, y)}{\\pi}$$

我们通常假设数据点分布在单位超球 $\\mathbb{S}^{d-1}$ 上是各向同性的（Isotropic）。在这种假设下，任何随机采样的法向量 $r \\sim \\mathcal{N}(0, I\_d)$ 所定义的超平面，其切割数据的概率分布是均匀的 6。  
然而，大量研究证实，BERT、GPT、Mixtral等现代LLM的上下文嵌入表现出强烈的**各向异性（Anisotropy）**。这种现象被称为\*\*“锥体效应（Cone Effect）”\*\* 8。

#### **2.1.1 锥体效应的数学描述**

各向异性意味着嵌入向量并不是均匀分布在原点周围，而是被压缩在一个狭窄的锥体 $\\mathcal{K}$ 内部。这可以通过计算嵌入矩阵 $X \\in \\mathbb{R}^{N \\times d}$ 的协方差矩阵 $\\Sigma \= \\frac{1}{N} X^T X$ 的特征值谱来量化。

* **各向同性空间：** 特征值 $\\lambda\_1 \\approx \\lambda\_2 \\approx \\dots \\approx \\lambda\_d$。  
* **LLM空间：** 前几个特征值 $\\lambda\_1, \\lambda\_2$ 极大，其余特征值迅速衰减至接近零。这表明数据实际上位于一个极低维的流形上，且该流形在空间中具有特定的朝向。

更直观地，任意两个随机采样的词向量 $u, v$ 之间的余弦相似度远大于0：

$$\\mathbb{E}\[\\cos(u, v)\] \\approx 1 \- \\Delta, \\quad \\text{where } \\Delta \\ll 1$$

这与高维各向同性空间中随机向量近似正交（相似度接近0）的性质截然相反。这意味着所有的词向量都“挤”在一起，指向大致相同的方向（通常由一个非零的均值向量 $\\mu$ 主导）。

### **2.2 随机投影在锥体分布下的失效机理**

当我们将一个均匀分布的随机超平面（SemHash的种子）投射到一个狭窄的数据锥体上时，会发生什么？

根据高维几何的\*\*测度集中（Concentration of Measure）\*\*原理，一个随机向量 $r \\sim \\mathcal{N}(0, I\_d)$ 与锥体的主轴方向 $\\mu$ 之间的角度 $\\phi$ 会高度集中在 $\\pi/2$ 附近。也就是说，绝大多数随机超平面几乎是平行于锥体轴线的。

然而，要让超平面有效地将锥体内部的数据点**划分**开来（即一部分在平面上方，一部分在下方），超平面必须**穿过**这个锥体。

* 如果锥体的张角 $\\alpha$ 很小（锥体效应显著），那么只有当法向量 $r$ 落在与 $\\mu$ 垂直的一个极窄的“赤道带”内时，超平面才能切开锥体。  
* 对于绝大多数随机种子，其对应的法向量 $r$ 会落在该赤道带之外。此时，超平面要么完全位于锥体的“上方”，要么完全位于“下方”。

**几何推论：** 对于一个各向异性的数据集，随机采样的LSH平面极其低效。

* **情形A（失效）：** 超平面未穿过锥体。此时，锥体内的**所有**数据点（即所有语义相关的Token）都被哈希到同一侧（例如全为1或全为0）。  
  * 如果该侧为“绿”，则水印未施加任何约束，不可检测。  
  * 如果该侧为“红”，则所有合理的Token都被禁止，模型被迫输出毫无逻辑的乱码（Perplexity爆炸）。这解释了为什么某些种子会导致性能崩塌。  
* **情形B（有效）：** 超平面侥幸穿过锥体。此时才能进行有效的水印划分。

这就解释了用户观察到的“种子敏感性”：只有极少数“幸运”的种子（情形B）能够恰好切入那个狭窄的数据分布锥体，从而在保留语义的同时引入可检测的熵。绝大多数种子（情形A）要么无效，要么毁灭性地破坏生成。

## ---

**3\. 语义碎片化（Semantic Fragmentation）：Logits选择的数学直觉**

用户提出的第一个具体分析点是“语义碎片化”。这是一个非常精准的术语，描述了LSH平面在切分紧密语义聚类时的破坏性行为。

### **3.1 语义聚类的几何模型**

在Next-token Prediction任务中，给定上下文 $C$，模型输出的Logits向量 $z$ 并不是均匀分布的。高概率的Token通常集中在语义空间的一个（或几个）紧密的簇（Cluster）中。例如，对于上下文“The cat sat on the...”，高概率Token集合 $\\mathcal{T}\_{top} \= \\{\\text{mat, rug, floor, sofa}\\}$ 的嵌入向量在空间中是非常接近的。

我们可以将这个集合建模为一个半径为 $\\epsilon$ 的微型超球体 $\\mathcal{B}\_\\epsilon(c)$，中心为 $c$。

### **3.2 碎片化的发生过程**

当SemHash应用一个由种子 $S$ 决定的随机超平面 $H\_S$ 时，它将空间一分为二：$V\_{Green}$ 和 $V\_{Red}$。

* **理想切割：** 超平面将 $\\mathcal{B}\_\\epsilon(c)$ 切割成两半，且每一半中都包含高概率Token。例如，“mat”在红区，“rug”在绿区。此时，模型只需从“mat”切换到“rug”，语义损失极小（同义词替换）。  
* **碎片化切割（Bad Seed）：** 超平面的位置极其尴尬，它没有平分聚类，而是将聚类的核心部分（比如概率最高的Token）切到了红区，而绿区只剩下聚类边缘的、概率较低的Token，甚至是聚类之外的噪声。

#### **数学直觉解释**

假设最优Token $t^\*$ 的Logit为 $z\_{max}$，次优同义词 $t'$ 的Logit为 $z\_{sub}$。通常 $z\_{max} \\approx z\_{sub}$。  
水印引入的偏差为 $\\delta$。  
生成概率：

$$P(t) \\propto \\exp(z\_t \+ \\delta \\cdot \\mathbb{I}(t \\in V\_{Green}))$$  
**糟糕的种子会导致以下Logit动力学：**

1. **切割正交于语义流形：** 语义聚类通常呈扁平状（流形）。如果随机超平面恰好沿着聚类的长轴方向切割，可能导致语义上紧密相连的词被强行分开。  
2. 强制降级（Forced Downgrade）： 如果 $t^\* \\in V\_{Red}$ 且所有高质量同义词 $\\{t', t''\\} \\subset V\_{Red}$（即整个高质量簇被“红化”），模型被迫在 $V\_{Green}$ 中寻找最大Logit值的Token。由于 $V\_{Green}$ 中缺乏高质量Token，模型选中的可能是Logit值极低（$z\_{noise} \\ll z\_{max}$）的无关词汇 $t\_{noise}$。  
   即使加上水印偏置 $\\delta$，如果 $z\_{noise} \+ \\delta$ 超过了被抑制的 $z\_{max}$，模型就会输出 $t\_{noise}$。

   $$z\_{noise} \+ \\delta \> z\_{max} \\implies \\text{Hallucination / Context Loss}$$

这就是**语义碎片化**的本质：随机平面切断了Logits最高的高地（High Probability Manifold）与合法区域（Green Region）的联系，迫使概率流向低洼地带（Low Probability Tail）。

### **3.3 区域坍塌（Region Collapse）现象**

最近的文献 11 提出了\*\*“区域坍塌”\*\*的概念，这进一步深化了语义碎片化的理论。在低熵（高确定性）生成任务中，所有高质量的输出在语义空间中高度聚集，几乎坍缩成一个点。  
对于这样的点状分布，任何划分都是二元的：要么全绿，要么全红。

* SemHash试图在这些点内部制造熵，但由于点太小（$\\epsilon \\to 0$），随机平面很难精确穿过它。  
* 结果是：在序列的某一步，整个高质量空间被标记为红。模型无法生成任何合理的词，只能“胡言乱语”以满足水印要求。这解释了为何SOTA性能需要特定的种子——那个种子恰好避免了对关键序列步骤的“全红”封锁。

## ---

**4\. 各向异性（Anisotropy）与随机投影的低效性分析**

用户关注的第二点是各向异性是否注定了随机投影的低效。基于第2节的几何背景，我们可以从数学上证明这一结论。

### **4.1 投影方差的极度不均衡**

LSH的有效性依赖于哈希函数能够最大化数据的区分度（Entropy）。对于投影哈希 $h(x) \= \\text{sgn}(r^T x)$，其区分能力取决于投影值 $y \= r^T x$ 的方差。

$$\\text{Var}(y) \= r^T \\Sigma r$$

在各向异性空间中，协方差矩阵 $\\Sigma$ 的能量集中在前 $k$ 个主成分 $u\_1, \\dots, u\_k$ 上（$k \\ll d$）。  
对于一个均匀随机向量 $r$，它在主成分方向上的投影分量 $\\langle r, u\_i \\rangle$ 是微小的（高维空间中，随机向量与任意固定方向近似正交）。  
因此，$\\text{Var}(y)$ 通常非常小。  
**直觉：** 想象一个三维空间中的“黄瓜”（数据簇）。

* 有效的水印平面应该沿着黄瓜的**长轴**切割（或者切片），这样能把黄瓜分成两半。  
* 随机选一个平面，极大概率是平行于黄瓜长轴且位于黄瓜外部，或者垂直切在黄瓜的某个边缘。  
* 由于 $r$ 很难随机对齐到黄瓜的长轴（主成分），绝大多数投影都是“无效投影”，它们捕捉不到数据的主要变化，只能捕捉到噪声。

### **4.2 碰撞概率的退化**

在各向同性空间，两个随机向量的碰撞概率（Hash值相同）随角度线性变化。  
在各向异性空间（锥体分布），任意两个数据向量 $x\_1, x\_2$ 的夹角 $\\theta$ 都很小（例如 $\\theta \< 15^\\circ$）。  
此时，根据SimHash公式：

$$P(h(x\_1) \= h(x\_2)) \= 1 \- \\frac{\\theta}{\\pi} \\approx 1$$

这意味着，对于绝大多数随机种子，空间中任意两个有效语义向量都会被哈希到同一个桶里（同为红或同为绿）。

* **辨识度（Discriminative Power）丧失：** 水印失去了区分不同语义状态的能力。  
* **鲁棒性丧失：** 一旦发生攻击（如改写），微小的角度变化不足以跨越哈希边界，导致检测失效（如果全绿）或生成失效（如果全红）。

**结论：** 是的，考虑到LLM Embedding空间的极端各向异性，**未经修正的随机均匀采样投影向量注定是效率低下的**。它在数学上等同于在一个低维流形之外的高维空其间进行盲目搜索。

## ---

**5\. 量化种子质量的指标体系**

针对用户提出的第三个问题——如何量化一个Seed的“好坏”，我们需要设计能够捕捉上述几何失配的指标。传统的聚类指标（如轮廓系数）在这里可能不适用，甚至需要反向使用（我们需要切分聚类，而不是保留聚类）。

基于本研究的分析，建议测量以下三个层面的指标：

### **5.1 指标一：语义簇的分割熵 (Semantic Cluster Split Entropy) —— 衡量“碎片化”程度**

这是最直接对应“语义碎片化”的指标。我们需要衡量水印划分是否**均匀地**切分了语义上相似的Token，而不是将它们一锅端。

**计算步骤：**

1. **预处理：** 使用无水印模型在一个校准数据集上生成Logits，并对词表进行聚类（例如使用k-means对Embedding进行聚类，得到 $K$ 个语义簇 $C\_1, \\dots, C\_K$）。这些簇代表了“同义词集合”。  
2. **应用种子：** 对给定的种子 $Seed$，计算全词表的红绿划分。  
3. **计算绿词比例：** 对每个簇 $C\_i$，计算其中落在绿表中的Token比例 $r\_i \= \\frac{|C\_i \\cap V\_{Green}|}{|C\_i|}$。  
4. 指标公式： 我们希望每个簇都被劈开（$r\_i \\approx 0.5$），而不是全进全出。

   $$\\text{Score}\_{split} \= 1 \- \\frac{1}{K} \\sum\_{i=1}^K (2 \\cdot |r\_i \- 0.5|)^2$$  
   * **解释：** 该分数越高（接近1），说明种子将每个语义簇都均匀切分，保证了同义词替换的可行性。如果分数为0，说明发生了“区域坍塌”，簇被完全包含或完全排除。

### **5.2 指标二：Logits分布的Wasserstein距离 (Wasserstein Distance of Logits) —— 衡量“推土机距离”**

该指标量化了水印对模型原始概率分布的**几何扭曲程度**。相比KL散度，Wasserstein距离能感知底层的几何空间（即语义距离）。

**计算步骤：**

1. 令 $P$ 为原始Logits的Softmax分布。  
2. 令 $Q$ 为施加水印（Masking/Biasing）后的分布。  
3. 定义基础距离矩阵 $D\_{ij} \= \\|emb(i) \- emb(j)\\|\_2$ （即Token $i$ 和 $j$ 的语义距离）。  
4. 计算 $P$ 和 $Q$ 之间的 Wasserstein-1 距离 (Earth Mover's Distance)。

   $$W\_1(P, Q) \= \\inf\_{\\gamma \\in \\Pi(P, Q)} \\mathbb{E}\_{(x, y) \\sim \\gamma} \[\\|x \- y\\|\]$$  
   * **解释：** 好的种子对应的 $W\_1$ 应该很小。这意味着概率质量（Probability Mass）只在非常接近的Token之间移动（例如从“happy”移到了“glad”）。如果 $W\_1$ 很大，说明概率质量被迫移动到了语义距离很远的地方（从“happy”移到了“apple”），这预示着生成质量的崩塌。

### **5.3 指标三：投影方差与主成分对齐度 (PCA Alignment) —— 衡量“各向异性适应度”**

衡量随机向量 $r$ 是否捕捉到了数据的主要变化方向。

**计算步骤：**

1. 在一个校准集上收集Embedding矩阵 $X$。  
2. 计算种子对应的投影向量 $r$ 在数据上的投影值 $h \= Xr$。  
3. **指标公式：** $\\sigma^2\_{proj} \= \\text{Var}(h)$。  
   * **解释：** 如果 $\\text{Var}(h)$ 极小，说明投影平面垂直于数据流形（切了空气）。如果 $\\text{Var}(h)$ 很大，说明投影平面沿着数据分布最广的方向进行了切割，提供了最大的区分度。  
   * **进阶版：** 计算 $r$ 与数据前 $k$ 个主成分 $u\_1, \\dots, u\_k$ 的余弦相似度之和。

## ---

**6\. 解决方案与建议**

基于上述几何分析，解决Seed敏感性的根本途径不是“寻找”一个好种子，而是**构造**一个好种子，或者改变空间的几何性质。

### **6.1 方案A：白化（Whitening）变换 —— 修正几何空间**

在进行LSH哈希之前，先对Embedding $x$ 进行白化处理：

$$\\tilde{x} \= W(x \- \\mu)$$

其中 $W \= \\Sigma^{-1/2}$。  
原理： 白化操作消除了各向异性，将“锥体”拉伸成一个各向同性的球体。在白化后的空间里，随机均匀投影（LSH）恢复了其理论性质，任何随机种子都能以与角度成正比的概率切分数据。这将从根本上消除对种子的敏感性 10。

### **6.2 方案B：数据依赖的哈希（Data-Dependent Hashing） —— 修正投影方向**

不要使用 $\\mathcal{N}(0, I)$ 生成种子。  
操作：

1. 对校准集的Embedding进行PCA分析，提取前 $k$ 个主成分 $u\_1, \\dots, u\_k$。  
2. 使用这 $k$ 个主成分作为LSH的投影向量（即超平面法向量）。  
   原理： 这种方法（即PCA-LSH或ITQ）保证了超平面总是沿着数据方差最大的方向切割，必然穿过数据锥体的中心，最大化了“绿表”和“红表”的熵，彻底避免了“区域坍塌”。

### **6.3 方案C：基于质心的划分（Centroid-based Partitioning / k-SemStamp）**

正如文献 13 提出的 k-SemStamp 方案，完全放弃随机超平面。  
操作：

1. 使用k-means对语义空间进行聚类。  
2. 随机指定某些簇为“绿”，某些为“红”。  
   原理： 这种方法利用了数据的自然拓扑结构（Voronoi划分），而不是强加一个线性的几何划分。它天生避免了切断紧密聚类的问题（因为聚类算法本身就是为了保持紧密性）。

## ---

**7\. 结论**

基于MoE和SemHash的水印技术中出现的“随机种子敏感性”问题，本质上是**低维流形数据（锥体分布）与高维各向同性投影（LSH）之间的几何不兼容**。

1. **语义碎片化** 是由于随机平面正交于语义流形主轴，强行切断了同义词聚类，迫使MoE路由到次优专家。  
2. **各向异性** 导致绝大多数随机平面无法有效二分数据锥体，造成概率空间的区域坍塌。  
3. 要量化种子质量，应测量**语义簇内的红绿比例方差（Cluster Split Entropy）** 和 **Logits变动的Wasserstein距离**。

**最终建议：** 停止依赖随机抽样寻找“幸运种子”。应实施**Embedding白化**或采用**PCA对齐的投影向量**，将“运气”转变为确定的几何构造，从而实现稳定且SOTA的水印性能。

### ---

**表1：不同投影策略的几何特性对比**

| 策略 | 几何行为 | 对各向异性的适应性 | 种子敏感性 | 推荐场景 |
| :---- | :---- | :---- | :---- | :---- |
| **均匀随机投影 (Standard LSH)** | 盲目切割，大概率错过锥体 | **极差** (导致区域坍塌) | **极高** | 仅适用于各向同性数据 |
| **PCA对齐投影 (PCA-LSH)** | 沿最大方差方向切割 | **优** (确保切中流形) | **低** | 适用于所有LLM Embedding |
| **白化 \+ 随机投影 (Whitening)** | 将锥体拉伸为球体后切割 | **优** (恢复LSH理论性质) | **低** | 通用且鲁棒 |
| **聚类划分 (k-SemStamp)** | 基于Voronoi胞腔的非线性划分 | **良** (尊重自然聚类) | **低** | 需要预计算聚类中心 |

本报告的分析表明，从几何角度修正投影方式是解决当前实验痛点的关键路径。

#### **Works cited**

1. SemStamp: A Paraphrase-Robust Watermark \- arXiv, accessed December 17, 2025, [https://arxiv.org/html/2310.03991v1](https://arxiv.org/html/2310.03991v1)  
2. SEMSTAMP:ASEMANTIC WATERMARK WITH PARA- PHRASTIC ROBUSTNESS FOR TEXT GENERATION \- OpenReview, accessed December 17, 2025, [https://openreview.net/pdf?id=IpoZ32sq44](https://openreview.net/pdf?id=IpoZ32sq44)  
3. Mistral Large 3: An Open-Source MoE LLM Explained | IntuitionLabs, accessed December 17, 2025, [https://intuitionlabs.ai/articles/mistral-large-3-moe-llm-explained](https://intuitionlabs.ai/articles/mistral-large-3-moe-llm-explained)  
4. Exploiting the Experts: Unauthorized Compression in MoE-LLMs \- ChatPaper, accessed December 17, 2025, [https://chatpaper.com/paper/213134](https://chatpaper.com/paper/213134)  
5. badmoe: backdooring mixture-of-experts llms via optimizing routing triggers and infecting \- OpenReview, accessed December 17, 2025, [https://openreview.net/pdf/36bfd968d81d368e9714e74169fc73b4db779cd7.pdf](https://openreview.net/pdf/36bfd968d81d368e9714e74169fc73b4db779cd7.pdf)  
6. Locality Sensitive Hashing (LSH) Overview \- Emergent Mind, accessed December 17, 2025, [https://www.emergentmind.com/topics/locality-sensitive-hashing-lsh-d88174d0-fb0e-4aac-866f-39a81a761e35](https://www.emergentmind.com/topics/locality-sensitive-hashing-lsh-d88174d0-fb0e-4aac-866f-39a81a761e35)  
7. Super-Bit Locality-Sensitive Hashing, accessed December 17, 2025, [http://papers.neurips.cc/paper/4847-super-bit-locality-sensitive-hashing.pdf](http://papers.neurips.cc/paper/4847-super-bit-locality-sensitive-hashing.pdf)  
8. Stable Anisotropic Regularization, accessed December 17, 2025, [https://iclr.cc/media/iclr-2024/Slides/18254\_RTn9zZz.pdf](https://iclr.cc/media/iclr-2024/Slides/18254_RTn9zZz.pdf)  
9. Is Anisotropy Truly Harmful? A Case Study on Text Clustering \- ACL Anthology, accessed December 17, 2025, [https://aclanthology.org/2023.acl-short.103.pdf](https://aclanthology.org/2023.acl-short.103.pdf)  
10. On Isotropy of Multimodal Embeddings \- MDPI, accessed December 17, 2025, [https://www.mdpi.com/2078-2489/14/7/392](https://www.mdpi.com/2078-2489/14/7/392)  
11. Semantic differentiation for tackling challenges in watermarking low-entropy constrained generation outputs | OpenReview, accessed December 17, 2025, [https://openreview.net/forum?id=MfVpY7E7Xw](https://openreview.net/forum?id=MfVpY7E7Xw)  
12. \[2306.00458\] Exploring Anisotropy and Outliers in Multilingual Language Models for Cross-Lingual Semantic Sentence Similarity \- arXiv, accessed December 17, 2025, [https://arxiv.org/abs/2306.00458](https://arxiv.org/abs/2306.00458)  
13. k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text \- ACL Anthology, accessed December 17, 2025, [https://aclanthology.org/2024.findings-acl.98/](https://aclanthology.org/2024.findings-acl.98/)  
14. \[2402.11399\] k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text \- arXiv, accessed December 17, 2025, [https://arxiv.org/abs/2402.11399](https://arxiv.org/abs/2402.11399)