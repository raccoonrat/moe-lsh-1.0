"""
几何质量指标模块
实现论文中提出的三类几何质量指标：
1. 语义簇分割熵 (Semantic Cluster Split Entropy)
2. Logits 分布的 Wasserstein 距离
3. 投影方差与 PCA 对齐度
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')


class GeometricQualityMetrics:
    """几何质量指标计算器"""
    
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        vocab_size: int,
        num_clusters: int = 100,
        device: str = "cuda"
    ):
        """
        Args:
            embedding_matrix: 词表嵌入矩阵 [vocab_size, embed_dim]
            vocab_size: 词表大小
            num_clusters: 语义聚类数量
            device: 计算设备
        """
        self.embedding_matrix = embedding_matrix.to(device)
        self.vocab_size = vocab_size
        self.num_clusters = num_clusters
        self.device = device
        
        # 归一化嵌入
        with torch.no_grad():
            self.normed_embeddings = self.embedding_matrix / (
                self.embedding_matrix.norm(dim=1, keepdim=True) + 1e-8
            )
            self.normed_embeddings.nan_to_num_(0.0)
        
        # 预计算语义聚类
        self.semantic_clusters = self._compute_semantic_clusters()
        
        # 预计算 PCA 主成分
        self.pca_components = self._compute_pca_components()
    
    def _compute_semantic_clusters(self) -> Dict[int, Set[int]]:
        """使用 k-means 对词表进行语义聚类"""
        embeddings_np = self.normed_embeddings.cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        clusters = {}
        for cluster_id in range(self.num_clusters):
            token_indices = set(np.where(cluster_labels == cluster_id)[0].tolist())
            clusters[cluster_id] = token_indices
        
        return clusters
    
    def _compute_pca_components(self, n_components: int = 10) -> torch.Tensor:
        """计算嵌入空间的主成分"""
        embeddings_np = self.normed_embeddings.cpu().numpy()
        
        pca = PCA(n_components=n_components)
        pca.fit(embeddings_np)
        
        components = torch.from_numpy(pca.components_).to(self.device)  # [n_components, embed_dim]
        return components
    
    def compute_split_entropy(
        self,
        green_list: Set[int]
    ) -> float:
        """
        计算语义簇分割熵
        
        Args:
            green_list: 绿色词表集合
            
        Returns:
            Score_split: 分割熵得分，范围 [0, 1]，越高越好
        """
        if len(self.semantic_clusters) == 0:
            return 0.0
        
        r_values = []
        for cluster_id, cluster_tokens in self.semantic_clusters.items():
            if len(cluster_tokens) == 0:
                continue
            
            green_in_cluster = len(cluster_tokens & green_list)
            r_i = green_in_cluster / len(cluster_tokens)
            r_values.append(r_i)
        
        if len(r_values) == 0:
            return 0.0
        
        # Score_split = 1 - (1/K) * sum((2|r_i - 0.5|)^2)
        squared_deviations = [(2 * abs(r - 0.5)) ** 2 for r in r_values]
        score = 1.0 - np.mean(squared_deviations)
        
        return max(0.0, min(1.0, score))
    
    def compute_wasserstein_distance(
        self,
        original_logits: torch.Tensor,
        watermarked_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> float:
        """
        计算 Logits 分布的 Wasserstein-1 距离
        
        Args:
            original_logits: 原始 logits [vocab_size]
            watermarked_logits: 水印后 logits [vocab_size]
            temperature: Softmax 温度
            
        Returns:
            W1 距离，越小越好
        """
        # 转换为概率分布
        P = torch.softmax(original_logits / temperature, dim=0).cpu().numpy()
        Q = torch.softmax(watermarked_logits / temperature, dim=0).cpu().numpy()
        
        # 计算语义距离矩阵
        embeddings_np = self.normed_embeddings.cpu().numpy()
        distance_matrix = cdist(embeddings_np, embeddings_np, metric='euclidean')
        
        # 使用 scipy 的 wasserstein_distance（简化版，实际应该用最优传输）
        # 这里使用近似：计算每个 token 的期望语义距离
        w1_approx = 0.0
        for i in range(self.vocab_size):
            if P[i] > 0 or Q[i] > 0:
                # 找到最接近的 token
                if P[i] > 0:
                    # 计算从 i 到所有 j 的加权距离
                    weighted_dist = np.sum(distance_matrix[i, :] * Q)
                    w1_approx += P[i] * weighted_dist
        
        return float(w1_approx)
    
    def compute_projection_variance(
        self,
        projection_vector: torch.Tensor,
        calibration_embeddings: torch.Tensor
    ) -> float:
        """
        计算投影方差
        
        Args:
            projection_vector: 投影向量 [input_dim]
            calibration_embeddings: 校准集嵌入 [N, input_dim]
            
        Returns:
            投影方差，越大越好
        """
        if calibration_embeddings.shape[0] == 0:
            return 0.0
        
        # h = X @ r
        projections = calibration_embeddings @ projection_vector
        variance = float(projections.var().item())
        
        return variance
    
    def compute_pca_alignment(
        self,
        projection_vector: torch.Tensor,
        k: int = 10
    ) -> float:
        """
        计算 PCA 对齐度
        
        Args:
            projection_vector: 投影向量 [input_dim]
            k: 使用前 k 个主成分
            
        Returns:
            对齐度得分，范围 [0, 1]，越大越好
        """
        if projection_vector.dim() > 1:
            projection_vector = projection_vector.flatten()
        
        # 归一化投影向量
        proj_norm = projection_vector / (projection_vector.norm() + 1e-8)
        
        # 计算与前 k 个主成分的余弦相似度平方和
        alignment = 0.0
        k_actual = min(k, self.pca_components.shape[0])
        
        for i in range(k_actual):
            component = self.pca_components[i]  # [embed_dim]
            component_norm = component / (component.norm() + 1e-8)
            
            # 如果投影向量维度与主成分不匹配，需要适配
            if proj_norm.shape[0] != component_norm.shape[0]:
                # 假设投影向量是路由权重的扁平化，需要映射到嵌入空间
                # 这里简化处理：如果维度不匹配，返回 0
                continue
            
            cos_sim = (proj_norm * component_norm).sum()
            alignment += cos_sim ** 2
        
        return float(alignment / k_actual) if k_actual > 0 else 0.0
    
    def evaluate_seed_quality(
        self,
        green_list: Set[int],
        original_logits: torch.Tensor = None,
        watermarked_logits: torch.Tensor = None,
        projection_vector: torch.Tensor = None,
        calibration_embeddings: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        综合评估种子质量
        
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 1. 分割熵
        metrics['split_entropy'] = self.compute_split_entropy(green_list)
        
        # 2. Wasserstein 距离（如果提供了 logits）
        if original_logits is not None and watermarked_logits is not None:
            metrics['wasserstein_distance'] = self.compute_wasserstein_distance(
                original_logits, watermarked_logits
            )
        
        # 3. 投影方差（如果提供了投影向量和校准集）
        if projection_vector is not None and calibration_embeddings is not None:
            metrics['projection_variance'] = self.compute_projection_variance(
                projection_vector, calibration_embeddings
            )
        
        # 4. PCA 对齐度（如果提供了投影向量）
        if projection_vector is not None:
            metrics['pca_alignment'] = self.compute_pca_alignment(projection_vector)
        
        return metrics

