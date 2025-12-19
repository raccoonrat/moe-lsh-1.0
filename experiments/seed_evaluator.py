"""
种子敏感性评估模块
评估不同随机种子对水印性能的影响
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import json
from tqdm import tqdm
import hashlib

from moe_lsh.MoELSHWatermark import MoELSHWatermark, MoELSHWatermarkConfig
from experiments.geometric_metrics import GeometricQualityMetrics


class SeedEvaluator:
    """种子敏感性评估器"""
    
    def __init__(
        self,
        watermark: MoELSHWatermark,
        geometric_metrics: GeometricQualityMetrics,
        calibration_texts: List[str],
        device: str = "cuda"
    ):
        """
        Args:
            watermark: MoE LSH 水印实例
            geometric_metrics: 几何质量指标计算器
            calibration_texts: 校准文本列表
            device: 计算设备
        """
        self.watermark = watermark
        self.geometric_metrics = geometric_metrics
        self.calibration_texts = calibration_texts
        self.device = device
    
    def generate_seeds(self, num_seeds: int, seed_base: int = 42) -> List[str]:
        """生成多个随机种子"""
        seeds = []
        for i in range(num_seeds):
            seed_str = f"seed_{seed_base}_{i}"
            seeds.append(seed_str)
        return seeds
    
    def evaluate_single_seed(
        self,
        seed: str,
        test_prompts: List[str],
        num_generations: int = 10
    ) -> Dict:
        """
        评估单个种子的性能
        
        Args:
            seed: 随机种子字符串
            test_prompts: 测试提示列表
            num_generations: 每个提示的生成数量
            
        Returns:
            评估结果字典
        """
        # 更新水印的 secret_key（这会改变 LSH 投影矩阵）
        original_key = self.watermark.encoder.secret_key
        self.watermark.encoder.secret_key = seed
        
        # 重新初始化 LSH 编码器（使用新种子）
        self._reinitialize_encoder(seed)
        
        results = {
            'seed': seed,
            'detection_scores': [],
            'generation_quality': [],
            'geometric_metrics': {},
            'green_list_stats': {}
        }
        
        # 生成带水印文本并检测
        all_generated_texts = []
        for prompt in test_prompts[:num_generations]:
            try:
                watermarked_text = self.watermark.generate_watermarked_text(prompt)
                all_generated_texts.append(watermarked_text)
                
                # 检测水印
                detection_result = self.watermark.detect_watermark(watermarked_text)
                results['detection_scores'].append(detection_result.get('score', 0.0))
            except Exception as e:
                print(f"Error generating with seed {seed}: {e}")
                continue
        
        # 计算平均检测分数
        if results['detection_scores']:
            results['avg_detection_score'] = np.mean(results['detection_scores'])
            results['detection_rate'] = sum(1 for s in results['detection_scores'] if s > 4.0) / len(results['detection_scores'])
        else:
            results['avg_detection_score'] = 0.0
            results['detection_rate'] = 0.0
        
        # 计算几何质量指标
        green_list = self._get_current_green_list()
        results['green_list_stats'] = {
            'size': len(green_list),
            'fraction': len(green_list) / self.watermark.encoder.vocab_size
        }
        
        # 使用校准集计算几何指标
        calibration_embeddings = self._get_calibration_embeddings()
        projection_vector = self._get_projection_vector()
        
        geometric_results = self.geometric_metrics.evaluate_seed_quality(
            green_list=green_list,
            projection_vector=projection_vector,
            calibration_embeddings=calibration_embeddings
        )
        results['geometric_metrics'] = geometric_results
        
        # 恢复原始密钥
        self.watermark.encoder.secret_key = original_key
        
        return results
    
    def _reinitialize_encoder(self, seed: str):
        """使用新种子重新初始化 LSH 编码器"""
        config = self.watermark.config.encoder_config.copy()
        config['secret_key'] = seed
        
        vocab_size = self.watermark.encoder.vocab_size
        embedding_matrix = self.watermark.encoder.embedding_matrix
        
        from moe_lsh.MoELSHWatermark import LSH_Semantic_Encoder
        self.watermark.encoder = LSH_Semantic_Encoder(
            config=config,
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix
        )
    
    def _get_current_green_list(self) -> Set[int]:
        """获取当前种子的绿色词表"""
        # 使用一个虚拟的路由权重向量来获取绿色词表
        dummy_rw = torch.zeros(
            self.watermark.encoder.input_dim,
            device=self.device
        )
        green_list = self.watermark.encoder.get_green_list(dummy_rw)
        return green_list
    
    def _get_projection_vector(self) -> Optional[torch.Tensor]:
        """获取当前 LSH 投影向量"""
        if hasattr(self.watermark.encoder, 'random_vectors'):
            # 返回第一个投影向量作为代表
            return self.watermark.encoder.random_vectors[0]
        return None
    
    def _get_calibration_embeddings(self) -> torch.Tensor:
        """从校准文本中提取嵌入向量"""
        # 简化实现：返回空张量
        # 实际应该从校准文本中提取路由权重或嵌入
        return torch.empty(0, self.watermark.encoder.input_dim, device=self.device)
    
    def evaluate_multiple_seeds(
        self,
        seeds: List[str],
        test_prompts: List[str],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Dict]:
        """
        评估多个种子的性能
        
        Args:
            seeds: 种子列表
            test_prompts: 测试提示列表
            output_dir: 结果保存目录
            
        Returns:
            所有种子的评估结果
        """
        all_results = {}
        
        for seed in tqdm(seeds, desc="Evaluating seeds"):
            try:
                result = self.evaluate_single_seed(seed, test_prompts)
                all_results[seed] = result
            except Exception as e:
                print(f"Error evaluating seed {seed}: {e}")
                continue
        
        # 保存结果
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "seed_evaluation_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        return all_results
    
    def analyze_seed_sensitivity(
        self,
        results: Dict[str, Dict]
    ) -> Dict:
        """
        分析种子敏感性
        
        Returns:
            敏感性分析结果
        """
        detection_scores = [r['avg_detection_score'] for r in results.values()]
        split_entropies = [r['geometric_metrics'].get('split_entropy', 0.0) 
                          for r in results.values()]
        
        analysis = {
            'num_seeds': len(results),
            'detection_score_stats': {
                'mean': np.mean(detection_scores),
                'std': np.std(detection_scores),
                'min': np.min(detection_scores),
                'max': np.max(detection_scores),
                'median': np.median(detection_scores)
            },
            'split_entropy_stats': {
                'mean': np.mean(split_entropies),
                'std': np.std(split_entropies),
                'min': np.min(split_entropies),
                'max': np.max(split_entropies)
            },
            'high_quality_seeds': [],
            'low_quality_seeds': []
        }
        
        # 找出高质量和低质量种子
        score_threshold = np.percentile(detection_scores, 75)
        entropy_threshold = np.percentile(split_entropies, 25)
        
        for seed, result in results.items():
            score = result['avg_detection_score']
            entropy = result['geometric_metrics'].get('split_entropy', 0.0)
            
            if score >= score_threshold and entropy >= entropy_threshold:
                analysis['high_quality_seeds'].append({
                    'seed': seed,
                    'detection_score': score,
                    'split_entropy': entropy
                })
            elif score < score_threshold or entropy < entropy_threshold:
                analysis['low_quality_seeds'].append({
                    'seed': seed,
                    'detection_score': score,
                    'split_entropy': entropy
                })
        
        return analysis

