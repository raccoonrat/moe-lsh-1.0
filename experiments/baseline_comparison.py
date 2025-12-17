"""
基线方法对比模块
实现与基线水印方法的对比评估
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json


class BaselineWatermark:
    """基线水印方法基类"""
    
    def generate_watermarked_text(self, prompt: str) -> str:
        raise NotImplementedError
    
    def detect_watermark(self, text: str) -> Dict:
        raise NotImplementedError


class KirchenbauerWatermark(BaselineWatermark):
    """Kirchenbauer et al. (2023) 绿色词表水印"""
    
    def __init__(self, model, tokenizer, delta: float = 2.0, gamma: float = 0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.delta = delta
        self.gamma = gamma
        self.vocab_size = len(tokenizer)
    
    def _get_green_list(self, previous_token_id: int) -> set:
        """基于前一个 token 生成绿色词表"""
        import hashlib
        seed = int(hashlib.md5(str(previous_token_id).encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        green_list = set()
        for token_id in range(self.vocab_size):
            if torch.rand(1).item() < self.gamma:
                green_list.add(token_id)
        return green_list
    
    def generate_watermarked_text(self, prompt: str) -> str:
        """生成带水印文本"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids.clone()
        
        previous_token_id = input_ids[0, -1].item()
        
        for _ in range(100):  # 最大生成长度
            # 获取 logits
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]
            
            # 获取绿色词表
            green_list = self._get_green_list(previous_token_id)
            
            # 应用偏置
            for token_id in green_list:
                logits[token_id] += self.delta
            
            # 采样
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=1)
            
            previous_token_id = next_token_id
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def detect_watermark(self, text: str) -> Dict:
        """检测水印"""
        token_ids = self.tokenizer.encode(text, return_tensors="pt")[0]
        
        if len(token_ids) < 2:
            return {"is_watermarked": False, "score": 0.0}
        
        hits = 0
        for i in range(1, len(token_ids)):
            previous_token_id = token_ids[i - 1].item()
            current_token_id = token_ids[i].item()
            
            green_list = self._get_green_list(previous_token_id)
            if current_token_id in green_list:
                hits += 1
        
        total = len(token_ids) - 1
        expected = total * self.gamma
        std = np.sqrt(total * self.gamma * (1 - self.gamma))
        z_score = (hits - expected) / std if std > 0 else 0.0
        
        return {
            "is_watermarked": z_score > 4.0,
            "score": z_score,
            "num_green_tokens": hits,
            "num_tokens": total
        }


class BaselineComparator:
    """基线方法对比器"""
    
    def __init__(
        self,
        our_method,  # MoELSHWatermark
        baseline_methods: List[BaselineWatermark]
    ):
        """
        Args:
            our_method: 我们的方法（MoELSHWatermark）
            baseline_methods: 基线方法列表
        """
        self.our_method = our_method
        self.baseline_methods = baseline_methods
    
    def compare_generation_quality(
        self,
        prompts: List[str],
        reference_texts: Optional[List[str]] = None
    ) -> Dict:
        """
        对比生成质量
        
        Args:
            prompts: 提示列表
            reference_texts: 参考文本列表（用于计算 BLEU/ROUGE）
            
        Returns:
            质量对比结果
        """
        results = {
            'our_method': [],
            'baselines': {f'baseline_{i}': [] for i in range(len(self.baseline_methods))}
        }
        
        for prompt in prompts:
            # 我们的方法
            try:
                our_text = self.our_method.generate_watermarked_text(prompt)
                results['our_method'].append(our_text)
            except Exception as e:
                print(f"Error generating with our method: {e}")
                results['our_method'].append("")
            
            # 基线方法
            for i, baseline in enumerate(self.baseline_methods):
                try:
                    baseline_text = baseline.generate_watermarked_text(prompt)
                    results['baselines'][f'baseline_{i}'].append(baseline_text)
                except Exception as e:
                    print(f"Error generating with baseline {i}: {e}")
                    results['baselines'][f'baseline_{i}'].append("")
        
        return results
    
    def compare_detection_robustness(
        self,
        watermarked_texts: List[str],
        attack_methods: List
    ) -> Dict:
        """
        对比检测鲁棒性
        
        Args:
            watermarked_texts: 带水印文本列表
            attack_methods: 攻击方法列表
            
        Returns:
            鲁棒性对比结果
        """
        results = {
            'our_method': {},
            'baselines': {f'baseline_{i}': {} for i in range(len(self.baseline_methods))}
        }
        
        # 评估我们的方法
        for attack_method in attack_methods:
            attack_name = attack_method.__class__.__name__
            detection_scores = []
            
            for text in watermarked_texts:
                attacked_text = attack_method.attack(text)
                detection_result = self.our_method.detect_watermark(attacked_text)
                detection_scores.append(detection_result.get('score', 0.0))
            
            results['our_method'][attack_name] = {
                'avg_score': np.mean(detection_scores) if detection_scores else 0.0,
                'detection_rate': sum(1 for s in detection_scores if s > 4.0) / len(detection_scores) if detection_scores else 0.0
            }
        
        # 评估基线方法
        for i, baseline in enumerate(self.baseline_methods):
            for attack_method in attack_methods:
                attack_name = attack_method.__class__.__name__
                detection_scores = []
                
                for text in watermarked_texts:
                    attacked_text = attack_method.attack(text)
                    detection_result = baseline.detect_watermark(attacked_text)
                    detection_scores.append(detection_result.get('score', 0.0))
                
                results['baselines'][f'baseline_{i}'][attack_name] = {
                    'avg_score': np.mean(detection_scores) if detection_scores else 0.0,
                    'detection_rate': sum(1 for s in detection_scores if s > 4.0) / len(detection_scores) if detection_scores else 0.0
                }
        
        return results
    
    def save_comparison_results(
        self,
        results: Dict,
        output_path: Path
    ):
        """保存对比结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化格式
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (dict, list)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

