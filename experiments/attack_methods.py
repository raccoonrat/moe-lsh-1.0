"""
攻击方法模块
实现论文中提到的各种攻击方法：
1. 人类改写 (Human Paraphrasing)
2. LLM 改写 (GPT-4 Paraphrasing)
3. Bigram 改写攻击
4. B4 黑盒清洗
5. 混合攻击
"""

import torch
from typing import List, Dict, Optional
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import re


class ParaphraseAttack:
    """改写攻击基类"""
    
    def __init__(self):
        pass
    
    def attack(self, text: str) -> str:
        """执行攻击，返回改写后的文本"""
        raise NotImplementedError


class HumanParaphraseAttack(ParaphraseAttack):
    """人类改写攻击（需要手动提供改写文本）"""
    
    def __init__(self, paraphrase_pairs: Optional[List[tuple]] = None):
        """
        Args:
            paraphrase_pairs: [(原文, 改写文), ...] 列表
        """
        self.paraphrase_pairs = paraphrase_pairs or []
    
    def attack(self, text: str) -> str:
        """如果提供了改写对，返回对应的改写；否则返回原文"""
        for original, paraphrased in self.paraphrase_pairs:
            if original.strip() == text.strip():
                return paraphrased
        return text


class LLMParaphraseAttack(ParaphraseAttack):
    """使用 LLM 进行改写攻击"""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        num_rounds: int = 1
    ):
        """
        Args:
            model_name: 使用的 LLM 模型名称
            api_key: OpenAI API 密钥
            num_rounds: 改写轮数
        """
        self.model_name = model_name
        self.api_key = api_key
        self.num_rounds = num_rounds
        
        if api_key:
            openai.api_key = api_key
    
    def attack(self, text: str) -> str:
        """使用 LLM 改写文本"""
        if not self.api_key:
            print("Warning: No API key provided, returning original text")
            return text
        
        current_text = text
        for round_num in range(self.num_rounds):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that paraphrases text while preserving its meaning."
                        },
                        {
                            "role": "user",
                            "content": f"Please paraphrase the following text while keeping the same meaning:\n\n{current_text}"
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                current_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error in LLM paraphrasing: {e}")
                break
        
        return current_text


class BigramParaphraseAttack(ParaphraseAttack):
    """Bigram 改写攻击（基于 n-gram 替换）"""
    
    def __init__(self, replacement_prob: float = 0.3):
        """
        Args:
            replacement_prob: 每个 bigram 被替换的概率
        """
        self.replacement_prob = replacement_prob
    
    def attack(self, text: str) -> str:
        """通过替换 bigram 进行改写"""
        words = text.split()
        if len(words) < 2:
            return text
        
        # 随机替换一些词对
        result_words = []
        i = 0
        while i < len(words) - 1:
            if random.random() < self.replacement_prob:
                # 尝试替换 bigram（简化实现：交换相邻词）
                if i + 1 < len(words):
                    words[i], words[i + 1] = words[i + 1], words[i]
                    result_words.append(words[i])
                    i += 1
                else:
                    result_words.append(words[i])
                    i += 1
            else:
                result_words.append(words[i])
                i += 1
        
        if i < len(words):
            result_words.append(words[i])
        
        return " ".join(result_words)


class B4ScrubbingAttack(ParaphraseAttack):
    """B4 黑盒清洗攻击"""
    
    def __init__(
        self,
        scrubbing_model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None
    ):
        """
        Args:
            scrubbing_model_name: 用于清洗的模型
            api_key: API 密钥
        """
        self.scrubbing_model_name = scrubbing_model_name
        self.api_key = api_key
        
        if api_key:
            openai.api_key = api_key
    
    def attack(self, text: str) -> str:
        """使用 B4 方法清洗水印"""
        if not self.api_key:
            print("Warning: No API key provided, returning original text")
            return text
        
        try:
            # B4 方法：要求模型重写文本以移除水印
            response = openai.ChatCompletion.create(
                model=self.scrubbing_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a text rewriting assistant. Rewrite the following text naturally while preserving its meaning."
                    },
                    {
                        "role": "user",
                        "content": f"Please rewrite this text:\n\n{text}"
                    }
                ],
                temperature=0.8,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in B4 scrubbing: {e}")
            return text


class MixedAttack(ParaphraseAttack):
    """混合攻击：结合多种攻击方法"""
    
    def __init__(
        self,
        attacks: List[ParaphraseAttack],
        truncate_prob: float = 0.1,
        mix_prob: float = 0.2
    ):
        """
        Args:
            attacks: 攻击方法列表
            truncate_prob: 截断概率
            mix_prob: 混合概率
        """
        self.attacks = attacks
        self.truncate_prob = truncate_prob
        self.mix_prob = mix_prob
    
    def attack(self, text: str) -> str:
        """执行混合攻击"""
        # 随机选择一个攻击方法
        attack = random.choice(self.attacks)
        attacked_text = attack.attack(text)
        
        # 可能截断
        if random.random() < self.truncate_prob:
            words = attacked_text.split()
            truncate_len = random.randint(len(words) // 2, len(words))
            attacked_text = " ".join(words[:truncate_len])
        
        # 可能混合（简化：随机打乱句子）
        if random.random() < self.mix_prob:
            sentences = re.split(r'[.!?]+', attacked_text)
            random.shuffle(sentences)
            attacked_text = ". ".join(s.strip() for s in sentences if s.strip())
        
        return attacked_text


class AttackEvaluator:
    """攻击评估器"""
    
    def __init__(
        self,
        watermark_detector,
        attack_methods: List[ParaphraseAttack]
    ):
        """
        Args:
            watermark_detector: 水印检测器（MoELSHWatermark 实例）
            attack_methods: 攻击方法列表
        """
        self.detector = watermark_detector
        self.attack_methods = attack_methods
    
    def evaluate_robustness(
        self,
        watermarked_texts: List[str],
        attack_names: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        评估对攻击的鲁棒性
        
        Args:
            watermarked_texts: 带水印的文本列表
            attack_names: 攻击方法名称列表（可选）
            
        Returns:
            评估结果字典
        """
        if attack_names is None:
            attack_names = [f"attack_{i}" for i in range(len(self.attack_methods))]
        
        results = {}
        
        for attack_name, attack_method in zip(attack_names, self.attack_methods):
            detection_scores = []
            detection_rates = []
            
            for text in watermarked_texts:
                # 执行攻击
                attacked_text = attack_method.attack(text)
                
                # 检测水印
                detection_result = self.detector.detect_watermark(attacked_text)
                score = detection_result.get('score', 0.0)
                is_watermarked = detection_result.get('is_watermarked', False)
                
                detection_scores.append(score)
                detection_rates.append(1.0 if is_watermarked else 0.0)
            
            results[attack_name] = {
                'avg_detection_score': sum(detection_scores) / len(detection_scores) if detection_scores else 0.0,
                'detection_rate': sum(detection_rates) / len(detection_rates) if detection_rates else 0.0,
                'num_texts': len(watermarked_texts)
            }
        
        return results

