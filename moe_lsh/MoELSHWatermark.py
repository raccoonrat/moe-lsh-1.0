import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats
from typing import List, Optional, Dict
import hashlib

# 从基础模块导入基类
from .base import BaseWatermark, BaseConfig
from .utils.transformers_config import TransformersConfig

# =============================================================================
#  辅助类 (我们的核心逻辑)
# =============================================================================

class MoE_RW_Extractor:
    """(重构版) 负责从MoE模型中提取路由权重向量。"""
    def __init__(self, model, config: dict, device="cuda"): # 移除了未使用的tokenizer
        self.model = model
        self.device = device
        self.start_layer = config.get('start_layer_idx', 6)
        self.end_layer = config.get('end_layer_idx', 23)
        self.use_top_k = config.get('use_top_k', True)
        self.k = config.get('k', 6)
        self.token_idx = config.get('token_idx', -1)

    def extract_from_embs(self, sent_embs: torch.Tensor) -> torch.Tensor:
        """ 核心处理逻辑，从已有的router weights张量中提取RW向量。"""
        num_moe_layers = sent_embs.shape[1]
        if not (0 <= self.start_layer < self.end_layer <= num_moe_layers):
            raise ValueError(f"Invalid layer slice specified: [{self.start_layer}, {self.end_layer})")
            
        sliced_layer_embs = sent_embs[:, self.start_layer:self.end_layer, :, :]
        selected_token_rws = sliced_layer_embs[:, :, self.token_idx, :]
        
        if self.use_top_k:
            topk_weights, top_k_indices = torch.topk(selected_token_rws, self.k, dim=-1)
            processed_rws = torch.zeros_like(selected_token_rws)
            processed_rws.scatter_(dim=-1, index=top_k_indices, src=topk_weights)
        else:
            processed_rws = selected_token_rws
            
        rw_vectors = processed_rws.flatten(start_dim=1).cpu()
        return rw_vectors

    def get_rw_vector(self, input_ids: torch.Tensor, past_key_values=None):
        """ 保持原有接口，用于独立调用。"""
        with torch.no_grad():
            outputs, sent_embs = self.model(
                input_ids, 
                past_key_values=past_key_values,
                use_cache=True,
                output_router_weights=True
            )
        new_past_key_values = outputs.past_key_values
        rw_vectors = self.extract_from_embs(sent_embs)
        return rw_vectors, new_past_key_values

class LSH_Semantic_Encoder:
    """(最终版) 实现LSH签名生成和“按位贡献”语义词汇池。"""        
    def __init__(self, config: dict, vocab_size: int, embedding_matrix: torch.Tensor):
        self.input_dim = config['input_dim']
        self.num_bits = config.get('num_bits', 256)
        self.pool_size = config.get('pool_size', 10)
        self.secret_key = config.get('secret_key', 'default_key')
        self.semantic_pools = config.get('semantic_pools', True)
        self.vocab_size = vocab_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_matrix = embedding_matrix.to(self.device)
        with torch.no_grad():
            self.normed_embeddings = self.embedding_matrix / self.embedding_matrix.norm(dim=1, keepdim=True)
            self.normed_embeddings.nan_to_num_(0.0) # 将 NaN 替换为 0
        master_seed = self._seed(f"{self.secret_key}_lsh")
        torch.manual_seed(master_seed)
        self.random_vectors = torch.randn(self.num_bits, self.input_dim).to(self.device)
        self.pools = self._precompute_pools()
        
    @staticmethod
    def _seed(s: str) -> int:
        return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], 'little')
        
    def _precompute_pools(self):
        pools, g = [], torch.Generator()          
        for i in range(self.num_bits):
            pool_seed = self._seed(f"{self.secret_key}_pool_{i}")
            g.manual_seed(pool_seed)
            if self.semantic_pools:
                anchor_idx = torch.randint(0, self.vocab_size, (1,), generator=g).item()
                anchor_embedding = self.embedding_matrix[anchor_idx].unsqueeze(0)
                anchor_norm = anchor_embedding / anchor_embedding.norm()
                anchor_norm.nan_to_num_(0.0)
                emb_norm = self.normed_embeddings
                sims = (anchor_norm * emb_norm).sum(dim=1)
                _, top_indices = torch.topk(sims, self.pool_size, largest=True)
                pools.append(set(top_indices.cpu().tolist()))
            else:
                pool_indices = torch.randperm(self.vocab_size, generator=g)[:self.pool_size]
                pools.append(set(pool_indices.tolist()))
        return pools

    def get_signature(self, rw_vector: torch.Tensor):
        rw_vector = rw_vector.to(self.device, dtype=self.random_vectors.dtype)
        dot_products = self.random_vectors.mv(rw_vector)
        return (dot_products > 0).int()

    def get_green_list(self, rw_vector: torch.Tensor):
        signature = self.get_signature(rw_vector)
        green_list = set()
        for i, bit in enumerate(signature):
            if bit == 1:
                green_list.update(self.pools[i])
        return green_list

class WatermarkGeneratorContext:
    """(重构版) 使用猴子补丁和上下文管理器的生成辅助类。"""
    def __init__(self, model, extractor: MoE_RW_Extractor, encoder: LSH_Semantic_Encoder, delta: float):
        self.model = model
        self.extractor = extractor
        self.encoder = encoder
        self.delta = delta
        self.original_forward = None

    def _watermarked_forward(self, *args, **kwargs):
        # 1. 调用原始的forward方法
        kwargs['output_router_weights'] = True
        outputs, sent_embs = self.original_forward(*args, **kwargs)
        
        if sent_embs is None:
            return outputs

        # 2. 直接调用extractor的核心逻辑
        rw_vectors = self.extractor.extract_from_embs(sent_embs)

        # 3. 生成绿色名单并修改logits (循环逻辑保持不变，因为是务实的选择)
        logits = outputs.logits
        for i in range(logits.shape[0]): # 对于小batch size是可接受的
            rw_vector = rw_vectors[i]
            green_list = self.encoder.get_green_list(rw_vector)
            if green_list:
                green_list_tensor = torch.tensor(list(green_list), device=logits.device, dtype=torch.long)
                logits[i, -1, green_list_tensor] += self.delta
        
        return outputs

    def __enter__(self):
        self.original_forward = self.model.forward
        self.model.forward = self._watermarked_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_forward is not None:
            self.model.forward = self.original_forward

# =============================================================================
#  与框架集成的类 (最终版本)
# =============================================================================

class MoELSHWatermarkConfig(BaseConfig):
    """MoE-LSH水印的配置类 (无改动)"""
    def __init__(self, config_dict: Optional[Dict] = None, transformers_config=None, *args, **kwargs):
        if config_dict is None:
            config_dict = {}
        super().__init__(config_dict, *args, **kwargs)
        self.transformers_config = transformers_config
    
    def initialize_parameters(self) -> None:
        self.extractor_config = self.config_dict.get('extractor_config', {})
        self.encoder_config = self.config_dict.get('encoder_config', {})
        self.delta = self.config_dict.get('watermark_strength_delta', 2.0)
        self.z_threshold = self.config_dict.get('detection_z_threshold', 4.0)
        # 模型和tokenizer从config_dict中获取
        self.generation_model = self.config_dict.get('generation_model')
        self.generation_tokenizer = self.config_dict.get('generation_tokenizer')
        self.device = self.config_dict.get('device', 'cuda')
        self.vocab_size = self.config_dict.get('vocab_size', 0)
        self.gen_kwargs = self.config_dict.get('gen_kwargs', {})
    
    @property
    def algorithm_name(self) -> str:
        return 'MoELSHWatermark'

class MoELSHWatermark(BaseWatermark):
    """MoE-LSH语义水印方案 (终极性能版)"""
    def __init__(self, algorithm_config: str | MoELSHWatermarkConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs):
        if isinstance(algorithm_config, str):
            self.config = MoELSHWatermarkConfig(algorithm_config, transformers_config, *args, **kwargs)
        elif isinstance(algorithm_config, MoELSHWatermarkConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a MoELSHWatermarkConfig instance")

        self.extractor = MoE_RW_Extractor(
            self.config.generation_model,
            self.config.extractor_config,
            self.config.device
        )

        # 尝试不同的属性名来获取专家数量（不同模型可能使用不同的属性名）
        model_config = self.config.generation_model.config
        num_experts = (
            getattr(model_config, 'n_routed_experts', None) or
            getattr(model_config, 'num_experts', None) or
            getattr(model_config, 'num_local_experts', None)
        )
        if num_experts is None:
            # 如果都找不到，尝试从模型名称或其他方式推断，或使用默认值
            # 对于 Qwen2MoE，通常是60个专家
            if 'qwen' in str(type(model_config)).lower() or 'qwen' in str(type(self.config.generation_model)).lower():
                num_experts = 60  # Qwen2MoE 默认专家数
            else:
                raise AttributeError("Model config does not have expert count attribute. Is this an MoE model?")

        num_sliced_layers = self.extractor.end_layer - self.extractor.start_layer
        self.config.encoder_config['input_dim'] = num_sliced_layers * num_experts

        vocab_size = self.config.vocab_size
        embedding_matrix = self.config.generation_model.get_input_embeddings().weight.detach().float()


        self.encoder = LSH_Semantic_Encoder(
            config=self.config.encoder_config, vocab_size=vocab_size,
            embedding_matrix=embedding_matrix
        )

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """(终极性能版) 使用猴子补丁和上下文管理器生成带水印的文本。"""
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        
        # 使用with语句来临时“激活”水印能力
        with WatermarkGeneratorContext(
            model=self.config.generation_model,
            extractor=self.extractor,
            encoder=self.encoder,
            delta=self.config.delta
        ):
            # 在这个代码块内，model.forward被临时替换了
            encoded_watermarked_text = self.config.generation_model.generate(
                **encoded_prompt, 
                **self.config.gen_kwargs
            )

        # (健壮性优化) 处理可能为空的输出
        if encoded_watermarked_text.shape[1] == 0:
            return ""
            
        # 离开with代码块后，model.forward会自动恢复原状
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """(终极性能版) 检测文本中的MoE-LSH水印，利用KV缓存。"""
        prompt = kwargs.get('prompt', '')
        
        # 1. 准备Token ID
        full_ids = self.config.generation_tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.config.device)
            
        if prompt:
            prompt_ids = self.config.generation_tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.config.device)
            if len(full_ids) >= len(prompt_ids) and torch.equal(full_ids[:len(prompt_ids)], prompt_ids):
                generated_ids = full_ids[len(prompt_ids):]
            else: # Fallback
                prompt_ids = torch.tensor([], dtype=torch.long, device=self.config.device)
                generated_ids = full_ids
        else:
            prompt_ids = torch.tensor([], dtype=torch.long, device=self.config.device)
            generated_ids = full_ids
        
        if len(generated_ids) == 0:
            return self._create_empty_result(return_dict)

        # 2. 预热KV缓存
        past_key_values = None
        if len(prompt_ids) > 0:
            with torch.no_grad():
                outputs = self.config.generation_model(prompt_ids.unsqueeze(0), use_cache=True)
                past_key_values = outputs.past_key_values
            current_input_token = prompt_ids[-1:].unsqueeze(0)
        else:
            # (逻辑优化) 无prompt时，使用BOS token作为第一个输入
            bos_token_id = self.config.generation_tokenizer.bos_token_id
            if bos_token_id is None: # Fallback
                bos_token_id = self.config.generation_model.config.bos_token_id
            current_input_token = torch.tensor([[bos_token_id]], device=self.config.device)
        # 3. 逐词元增量检测 (O(N) 性能)
        hits = 0
        for i in range(len(generated_ids)):
            target_token = generated_ids[i].item()
            rw_vector, past_key_values = self.extractor.get_rw_vector(current_input_token, past_key_values)
            green_list = self.encoder.get_green_list(rw_vector.squeeze(0))
            if target_token in green_list:
                hits += 1
            current_input_token = generated_ids[i:i+1].unsqueeze(0)

        # 4. 统计裁决
        total_tokens = len(generated_ids)
        avg_green_list_size = 0.5 * self.encoder.num_bits * self.encoder.pool_size
        expected_fraction = avg_green_list_size / self.encoder.vocab_size
        numerator = hits - expected_fraction * total_tokens
        denominator = np.sqrt(total_tokens * expected_fraction * (1 - expected_fraction))
        z_score = numerator / denominator if denominator > 0 else 0.0
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score, "num_green_tokens": hits, "num_tokens": total_tokens}
        else:
            return (is_watermarked, z_score)

    def _create_empty_result(self, return_dict):
        if return_dict:
            return {"is_watermarked": False, "score": 0.0, "num_green_tokens":0, "num_tokens":0}
        else:
            return (False, 0.0)