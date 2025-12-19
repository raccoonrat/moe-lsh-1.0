"""
基础类定义
用于水印框架的基类
"""

from typing import Dict, Optional, Any


class BaseConfig:
    """基础配置类"""
    def __init__(self, config_dict: Optional[Dict] = None, *args, **kwargs):
        if config_dict is None:
            config_dict = {}
        self.config_dict = config_dict
        self.initialize_parameters()
    
    def initialize_parameters(self) -> None:
        """初始化参数，子类应该重写这个方法"""
        pass
    
    @property
    def algorithm_name(self) -> str:
        """返回算法名称，子类应该重写这个方法"""
        return "BaseWatermark"
    
    def __getattr__(self, name: str) -> Any:
        """允许通过属性访问配置字典中的值"""
        if hasattr(self, 'config_dict') and name in self.config_dict:
            return self.config_dict[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class BaseWatermark:
    """基础水印类"""
    def __init__(self, *args, **kwargs):
        pass
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """生成带水印的文本，子类应该重写这个方法"""
        raise NotImplementedError("Subclass must implement generate_watermarked_text")
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """检测文本中的水印，子类应该重写这个方法"""
        raise NotImplementedError("Subclass must implement detect_watermark")

