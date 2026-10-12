from .base import SoftmaxTopKAdapter


class QwenMoeAdapter(SoftmaxTopKAdapter):
    family = "qwen-moe"
    model_types = ("qwen2_moe", "qwen3_moe")
    router_classes = ("Qwen2MoeTopKRouter", "Qwen3MoeTopKRouter")
