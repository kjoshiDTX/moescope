from .base import SoftmaxTopKAdapter


class OlmoeAdapter(SoftmaxTopKAdapter):
    family = "olmoe"
    model_types = ("olmoe",)
    router_classes = ("OlmoeTopKRouter",)
