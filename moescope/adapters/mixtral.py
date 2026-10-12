from .base import SoftmaxTopKAdapter


class MixtralAdapter(SoftmaxTopKAdapter):
    family = "mixtral"
    model_types = ("mixtral",)
    router_classes = ("MixtralTopKRouter",)
    always_renormalize = True
