from .base import LogitsTopKAdapter


class GraniteMoeAdapter(LogitsTopKAdapter):
    family = "granite-moe"
    model_types = ("granitemoe", "granitemoeshared")
    router_classes = ("GraniteMoeTopKRouter", "GraniteMoeSharedTopKRouter")
