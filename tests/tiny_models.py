"""Tiny random-init MoE models, one per supported family. No downloads."""

import torch

TINY = dict(vocab_size=128, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)


def tiny_model(family: str, seed: int = 0):
    from transformers import (
        GraniteMoeConfig,
        GraniteMoeForCausalLM,
        MixtralConfig,
        MixtralForCausalLM,
        OlmoeConfig,
        OlmoeForCausalLM,
        Qwen2MoeConfig,
        Qwen2MoeForCausalLM,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    torch.manual_seed(seed)
    if family == "granite":
        cfg = GraniteMoeConfig(**TINY, intermediate_size=16, num_hidden_layers=2, num_local_experts=8, num_experts_per_tok=2)
        return GraniteMoeForCausalLM(cfg)
    if family == "qwen3":  # layer 1 is dense, so the trace has 2 MoE layers out of 3
        cfg = Qwen3MoeConfig(**TINY, intermediate_size=32, moe_intermediate_size=16, num_hidden_layers=3, head_dim=8,
                             num_experts=8, num_experts_per_tok=2, mlp_only_layers=[1])
        return Qwen3MoeForCausalLM(cfg)
    if family == "qwen2":
        cfg = Qwen2MoeConfig(**TINY, intermediate_size=32, moe_intermediate_size=16, shared_expert_intermediate_size=16,
                             num_hidden_layers=2, num_experts=8, num_experts_per_tok=2)
        return Qwen2MoeForCausalLM(cfg)
    if family == "olmoe":
        cfg = OlmoeConfig(**TINY, intermediate_size=16, num_hidden_layers=2, num_experts=8, num_experts_per_tok=2)
        return OlmoeForCausalLM(cfg)
    if family == "mixtral":
        cfg = MixtralConfig(**TINY, intermediate_size=16, num_hidden_layers=2, num_local_experts=8, num_experts_per_tok=2)
        return MixtralForCausalLM(cfg)
    raise ValueError(family)


FAMILIES = ["granite", "qwen3", "qwen2", "olmoe", "mixtral"]
