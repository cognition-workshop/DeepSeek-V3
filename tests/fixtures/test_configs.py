from inference.model import ModelArgs


def get_tiny_config() -> ModelArgs:
    return ModelArgs(
        max_batch_size=1,
        max_seq_len=64,
        dtype="bf16",
        vocab_size=512,
        dim=128,
        inter_dim=256,
        moe_inter_dim=64,
        n_layers=1,
        n_dense_layers=1,
        n_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=1,
        q_lora_rank=0,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
    )


def get_small_config() -> ModelArgs:
    return ModelArgs(
        max_batch_size=2,
        max_seq_len=128,
        dtype="bf16",
        vocab_size=1024,
        dim=256,
        inter_dim=512,
        moe_inter_dim=128,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=0,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
    )


def get_fp8_config() -> ModelArgs:
    config = get_small_config()
    config.dtype = "fp8"
    return config


def get_distributed_config() -> ModelArgs:
    return ModelArgs(
        max_batch_size=4,
        max_seq_len=256,
        dtype="bf16",
        vocab_size=2048,
        dim=512,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=4,
        n_dense_layers=1,
        n_heads=8,
        n_routed_experts=16,
        n_shared_experts=2,
        n_activated_experts=4,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
    )
