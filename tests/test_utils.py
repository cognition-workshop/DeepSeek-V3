import pytest
import torch
import math
from model import precompute_freqs_cis, apply_rotary_emb, ModelArgs


class TestRotaryEmbeddingUtils:
    def test_find_correction_dim_basic(self):
        args = ModelArgs()
        freqs_cis = precompute_freqs_cis(args)
        
        assert freqs_cis.shape[0] == args.max_seq_len
        assert freqs_cis.shape[1] == args.qk_rope_head_dim // 2

    def test_find_correction_range_edge_cases(self):
        args = ModelArgs(
            max_seq_len=8192,
            original_seq_len=4096,
            qk_rope_head_dim=64,
            beta_fast=32,
            beta_slow=1
        )
        
        freqs_cis = precompute_freqs_cis(args)
        assert freqs_cis.shape == (8192, 32)

    def test_linear_ramp_factor_edge_cases(self):
        args = ModelArgs(
            max_seq_len=8192,
            original_seq_len=4096,
            qk_rope_head_dim=64
        )
        
        freqs_cis = precompute_freqs_cis(args)
        assert not torch.isnan(freqs_cis).any()
        assert not torch.isinf(freqs_cis).any()

    def test_rotary_embedding_consistency(self):
        args = ModelArgs(max_seq_len=32, qk_rope_head_dim=16)
        freqs_cis = precompute_freqs_cis(args)
        
        x = torch.randn(2, 16, 4, 16)
        
        output1 = apply_rotary_emb(x, freqs_cis[:16])
        output2 = apply_rotary_emb(x, freqs_cis[:16])
        
        assert torch.allclose(output1, output2)

    def test_rotary_embedding_different_positions(self):
        args = ModelArgs(max_seq_len=32, qk_rope_head_dim=16)
        freqs_cis = precompute_freqs_cis(args)
        
        x = torch.randn(1, 1, 4, 16)
        
        output_pos_0 = apply_rotary_emb(x, freqs_cis[0:1])
        output_pos_5 = apply_rotary_emb(x, freqs_cis[5:6])
        
        assert not torch.allclose(output_pos_0, output_pos_5)

    def test_rotary_embedding_shape_preservation(self):
        args = ModelArgs(qk_rope_head_dim=32)
        freqs_cis = precompute_freqs_cis(args)
        
        test_shapes = [
            (1, 1, 1, 32),
            (2, 8, 4, 32),
            (4, 16, 8, 32),
        ]
        
        for shape in test_shapes:
            x = torch.randn(*shape)
            seq_len = shape[1]
            output = apply_rotary_emb(x, freqs_cis[:seq_len])
            assert output.shape == shape

    def test_rotary_embedding_dtype_preservation(self):
        args = ModelArgs(qk_rope_head_dim=16)
        freqs_cis = precompute_freqs_cis(args)
        
        dtypes = [torch.float32, torch.bfloat16]
        if torch.cuda.is_available():
            dtypes.append(torch.float16)
        
        for dtype in dtypes:
            x = torch.randn(2, 8, 4, 16, dtype=dtype)
            output = apply_rotary_emb(x, freqs_cis[:8])
            assert output.dtype == dtype

    def test_rope_theta_parameter_effect(self):
        args1 = ModelArgs(rope_theta=10000.0, max_seq_len=16, qk_rope_head_dim=8)
        args2 = ModelArgs(rope_theta=20000.0, max_seq_len=16, qk_rope_head_dim=8)
        
        freqs_cis1 = precompute_freqs_cis(args1)
        freqs_cis2 = precompute_freqs_cis(args2)
        
        assert not torch.allclose(freqs_cis1, freqs_cis2)

    def test_rope_factor_scaling(self):
        args = ModelArgs(
            max_seq_len=8192,
            original_seq_len=4096,
            rope_factor=2.0,
            qk_rope_head_dim=16
        )
        
        freqs_cis = precompute_freqs_cis(args)
        assert freqs_cis.shape == (8192, 8)
        assert not torch.isnan(freqs_cis).any()


class TestModelArgsValidation:
    def test_moe_expert_relationships(self):
        args = ModelArgs()
        
        assert args.n_activated_experts <= args.n_routed_experts
        assert args.n_expert_groups >= 1
        assert args.n_limited_groups <= args.n_expert_groups

    def test_attention_dimension_relationships(self):
        args = ModelArgs()
        
        qk_total_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        assert qk_total_dim > 0
        assert args.v_head_dim > 0
        assert args.kv_lora_rank >= 0
        assert args.q_lora_rank >= 0

    def test_sequence_length_relationships(self):
        args = ModelArgs()
        
        assert args.max_seq_len > 0
        assert args.original_seq_len > 0
        assert args.max_batch_size > 0

    def test_dimension_relationships(self):
        args = ModelArgs()
        
        assert args.dim > 0
        assert args.inter_dim > 0
        assert args.moe_inter_dim > 0
        assert args.vocab_size > 0
        assert args.n_heads > 0
        assert args.n_layers > 0

    def test_rope_parameters(self):
        args = ModelArgs()
        
        assert args.rope_theta > 0
        assert args.rope_factor > 0
        assert args.beta_fast > 0
        assert args.beta_slow > 0
        assert args.mscale > 0

    def test_score_function_values(self):
        args = ModelArgs()
        assert args.score_func in ["softmax", "sigmoid"]

    def test_dtype_values(self):
        args = ModelArgs()
        assert args.dtype in ["bf16", "fp8"]


class TestNumericalStability:
    def test_large_input_values(self):
        args = ModelArgs(qk_rope_head_dim=64, max_seq_len=16)
        
        x = torch.randn(2, 8, 4, 64) * 100  # 4D tensor for rotary embedding
        freqs_cis = precompute_freqs_cis(args)
        
        output = apply_rotary_emb(x, freqs_cis[:8])
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_small_input_values(self):
        args = ModelArgs(qk_rope_head_dim=64, max_seq_len=16)
        
        x = torch.randn(2, 8, 4, 64) * 1e-6  # 4D tensor for rotary embedding
        freqs_cis = precompute_freqs_cis(args)
        
        output = apply_rotary_emb(x, freqs_cis[:8])
        assert not torch.isnan(output).any()

    def test_zero_input_handling(self):
        args = ModelArgs(qk_rope_head_dim=64, max_seq_len=16)
        
        x = torch.zeros(2, 8, 4, 64)  # 4D tensor for rotary embedding
        freqs_cis = precompute_freqs_cis(args)
        
        output = apply_rotary_emb(x, freqs_cis[:8])
        assert torch.allclose(output, torch.zeros_like(output))

    def test_extreme_sequence_lengths(self):
        args = ModelArgs(max_seq_len=1, qk_rope_head_dim=4)
        freqs_cis = precompute_freqs_cis(args)
        assert freqs_cis.shape == (1, 2)
        
        args = ModelArgs(max_seq_len=65536, qk_rope_head_dim=4)
        freqs_cis = precompute_freqs_cis(args)
        assert freqs_cis.shape == (65536, 2)
        assert not torch.isnan(freqs_cis).any()
