import pytest
import torch
import torch.distributed as dist

from inference.model import Transformer
from tests.fixtures.test_configs import get_distributed_config
from tests.utils import (create_test_model, generate_test_tokens,
                         validate_model_output)


class TestDistributed:

    @pytest.mark.distributed
    def test_single_gpu_as_distributed(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Distributed testing requires CUDA")

        args = get_distributed_config()

        model = create_test_model(args, test_device)
        tokens = generate_test_tokens(
            batch_size=1, seq_len=32, vocab_size=args.vocab_size, device=test_device
        )

        with torch.inference_mode():
            output = model(tokens)

        expected_shape = (1, args.vocab_size)
        validate_model_output(output, expected_shape, args.vocab_size)

    @pytest.mark.distributed
    def test_distributed_model_structure(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Distributed testing requires CUDA")

        args = get_distributed_config()
        model = create_test_model(args, test_device)

        assert hasattr(model, "embed"), "Model should have embedding layer"
        assert hasattr(
            model.embed, "part_vocab_size"
        ), "Embedding should have parallel vocab size"

        for layer in model.layers:
            if hasattr(layer.ffn, "experts"):
                assert hasattr(
                    layer.ffn, "n_local_experts"
                ), "MoE should have local expert count"
                assert hasattr(
                    layer.ffn, "experts_start_idx"
                ), "MoE should have expert start index"

    @pytest.mark.distributed
    def test_parallel_embedding_single_process(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Distributed testing requires CUDA")

        args = get_distributed_config()
        model = create_test_model(args, test_device)

        vocab_size = args.vocab_size
        tokens = torch.randint(0, vocab_size, (2, 16), device=test_device)

        embeddings = model.embed(tokens)

        assert embeddings.shape == (2, 16, args.dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    @pytest.mark.distributed
    def test_moe_expert_routing_single_process(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Distributed testing requires CUDA")

        args = get_distributed_config()
        model = create_test_model(args, test_device)

        x = torch.randn(4, args.dim, device=test_device, dtype=torch.bfloat16)

        for layer in model.layers:
            if hasattr(layer.ffn, "gate"):
                weights, indices = layer.ffn.gate(x)

                assert weights.shape[0] == x.shape[0]
                assert weights.shape[1] == args.n_activated_experts
                assert indices.shape[0] == x.shape[0]
                assert indices.shape[1] == args.n_activated_experts

                assert (indices >= 0).all()
                assert (indices < args.n_routed_experts).all()

                assert torch.allclose(
                    weights.sum(dim=1),
                    torch.ones(x.shape[0], device=test_device),
                    atol=1e-5,
                )

    @pytest.mark.distributed
    def test_distributed_inference_consistency(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Distributed testing requires CUDA")

        args = get_distributed_config()

        torch.manual_seed(42)
        model1 = create_test_model(args, test_device)

        torch.manual_seed(42)
        model2 = create_test_model(args, test_device)

        tokens = generate_test_tokens(
            batch_size=1, seq_len=32, vocab_size=args.vocab_size, device=test_device
        )

        torch.manual_seed(123)
        with torch.inference_mode():
            output1 = model1(tokens)

        torch.manual_seed(123)
        with torch.inference_mode():
            output2 = model2(tokens)

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)

    @pytest.mark.distributed
    def test_column_parallel_linear(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Distributed testing requires CUDA")

        args = get_distributed_config()
        model = create_test_model(args, test_device)

        for layer in model.layers:
            if hasattr(layer.attn, "wq"):
                x = torch.randn(
                    2, 16, args.dim, device=test_device, dtype=torch.bfloat16
                )
                output = layer.attn.wq(x)

                expected_out_features = args.n_heads * (
                    args.qk_nope_head_dim + args.qk_rope_head_dim
                )
                assert output.shape == (2, 16, expected_out_features)

    @pytest.mark.distributed
    def test_row_parallel_linear(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Distributed testing requires CUDA")

        args = get_distributed_config()
        model = create_test_model(args, test_device)

        for layer in model.layers:
            if hasattr(layer.attn, "wo"):
                x = torch.randn(
                    2,
                    16,
                    args.n_heads * args.v_head_dim,
                    device=test_device,
                    dtype=torch.bfloat16,
                )
                output = layer.attn.wo(x)

                assert output.shape == (2, 16, args.dim)

    @pytest.mark.distributed
    def test_distributed_memory_efficiency(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Distributed testing requires CUDA")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        args = get_distributed_config()
        model = create_test_model(args, test_device)

        tokens = generate_test_tokens(
            batch_size=args.max_batch_size,
            seq_len=64,
            vocab_size=args.vocab_size,
            device=test_device,
        )

        with torch.inference_mode():
            output = model(tokens)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        assert peak_memory > 0, "Should use GPU memory"
        assert peak_memory < 8.0, f"Memory usage seems too high: {peak_memory:.2f} GB"
