import pytest
import torch

from inference.model import Transformer
from tests.fixtures.test_configs import get_fp8_config, get_small_config
from tests.utils import create_test_model, generate_test_tokens, validate_model_output


class TestQuantization:

    def test_bf16_inference(self, setup_torch, test_device):
        args = get_small_config()
        args.dtype = "bf16"

        model = create_test_model(args, test_device)
        tokens = generate_test_tokens(
            batch_size=1, seq_len=32, vocab_size=args.vocab_size, device=test_device
        )

        with torch.inference_mode():
            output = model(tokens)

        expected_shape = (1, args.vocab_size)
        validate_model_output(output, expected_shape, args.vocab_size)

        for param in model.parameters():
            if param.element_size() > 1:
                assert (
                    param.dtype == torch.bfloat16
                ), f"Expected bfloat16, got {param.dtype}"

    @pytest.mark.gpu
    def test_fp8_inference(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("FP8 testing requires CUDA")

        args = get_fp8_config()
        model = create_test_model(args, test_device)
        tokens = generate_test_tokens(
            batch_size=1, seq_len=32, vocab_size=args.vocab_size, device=test_device
        )

        with torch.inference_mode():
            output = model(tokens)

        expected_shape = (1, args.vocab_size)
        validate_model_output(output, expected_shape, args.vocab_size)

    @pytest.mark.gpu
    def test_quantization_consistency(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Quantization consistency testing requires CUDA")

        args_bf16 = get_small_config()
        args_bf16.dtype = "bf16"

        args_fp8 = get_small_config()
        args_fp8.dtype = "fp8"

        torch.manual_seed(42)
        model_bf16 = create_test_model(args_bf16, test_device)

        torch.manual_seed(42)
        model_fp8 = create_test_model(args_fp8, test_device)

        tokens = generate_test_tokens(
            batch_size=1,
            seq_len=16,
            vocab_size=args_bf16.vocab_size,
            device=test_device,
        )

        with torch.inference_mode():
            output_bf16 = model_bf16(tokens)
            output_fp8 = model_fp8(tokens)

        assert output_bf16.shape == output_fp8.shape, "Outputs should have same shape"

        correlation = torch.corrcoef(
            torch.stack([output_bf16.flatten(), output_fp8.flatten()])
        )[0, 1]

        assert (
            correlation > 0.5
        ), f"Outputs should be reasonably correlated, got {correlation}"

    def test_mixed_precision_components(self, setup_torch, test_device):
        args = get_small_config()
        args.dtype = "bf16"

        model = create_test_model(args, test_device)

        embedding_dtype = next(model.embed.parameters()).dtype
        assert (
            embedding_dtype == torch.bfloat16
        ), f"Embedding should be bfloat16, got {embedding_dtype}"

        head_dtype = next(model.head.parameters()).dtype
        assert head_dtype in [
            torch.bfloat16,
            torch.float32,
        ], f"Head should be bfloat16 or float32, got {head_dtype}"

    @pytest.mark.gpu
    def test_quantization_memory_usage(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Memory testing requires CUDA")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        args_bf16 = get_small_config()
        args_bf16.dtype = "bf16"

        model_bf16 = create_test_model(args_bf16, test_device)
        bf16_memory = torch.cuda.max_memory_allocated()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        args_fp8 = get_small_config()
        args_fp8.dtype = "fp8"

        model_fp8 = create_test_model(args_fp8, test_device)
        fp8_memory = torch.cuda.max_memory_allocated()

        memory_ratio = fp8_memory / bf16_memory
        assert (
            memory_ratio <= 1.0
        ), f"FP8 should use less or equal memory than BF16, ratio: {memory_ratio}"

    def test_dtype_parameter_consistency(self, setup_torch, test_device):
        args = get_small_config()
        args.dtype = "bf16"

        model = create_test_model(args, test_device)

        bf16_params = 0
        total_params = 0

        for param in model.parameters():
            total_params += 1
            if param.dtype == torch.bfloat16:
                bf16_params += 1

        assert total_params > 0, "Model should have parameters"
        bf16_ratio = bf16_params / total_params
        assert (
            bf16_ratio > 0.5
        ), f"Most parameters should be bfloat16, ratio: {bf16_ratio}"

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    def test_quantization_inference_stability(self, dtype, setup_torch, test_device):
        if dtype == "fp8" and not torch.cuda.is_available():
            pytest.skip("FP8 testing requires CUDA")

        args = get_small_config()
        args.dtype = dtype

        model = create_test_model(args, test_device)
        tokens = generate_test_tokens(
            batch_size=1, seq_len=32, vocab_size=args.vocab_size, device=test_device
        )

        outputs = []
        for _ in range(3):
            with torch.inference_mode():
                output = model(tokens)
            outputs.append(output)

        for i in range(1, len(outputs)):
            torch.testing.assert_close(
                outputs[0],
                outputs[i],
                rtol=1e-4,
                atol=1e-5,
                msg=f"Inference should be deterministic for {dtype}",
            )
