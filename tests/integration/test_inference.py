import pytest
import torch

from inference.model import Transformer
from tests.fixtures.test_configs import get_small_config, get_tiny_config
from tests.utils import (benchmark_inference, create_test_model,
                         generate_test_tokens, measure_time,
                         validate_model_output)


class TestInference:

    def test_basic_forward_pass(self, minimal_model_args, setup_torch, test_device):
        model = create_test_model(minimal_model_args, test_device)
        tokens = generate_test_tokens(
            batch_size=1,
            seq_len=16,
            vocab_size=minimal_model_args.vocab_size,
            device=test_device,
        )

        with torch.inference_mode():
            output = model(tokens)

        expected_shape = (1, minimal_model_args.vocab_size)
        validate_model_output(output, expected_shape, minimal_model_args.vocab_size)

    def test_batch_inference(self, minimal_model_args, setup_torch, test_device):
        model = create_test_model(minimal_model_args, test_device)
        batch_size = minimal_model_args.max_batch_size
        tokens = generate_test_tokens(
            batch_size=batch_size,
            seq_len=32,
            vocab_size=minimal_model_args.vocab_size,
            device=test_device,
        )

        with torch.inference_mode():
            output = model(tokens)

        expected_shape = (batch_size, minimal_model_args.vocab_size)
        validate_model_output(output, expected_shape, minimal_model_args.vocab_size)

    def test_variable_sequence_lengths(
        self, minimal_model_args, setup_torch, test_device
    ):
        model = create_test_model(minimal_model_args, test_device)

        seq_lengths = [8, 16, 32, 64]
        for seq_len in seq_lengths:
            if seq_len <= minimal_model_args.max_seq_len:
                tokens = generate_test_tokens(
                    batch_size=1,
                    seq_len=seq_len,
                    vocab_size=minimal_model_args.vocab_size,
                    device=test_device,
                )

                with torch.inference_mode():
                    output = model(tokens)

                expected_shape = (1, minimal_model_args.vocab_size)
                validate_model_output(
                    output, expected_shape, minimal_model_args.vocab_size
                )

    def test_inference_with_start_pos(
        self, minimal_model_args, setup_torch, test_device
    ):
        model = create_test_model(minimal_model_args, test_device)
        tokens = generate_test_tokens(
            batch_size=1,
            seq_len=16,
            vocab_size=minimal_model_args.vocab_size,
            device=test_device,
        )

        start_positions = [0, 8, 16]
        for start_pos in start_positions:
            if start_pos < minimal_model_args.max_seq_len:
                with torch.inference_mode():
                    output = model(tokens, start_pos=start_pos)

                expected_shape = (1, minimal_model_args.vocab_size)
                validate_model_output(
                    output, expected_shape, minimal_model_args.vocab_size
                )

    def test_inference_deterministic(
        self, minimal_model_args, setup_torch, test_device
    ):
        torch.manual_seed(42)
        model1 = create_test_model(minimal_model_args, test_device)

        torch.manual_seed(42)
        model2 = create_test_model(minimal_model_args, test_device)

        tokens = generate_test_tokens(
            batch_size=1,
            seq_len=16,
            vocab_size=minimal_model_args.vocab_size,
            device=test_device,
        )

        torch.manual_seed(123)
        with torch.inference_mode():
            output1 = model1(tokens)

        torch.manual_seed(123)
        with torch.inference_mode():
            output2 = model2(tokens)

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("config_func", [get_tiny_config, get_small_config])
    def test_inference_different_configs(self, config_func, setup_torch, test_device):
        args = config_func()
        model = create_test_model(args, test_device)

        tokens = generate_test_tokens(
            batch_size=1,
            seq_len=min(32, args.max_seq_len),
            vocab_size=args.vocab_size,
            device=test_device,
        )

        with torch.inference_mode():
            output = model(tokens)

        expected_shape = (1, args.vocab_size)
        validate_model_output(output, expected_shape, args.vocab_size)

    def test_inference_max_sequence_length(
        self, minimal_model_args, setup_torch, test_device
    ):
        model = create_test_model(minimal_model_args, test_device)
        max_seq_len = minimal_model_args.max_seq_len

        tokens = generate_test_tokens(
            batch_size=1,
            seq_len=max_seq_len,
            vocab_size=minimal_model_args.vocab_size,
            device=test_device,
        )

        with torch.inference_mode():
            output = model(tokens)

        expected_shape = (1, minimal_model_args.vocab_size)
        validate_model_output(output, expected_shape, minimal_model_args.vocab_size)

    def test_inference_output_distribution(
        self, minimal_model_args, setup_torch, test_device
    ):
        model = create_test_model(minimal_model_args, test_device)
        tokens = generate_test_tokens(
            batch_size=1,
            seq_len=16,
            vocab_size=minimal_model_args.vocab_size,
            device=test_device,
        )

        with torch.inference_mode():
            output = model(tokens)

        logits = output[0]
        probs = torch.softmax(logits, dim=-1)

        assert torch.allclose(
            probs.sum(), torch.tensor(1.0, device=test_device), atol=1e-6
        )
        assert (probs >= 0).all(), "Probabilities should be non-negative"
        assert (probs <= 1).all(), "Probabilities should not exceed 1"

    @pytest.mark.slow
    def test_inference_performance(self, minimal_model_args, setup_torch, test_device):
        model = create_test_model(minimal_model_args, test_device)
        tokens = generate_test_tokens(
            batch_size=minimal_model_args.max_batch_size,
            seq_len=64,
            vocab_size=minimal_model_args.vocab_size,
            device=test_device,
        )

        benchmark_results = benchmark_inference(model, tokens, num_runs=3)

        assert (
            benchmark_results["mean_time"] > 0
        ), "Inference should take measurable time"
        assert (
            benchmark_results["mean_time"] < 10.0
        ), "Inference should complete within reasonable time"
        assert (
            benchmark_results["std_time"] >= 0
        ), "Standard deviation should be non-negative"

    def test_model_eval_mode(self, minimal_model_args, setup_torch, test_device):
        model = create_test_model(minimal_model_args, test_device)

        model.train()
        assert model.training, "Model should be in training mode"

        model.eval()
        assert not model.training, "Model should be in evaluation mode"

        tokens = generate_test_tokens(
            batch_size=1,
            seq_len=16,
            vocab_size=minimal_model_args.vocab_size,
            device=test_device,
        )

        with torch.inference_mode():
            output = model(tokens)

        expected_shape = (1, minimal_model_args.vocab_size)
        validate_model_output(output, expected_shape, minimal_model_args.vocab_size)
