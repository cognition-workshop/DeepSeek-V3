import pytest
import torch

from inference.model import ModelArgs, Transformer
from tests.fixtures.test_configs import get_small_config
from tests.utils import (benchmark_inference, create_test_model,
                         generate_test_tokens, validate_model_output)


class TestSampleFunctionality:

    def test_complete_inference_pipeline(self, setup_torch, test_device):
        args = get_small_config()
        model = create_test_model(args, test_device)

        batch_size = 2
        seq_len = 32
        tokens = generate_test_tokens(batch_size, seq_len, args.vocab_size, test_device)

        model.eval()
        with torch.inference_mode():
            logits = model(tokens)

        expected_shape = (batch_size, args.vocab_size)
        validate_model_output(logits, expected_shape, args.vocab_size)

        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(
            probs.sum(dim=-1), torch.ones(batch_size, device=test_device), atol=1e-6
        )

        predicted_tokens = logits.argmax(dim=-1)
        assert predicted_tokens.shape == (batch_size,)
        assert (predicted_tokens >= 0).all()
        assert (predicted_tokens < args.vocab_size).all()

    def test_text_generation_simulation(self, setup_torch, test_device):
        args = get_small_config()
        model = create_test_model(args, test_device)

        initial_tokens = generate_test_tokens(1, 16, args.vocab_size, test_device)
        generated_sequence = initial_tokens.clone()

        max_new_tokens = 8
        for step in range(max_new_tokens):
            current_seq_len = generated_sequence.size(1)
            if current_seq_len >= args.max_seq_len:
                break

            with torch.inference_mode():
                logits = model(generated_sequence)

            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

        assert generated_sequence.size(1) == initial_tokens.size(1) + max_new_tokens
        assert (generated_sequence >= 0).all()
        assert (generated_sequence < args.vocab_size).all()

    def test_model_caching_behavior(self, setup_torch, test_device):
        args = get_small_config()
        model = create_test_model(args, test_device)

        tokens = generate_test_tokens(1, 32, args.vocab_size, test_device)

        with torch.inference_mode():
            output1 = model(tokens, start_pos=0)

        with torch.inference_mode():
            output2 = model(tokens, start_pos=0)

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)

    def test_attention_mask_functionality(self, setup_torch, test_device):
        args = get_small_config()
        model = create_test_model(args, test_device)

        seq_len = 16
        tokens = generate_test_tokens(1, seq_len, args.vocab_size, test_device)

        with torch.inference_mode():
            output_full = model(tokens)

        shorter_tokens = tokens[:, : seq_len // 2]
        with torch.inference_mode():
            output_short = model(shorter_tokens)

        assert output_full.shape[1] == args.vocab_size
        assert output_short.shape[1] == args.vocab_size
        assert not torch.allclose(output_full, output_short, atol=1e-3)

    def test_model_parameter_updates(self, setup_torch, test_device):
        args = get_small_config()
        model = create_test_model(args, test_device)

        for param in model.parameters():
            param.requires_grad_(True)

        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.clone()

        tokens = generate_test_tokens(1, 16, args.vocab_size, test_device)

        model.train()
        output = model(tokens)
        loss = output.sum()
        loss.backward()

        gradients_exist = False
        for param in model.parameters():
            if param.grad is not None:
                gradients_exist = True
                break

        assert gradients_exist, "Model should have gradients after backward pass"

        non_zero_gradients = 0
        total_gradients = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_gradients += 1
                if not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                    non_zero_gradients += 1

        assert (
            non_zero_gradients > 0
        ), "At least some parameters should have non-zero gradients"
        assert (
            non_zero_gradients >= total_gradients * 0.3
        ), f"Expected at least 30% of parameters to have non-zero gradients, got {non_zero_gradients}/{total_gradients}"

    def test_model_memory_management(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Memory testing requires CUDA")

        args = get_small_config()

        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        model = create_test_model(args, test_device)
        after_model_memory = torch.cuda.memory_allocated()

        tokens = generate_test_tokens(
            args.max_batch_size, 64, args.vocab_size, test_device
        )

        with torch.inference_mode():
            output = model(tokens)

        after_inference_memory = torch.cuda.memory_allocated()

        del model
        del output
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        model_memory = after_model_memory - initial_memory
        inference_memory = after_inference_memory - after_model_memory

        assert model_memory > 0, "Model should use memory"
        assert inference_memory >= 0, "Inference should use some memory"
        assert (
            final_memory <= initial_memory + 1024 * 1024
        ), "Memory should be mostly freed"

    def test_model_performance_characteristics(self, setup_torch, test_device):
        args = get_small_config()
        model = create_test_model(args, test_device)

        small_tokens = generate_test_tokens(1, 16, args.vocab_size, test_device)
        large_tokens = generate_test_tokens(
            args.max_batch_size, 64, args.vocab_size, test_device
        )

        small_benchmark = benchmark_inference(model, small_tokens, num_runs=3)
        large_benchmark = benchmark_inference(model, large_tokens, num_runs=3)

        assert small_benchmark["mean_time"] > 0
        assert large_benchmark["mean_time"] > 0
        assert large_benchmark["mean_time"] >= small_benchmark["mean_time"]

    def test_model_output_consistency_across_runs(self, setup_torch, test_device):
        args = get_small_config()

        outputs = []
        for run in range(3):
            torch.manual_seed(42)
            model = create_test_model(args, test_device)

            torch.manual_seed(123)
            tokens = generate_test_tokens(1, 16, args.vocab_size, test_device)

            with torch.inference_mode():
                output = model(tokens)
            outputs.append(output)

        for i in range(1, len(outputs)):
            torch.testing.assert_close(
                outputs[0],
                outputs[i],
                rtol=1e-5,
                atol=1e-6,
                msg=f"Output should be consistent across runs",
            )

    def test_model_robustness_edge_cases(self, setup_torch, test_device):
        args = get_small_config()
        model = create_test_model(args, test_device)

        single_token = generate_test_tokens(1, 1, args.vocab_size, test_device)
        with torch.inference_mode():
            output_single = model(single_token)
        validate_model_output(output_single, (1, args.vocab_size), args.vocab_size)

        max_tokens = generate_test_tokens(
            1, args.max_seq_len, args.vocab_size, test_device
        )
        with torch.inference_mode():
            output_max = model(max_tokens)
        validate_model_output(output_max, (1, args.vocab_size), args.vocab_size)

        boundary_tokens = torch.zeros(1, 16, device=test_device, dtype=torch.long)
        with torch.inference_mode():
            output_boundary = model(boundary_tokens)
        validate_model_output(output_boundary, (1, args.vocab_size), args.vocab_size)
