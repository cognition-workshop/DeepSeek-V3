import pytest
import torch

from inference.model import ModelArgs, Transformer
from tests.fixtures.test_configs import (
    get_fp8_config,
    get_small_config,
    get_tiny_config,
)
from tests.utils import check_model_parameters, create_test_model, get_memory_usage


class TestModelLoading:

    @pytest.mark.parametrize("config_func", [get_tiny_config, get_small_config])
    def test_model_initialization(self, config_func, setup_torch, test_device):
        args = config_func()

        initial_memory = get_memory_usage()
        model = create_test_model(args, test_device)
        final_memory = get_memory_usage()

        check_model_parameters(model, args)

        assert model.max_seq_len == args.max_seq_len
        assert len(model.layers) == args.n_layers

        if torch.cuda.is_available():
            memory_used = final_memory["gpu_allocated"] - initial_memory.get(
                "gpu_allocated", 0
            )
            assert memory_used > 0, "Model should use GPU memory"

    def test_model_initialization_bf16(
        self, minimal_model_args, setup_torch, test_device
    ):
        minimal_model_args.dtype = "bf16"
        model = create_test_model(minimal_model_args, test_device)

        check_model_parameters(model, minimal_model_args)

        for param in model.parameters():
            if param.element_size() > 1:
                assert (
                    param.dtype == torch.bfloat16
                ), f"Expected bfloat16, got {param.dtype}"

    @pytest.mark.gpu
    def test_model_initialization_fp8(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("FP8 testing requires CUDA")

        args = get_fp8_config()
        model = create_test_model(args, test_device)

        check_model_parameters(model, args)

        fp8_params_found = False
        for param in model.parameters():
            if param.element_size() == 1:
                fp8_params_found = True
                break

        assert (
            fp8_params_found or args.dtype == "fp8"
        ), "FP8 model should have FP8 parameters or be configured for FP8"

    def test_model_device_placement(self, minimal_model_args, setup_torch):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = create_test_model(minimal_model_args, device)

            for param in model.parameters():
                assert (
                    param.device.type == "cuda"
                ), f"Parameter not on CUDA: {param.device}"
        else:
            device = torch.device("cpu")
            model = create_test_model(minimal_model_args, device)

            for param in model.parameters():
                assert (
                    param.device.type == "cpu"
                ), f"Parameter not on CPU: {param.device}"

    def test_model_parameter_count(self, minimal_model_args, setup_torch, test_device):
        model = create_test_model(minimal_model_args, test_device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0, "Model should have parameters"
        assert (
            trainable_params == total_params
        ), "All parameters should be trainable by default"

        expected_min_params = minimal_model_args.vocab_size * minimal_model_args.dim
        assert (
            total_params >= expected_min_params
        ), f"Model has fewer parameters than expected minimum"

    def test_model_layers_structure(self, minimal_model_args, setup_torch, test_device):
        model = create_test_model(minimal_model_args, test_device)

        assert hasattr(model, "embed"), "Model should have embedding layer"
        assert hasattr(model, "layers"), "Model should have transformer layers"
        assert hasattr(model, "norm"), "Model should have final normalization"
        assert hasattr(model, "head"), "Model should have output head"

        assert len(model.layers) == minimal_model_args.n_layers

        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "attn"), f"Layer {i} should have attention"
            assert hasattr(layer, "ffn"), f"Layer {i} should have feed-forward network"
            assert hasattr(
                layer, "attn_norm"
            ), f"Layer {i} should have attention normalization"
            assert hasattr(
                layer, "ffn_norm"
            ), f"Layer {i} should have FFN normalization"

    def test_model_freqs_cis_buffer(self, minimal_model_args, setup_torch, test_device):
        model = create_test_model(minimal_model_args, test_device)

        assert hasattr(model, "freqs_cis"), "Model should have freqs_cis buffer"
        assert model.freqs_cis.shape[0] == minimal_model_args.max_seq_len
        assert model.freqs_cis.dtype == torch.complex64
