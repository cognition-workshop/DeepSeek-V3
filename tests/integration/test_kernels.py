import pytest
import torch

from inference.kernel import act_quant, weight_dequant, fp8_gemm


class TestKernels:

    @pytest.mark.gpu
    def test_act_quant_basic(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        x = torch.randn(32, 128, device=test_device, dtype=torch.bfloat16)

        y, s = act_quant(x, block_size=128)

        assert y.dtype == torch.float8_e4m3fn, f"Expected float8_e4m3fn, got {y.dtype}"
        assert s.dtype == torch.float32, f"Expected float32 for scale, got {s.dtype}"
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
        assert s.shape == (
            *x.shape[:-1],
            x.shape[-1] // 128,
        ), f"Scale shape mismatch: {s.shape}"

    @pytest.mark.gpu
    def test_act_quant_different_block_sizes(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        x = torch.randn(16, 256, device=test_device, dtype=torch.bfloat16)

        for block_size in [64, 128, 256]:
            if x.size(-1) % block_size == 0:
                y, s = act_quant(x, block_size=block_size)

                assert y.dtype == torch.float8_e4m3fn
                assert s.dtype == torch.float32
                assert y.shape == x.shape
                expected_scale_shape = (*x.shape[:-1], x.shape[-1] // block_size)
                assert s.shape == expected_scale_shape

    @pytest.mark.gpu
    def test_weight_dequant_basic(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        M, N = 64, 128
        block_size = 128

        x = torch.randint(0, 255, (M, N), device=test_device, dtype=torch.uint8)
        s = torch.randn(
            M,
            (N + block_size - 1) // block_size,
            device=test_device,
            dtype=torch.float32,
        )

        y = weight_dequant(x, s, block_size=block_size)

        assert y.shape == (M, N), f"Shape mismatch: {y.shape} vs {(M, N)}"
        assert y.dtype == torch.get_default_dtype(), f"Unexpected dtype: {y.dtype}"

    @pytest.mark.gpu
    def test_fp8_gemm_basic(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        M, K, N = 32, 64, 48
        block_size = 128

        a = torch.randn(M, K, device=test_device, dtype=torch.float8_e4m3fn)
        b = torch.randn(N, K, device=test_device, dtype=torch.float8_e4m3fn)

        a_s = torch.randn(
            M,
            (K + block_size - 1) // block_size,
            device=test_device,
            dtype=torch.float32,
        )
        b_s = torch.randn(
            N,
            (K + block_size - 1) // block_size,
            device=test_device,
            dtype=torch.float32,
        )

        c = fp8_gemm(a, a_s, b, b_s)

        assert c.shape == (M, N), f"Shape mismatch: {c.shape} vs {(M, N)}"
        assert c.dtype == torch.get_default_dtype(), f"Unexpected dtype: {c.dtype}"

    @pytest.mark.gpu
    def test_act_quant_contiguous_requirement(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        x = torch.randn(32, 128, device=test_device, dtype=torch.bfloat16)
        x_non_contiguous = x.transpose(0, 1).transpose(0, 1)

        assert not x_non_contiguous.is_contiguous()

        with pytest.raises(AssertionError, match="Input tensor must be contiguous"):
            act_quant(x_non_contiguous, block_size=128)

    @pytest.mark.gpu
    def test_weight_dequant_contiguous_requirement(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        M, N = 64, 128
        block_size = 128

        x = torch.randint(0, 255, (M, N), device=test_device, dtype=torch.uint8)
        s = torch.randn(
            M,
            (N + block_size - 1) // block_size,
            device=test_device,
            dtype=torch.float32,
        )

        x_non_contiguous = x.transpose(0, 1).transpose(0, 1)
        assert not x_non_contiguous.is_contiguous()

        with pytest.raises(AssertionError, match="Input tensors must be contiguous"):
            weight_dequant(x_non_contiguous, s, block_size=block_size)

    @pytest.mark.gpu
    def test_fp8_gemm_contiguous_requirement(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        M, K, N = 32, 64, 48
        block_size = 128

        a = torch.randn(M, K, device=test_device, dtype=torch.float8_e4m3fn)
        b = torch.randn(N, K, device=test_device, dtype=torch.float8_e4m3fn)

        a_s = torch.randn(
            M,
            (K + block_size - 1) // block_size,
            device=test_device,
            dtype=torch.float32,
        )
        b_s = torch.randn(
            N,
            (K + block_size - 1) // block_size,
            device=test_device,
            dtype=torch.float32,
        )

        a_non_contiguous = a.transpose(0, 1).transpose(0, 1)
        assert not a_non_contiguous.is_contiguous()

        with pytest.raises(AssertionError, match="Input tensors must be contiguous"):
            fp8_gemm(a_non_contiguous, a_s, b, b_s)

    @pytest.mark.gpu
    def test_act_quant_block_size_divisibility(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        x = torch.randn(32, 127, device=test_device, dtype=torch.bfloat16)

        with pytest.raises(
            AssertionError, match="Last dimension size must be divisible by block_size"
        ):
            act_quant(x, block_size=128)

    @pytest.mark.gpu
    def test_kernels_numerical_stability(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("Kernel testing requires CUDA")

        x = torch.randn(16, 128, device=test_device, dtype=torch.bfloat16)

        y, s = act_quant(x, block_size=128)

        assert not torch.isnan(y).any(), "Quantized output contains NaN"
        assert not torch.isnan(s).any(), "Scale output contains NaN"
        assert not torch.isinf(s).any(), "Scale output contains Inf"
        assert (s > 0).all(), "All scales should be positive"
