import torch
import torch.nn as nn
from unittest.mock import patch
from inference.model import ModelArgs


def create_test_tensor(shape, dtype=None, device=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    return torch.randn(shape, dtype=dtype, device=device)


def assert_tensor_properties(tensor, expected_shape, expected_dtype=None):
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"


def mock_world_size_rank(world_size=1, rank=0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch('inference.model.world_size', world_size), \
                 patch('inference.model.rank', rank):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class MockLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


def create_minimal_config(**overrides):
    defaults = {
        'dim': 512,
        'moe_inter_dim': 256,
        'n_routed_experts': 8,
        'n_shared_experts': 1,
        'n_activated_experts': 2,
        'n_expert_groups': 1,
        'n_limited_groups': 1,
        'route_scale': 1.0,
        'score_func': 'softmax',
        'dtype': 'bf16'
    }
    defaults.update(overrides)
    return ModelArgs(**defaults)


def initialize_gate_weights(gate):
    """Initialize gate weights to avoid NaN values during testing."""
    with torch.no_grad():
        torch.nn.init.normal_(gate.weight, mean=0.0, std=0.1)
        if gate.bias is not None:
            torch.nn.init.zeros_(gate.bias)


def initialize_module_weights(module):
    """Initialize all module weights to avoid NaN values during testing."""
    with torch.no_grad():
        for param in module.parameters():
            if param.dim() >= 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.1)
            else:
                torch.nn.init.zeros_(param)


def check_gradient_flow(model, input_tensor):
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    
    params_with_grad = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"
    
    grad_ratio = params_with_grad / total_params if total_params > 0 else 0
    assert grad_ratio >= 0.5, f"Only {params_with_grad}/{total_params} parameters have gradients (ratio: {grad_ratio:.2f})"


def verify_routing_properties(weights, indices, n_activated_experts, n_routed_experts):
    batch_size = weights.shape[0]
    
    assert weights.shape == (batch_size, n_activated_experts), \
        f"Expected weights shape ({batch_size}, {n_activated_experts}), got {weights.shape}"
    
    assert indices.shape == (batch_size, n_activated_experts), \
        f"Expected indices shape ({batch_size}, {n_activated_experts}), got {indices.shape}"
    
    assert torch.all(indices >= 0), "All indices should be non-negative"
    assert torch.all(indices < n_routed_experts), f"All indices should be less than {n_routed_experts}"
    
    assert torch.all(weights >= -1e-6), "All weights should be non-negative (within tolerance)"
    
    for i in range(batch_size):
        unique_indices = torch.unique(indices[i])
        assert len(unique_indices) == n_activated_experts, \
            f"Expected {n_activated_experts} unique indices per sample, got {len(unique_indices)}"
