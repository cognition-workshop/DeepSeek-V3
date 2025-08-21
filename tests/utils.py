import time
import psutil
import torch
from typing import Dict, Any, Tuple, Optional
from contextlib import contextmanager

from inference.model import Transformer, ModelArgs


def get_memory_usage() -> Dict[str, float]:
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
        memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
        memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3
    
    process = psutil.Process()
    memory_info['cpu_memory'] = process.memory_info().rss / 1024**3
    
    return memory_info


@contextmanager
def measure_time():
    start_time = time.time()
    yield lambda: time.time() - start_time
    

def create_test_model(args: ModelArgs, device: Optional[torch.device] = None) -> Transformer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.device(device):
        model = Transformer(args)
    
    def init_weights(module):
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) >= 2:
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    model.apply(init_weights)
    
    if args.dtype == "bf16":
        complex_buffers = {}
        for name, buffer in model.named_buffers():
            if buffer.dtype == torch.complex64:
                complex_buffers[name] = buffer.clone()
        
        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
        
        for name, buffer in complex_buffers.items():
            parts = name.split('.')
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], buffer)
            
    elif args.dtype == "fp16":
        model = model.to(torch.float16)
    elif args.dtype == "fp8":
        model = model.to(torch.bfloat16)
    
    model.eval()
    
    for param in model.parameters():
        param.requires_grad_(True)
    
    return model


def validate_model_output(output: torch.Tensor, expected_shape: Tuple[int, ...], vocab_size: int):
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert output.dtype in [torch.float32, torch.bfloat16], f"Unexpected output dtype: {output.dtype}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    assert output.size(-1) == vocab_size, f"Expected vocab size {vocab_size}, got {output.size(-1)}"


def check_model_parameters(model: Transformer, args: ModelArgs):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0, "Model has no parameters"
    assert trainable_params > 0, "Model has no trainable parameters"
    
    assert hasattr(model, 'embed'), "Model missing embedding layer"
    assert hasattr(model, 'layers'), "Model missing transformer layers"
    assert hasattr(model, 'norm'), "Model missing normalization layer"
    assert hasattr(model, 'head'), "Model missing output head"
    
    assert len(model.layers) == args.n_layers, f"Expected {args.n_layers} layers, got {len(model.layers)}"


def generate_test_tokens(batch_size: int, seq_len: int, vocab_size: int, device: torch.device, dtype: torch.dtype = torch.long) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=dtype)


def assert_tensor_properties(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                           expected_dtype: torch.dtype = None, check_finite: bool = True):
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if check_finite:
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"


def benchmark_inference(model: Transformer, tokens: torch.Tensor, num_runs: int = 5) -> Dict[str, float]:
    model.eval()
    
    warmup_runs = 2
    for _ in range(warmup_runs):
        with torch.inference_mode():
            _ = model(tokens)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.inference_mode():
            output = model(tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
    }
