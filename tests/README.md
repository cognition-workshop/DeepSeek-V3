# DeepSeek-V3 Testing Infrastructure

This directory contains comprehensive integration and unit tests for the DeepSeek-V3 model inference system.

## Overview

The testing infrastructure is designed to validate:
- Model loading and initialization
- Inference functionality across different configurations
- Quantization modes (FP8/BF16)
- Custom Triton kernels
- Distributed inference capabilities
- Performance and memory characteristics

## Directory Structure

```
tests/
├── README.md                          # This file
├── requirements.txt                   # Testing dependencies
├── conftest.py                        # Pytest configuration and fixtures
├── utils.py                          # Testing utilities and helpers
├── fixtures/
│   ├── __init__.py
│   └── test_configs.py               # Test model configurations
├── integration/
│   ├── __init__.py
│   ├── test_model_loading.py         # Model initialization tests
│   ├── test_inference.py             # Basic inference functionality
│   ├── test_quantization.py          # FP8/BF16 quantization tests
│   ├── test_kernels.py               # Custom Triton kernel tests
│   ├── test_distributed.py           # Distributed inference tests
│   └── test_sample_functionality.py  # Comprehensive integration tests
└── unit/
    └── __init__.py                    # Unit tests (future expansion)
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for full test suite)
- Triton 3.0+

### Install Testing Dependencies

```bash
# From the repository root
pip install -r tests/requirements.txt

# Or install the inference requirements first
pip install -r inference/requirements.txt
pip install -r tests/requirements.txt
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest tests/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/integration/test_inference.py

# Run with verbose output
pytest tests/ -v

# Run tests in parallel (if pytest-xdist is installed)
pytest tests/ -n auto
```

### Test Categories and Markers

Tests are organized with pytest markers:

- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.distributed`: Tests for distributed functionality
- `@pytest.mark.slow`: Long-running tests

```bash
# Run only GPU tests
pytest tests/ -m gpu

# Skip slow tests
pytest tests/ -m "not slow"

# Run only integration tests that don't require GPU
pytest tests/ -m "integration and not gpu"
```

## Test Configurations

The test suite uses minimal model configurations for fast execution:

### Tiny Configuration
- Vocabulary: 512 tokens
- Dimensions: 128
- Layers: 1
- Heads: 2
- Experts: 4

### Small Configuration  
- Vocabulary: 1024 tokens
- Dimensions: 256
- Layers: 2
- Heads: 4
- Experts: 8

### FP8 Configuration
- Same as small configuration but with FP8 quantization enabled

## Test Categories

### Model Loading Tests (`test_model_loading.py`)
- Model initialization with different configurations
- Parameter validation and device placement
- Memory usage verification
- Layer structure validation

### Inference Tests (`test_inference.py`)
- Basic forward pass functionality
- Batch inference
- Variable sequence lengths
- Deterministic behavior
- Performance benchmarking

### Quantization Tests (`test_quantization.py`)
- BF16 mode validation
- FP8 mode validation (GPU required)
- Quantization consistency
- Memory efficiency comparison

### Kernel Tests (`test_kernels.py`)
- Custom Triton kernel functionality
- Activation quantization (`act_quant`)
- Weight dequantization (`weight_dequant`)
- FP8 GEMM operations (`fp8_gemm`)
- Input validation and error handling

### Distributed Tests (`test_distributed.py`)
- Single-GPU distributed simulation
- Parallel embedding layers
- MoE expert routing
- Column/row parallel linear layers

### Sample Functionality Tests (`test_sample_functionality.py`)
- Complete inference pipeline
- Text generation simulation
- Model caching behavior
- Memory management
- Performance characteristics

## Writing New Tests

### Test Structure

Follow this pattern for new tests:

```python
import pytest
import torch
from tests.utils import create_test_model, validate_model_output

class TestNewFeature:
    
    def test_basic_functionality(self, setup_torch, test_device):
        # Test setup
        args = get_small_config()
        model = create_test_model(args, test_device)
        
        # Test execution
        result = model.some_method()
        
        # Assertions
        assert result is not None
        validate_model_output(result, expected_shape, vocab_size)
    
    @pytest.mark.gpu
    def test_gpu_specific_feature(self, setup_torch, test_device):
        if not torch.cuda.is_available():
            pytest.skip("GPU required for this test")
        # GPU-specific test logic
```

### Fixtures and Utilities

Use the provided fixtures and utilities:

- `setup_torch`: Configures PyTorch defaults
- `test_device`: Provides appropriate device (CUDA/CPU)
- `minimal_model_args`: Small model configuration
- `create_test_model()`: Creates model instances
- `validate_model_output()`: Validates inference outputs
- `benchmark_inference()`: Performance testing

### Best Practices

1. **Use appropriate markers**: Mark tests with `@pytest.mark.gpu`, `@pytest.mark.slow`, etc.
2. **Handle device availability**: Skip GPU tests when CUDA is unavailable
3. **Use minimal configurations**: Keep test models small for fast execution
4. **Validate outputs thoroughly**: Check shapes, dtypes, and numerical properties
5. **Clean up resources**: Use fixtures for proper setup/teardown
6. **Test edge cases**: Include boundary conditions and error cases

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or sequence length in test configurations
   # Run tests sequentially instead of in parallel
   pytest tests/ -n 1
   ```

2. **Triton Compilation Errors**
   ```bash
   # Ensure CUDA toolkit is properly installed
   # Check Triton version compatibility
   pip install triton==3.0.0
   ```

3. **Import Errors**
   ```bash
   # Ensure you're running from repository root
   cd /path/to/DeepSeek-V3
   python -m pytest tests/
   ```

4. **Slow Test Execution**
   ```bash
   # Skip slow tests
   pytest tests/ -m "not slow"
   
   # Use smaller test configurations
   # Run specific test files instead of full suite
   ```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate markers and documentation
3. Include both positive and negative test cases
4. Update this README if adding new test categories
5. Ensure tests pass in both CPU and GPU environments

## CI/CD Integration

The test suite is designed for automated testing:

- Tests are organized by execution time and resource requirements
- GPU tests are properly marked and can be skipped in CPU-only environments
- Timeout protection prevents hanging tests
- Comprehensive error reporting for debugging failures

For CI/CD pipelines, consider:

```bash
# Fast smoke tests (CPU only)
pytest tests/ -m "not gpu and not slow" --timeout=60

# Full test suite (with GPU)
pytest tests/ --timeout=300

# Performance regression tests
pytest tests/ -m slow --timeout=600
```
