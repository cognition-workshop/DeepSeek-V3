import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator

import pytest
import torch
import torch.distributed as dist

from inference.model import ModelArgs, Transformer


@pytest.fixture(scope="session")
def test_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def setup_torch():
    torch.set_default_dtype(torch.bfloat16)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    torch.manual_seed(42)
    yield
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture
def minimal_model_args() -> ModelArgs:
    return ModelArgs(
        max_batch_size=2,
        max_seq_len=128,
        dtype="bf16",
        vocab_size=1024,
        dim=256,
        inter_dim=512,
        moe_inter_dim=128,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=0,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
    )


@pytest.fixture
def fp8_model_args() -> ModelArgs:
    return ModelArgs(
        max_batch_size=2,
        max_seq_len=128,
        dtype="fp8",
        vocab_size=1024,
        dim=256,
        inter_dim=512,
        moe_inter_dim=128,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=0,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
    )


@pytest.fixture
def sample_tokens(test_device):
    return torch.randint(0, 1024, (2, 32), device=test_device)


@pytest.fixture
def temp_config_file(minimal_model_args) -> Generator[str, None, None]:
    config_dict = {
        "max_batch_size": minimal_model_args.max_batch_size,
        "max_seq_len": minimal_model_args.max_seq_len,
        "dtype": minimal_model_args.dtype,
        "vocab_size": minimal_model_args.vocab_size,
        "dim": minimal_model_args.dim,
        "inter_dim": minimal_model_args.inter_dim,
        "moe_inter_dim": minimal_model_args.moe_inter_dim,
        "n_layers": minimal_model_args.n_layers,
        "n_dense_layers": minimal_model_args.n_dense_layers,
        "n_heads": minimal_model_args.n_heads,
        "n_routed_experts": minimal_model_args.n_routed_experts,
        "n_shared_experts": minimal_model_args.n_shared_experts,
        "n_activated_experts": minimal_model_args.n_activated_experts,
        "q_lora_rank": minimal_model_args.q_lora_rank,
        "kv_lora_rank": minimal_model_args.kv_lora_rank,
        "qk_nope_head_dim": minimal_model_args.qk_nope_head_dim,
        "qk_rope_head_dim": minimal_model_args.qk_rope_head_dim,
        "v_head_dim": minimal_model_args.v_head_dim,
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        temp_path = f.name
    
    yield temp_path
    
    os.unlink(temp_path)


def pytest_configure(config):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="GPU not available"))
        
        if "distributed" in item.keywords:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                item.add_marker(pytest.mark.skip(reason="Distributed testing requires multiple GPUs"))
