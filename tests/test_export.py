import os
import pytest
import torch
from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from atomic.export import export_torchscript, export_onnx

import tempfile
import shutil

@pytest.fixture
def temp_export_dir():
    d = tempfile.mkdtemp(prefix="test_export_", dir=".")
    yield d
    shutil.rmtree(d, ignore_errors=True)

def test_export_torchscript(temp_export_dir):
    config = AtomicConfig.nano(vocab_size=100)
    model = AtomicForCausalLM(config)
    model.eval()

    output_path = os.path.join(temp_export_dir, "model_scripted.pt")
    scripted_path = export_torchscript(model, output_path, example_seq_len=16)
    
    assert os.path.exists(scripted_path)
    assert os.path.getsize(scripted_path) > 0

    # Load and verify inference
    loaded = torch.jit.load(scripted_path)
    dummy_input = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        out_orig, _, _ = model(dummy_input)
        out_scripted = loaded(dummy_input)
        assert torch.allclose(out_orig, out_scripted, atol=1e-4)

def test_export_onnx(temp_export_dir):
    config = AtomicConfig.nano(vocab_size=100)
    model = AtomicForCausalLM(config)
    model.eval()

    output_path = os.path.join(temp_export_dir, "model.onnx")
    onnx_path = export_onnx(model, output_path, example_seq_len=16)

    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
