import os
import sys
import argparse
import warnings
from typing import Optional
import torch
import torch.nn as nn

from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM


class AtomicInferenceWrapper(nn.Module):
    """
    Inference wrapper that maps input_ids directly to next-token logits.
    Guarantees clean single-tensor input / output signatures for JIT/ONNX engines.
    """
    def __init__(self, model: AtomicForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence inference.
        
        Args:
            input_ids: LongTensor of shape (B, T)
        Returns:
            logits: FloatTensor of shape (B, T, vocab_size)
        """
        logits, _, _ = self.model(input_ids)
        return logits


def export_torchscript(
    model: AtomicForCausalLM,
    output_path: str,
    example_seq_len: int = 64
) -> str:
    """
    Export ATOMIC causal LM to a TorchScript traced artifact (.pt).
    
    Args:
        model: Trained AtomicForCausalLM instance.
        output_path: Path to save the scripted model.
        example_seq_len: Sequence length for tracing.
    Returns:
        output_path: Path where artifact is saved.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.eval()

    wrapper = AtomicInferenceWrapper(model)
    example_input = torch.randint(0, model.config.vocab_size, (1, example_seq_len), dtype=torch.long)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        traced_model = torch.jit.trace(wrapper, example_input)

    traced_model.save(output_path)
    print(f"[OK] Exported TorchScript model to: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")
    return output_path


def export_onnx(
    model: AtomicForCausalLM,
    output_path: str,
    example_seq_len: int = 64,
    opset_version: int = 17
) -> str:
    """
    Export ATOMIC causal LM to an ONNX artifact (.onnx) with dynamic batch and sequence axes.
    
    Args:
        model: Trained AtomicForCausalLM instance.
        output_path: Destination path for .onnx file.
        example_seq_len: Sequence length for sample trace.
        opset_version: ONNX operator set version (default 17).
    Returns:
        output_path: Path where artifact is saved.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("The 'onnx' library is required to export to ONNX format. Install with: pip install onnx")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.eval()

    wrapper = AtomicInferenceWrapper(model)
    example_input = torch.randint(0, model.config.vocab_size, (1, example_seq_len), dtype=torch.long)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            wrapper,
            example_input,
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False
        )

    # Verify exported ONNX model integrity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"[OK] Exported verified ONNX model to: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="ATOMIC Model Exporter (TorchScript / ONNX)")
    parser.add_argument("--model-dir", type=str, default="models/atomic-reasoning-prod", help="Directory containing model config and weights")
    parser.add_argument("--output-dir", type=str, default="exported", help="Directory to save exported artifacts")
    parser.add_argument("--format", type=str, choices=["torchscript", "onnx", "both"], default="both", help="Export target format")
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, "config.json")
    weights_path = os.path.join(args.model_dir, "best_model.pt")

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    config = AtomicConfig.from_json(config_path)
    model = AtomicForCausalLM(config)

    if os.path.exists(weights_path):
        weights = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(weights)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Notice: No weights file found at {weights_path}, exporting initialized architecture.")

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.format in ("torchscript", "both"):
        ts_path = os.path.join(args.output_dir, "atomic_model.pt")
        export_torchscript(model, ts_path)

    if args.format in ("onnx", "both"):
        onnx_path = os.path.join(args.output_dir, "atomic_model.onnx")
        export_onnx(model, onnx_path)


if __name__ == "__main__":
    main()
