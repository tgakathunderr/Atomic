"""
ATOMIC: Adaptive Token Operator with Memory-efficient Inference on CPU
Ultra-fast, constant-memory, transformer-competitive language modeling.
"""

from atomic.config import AtomicConfig
from atomic.model import AtomicModel, AtomicForCausalLM
from atomic.tokenizer import AtomicTokenizer
from atomic.generate import generate, generate_stream, GenerationConfig

__version__ = "0.1.0"
__all__ = [
    "AtomicConfig",
    "AtomicModel",
    "AtomicForCausalLM",
    "AtomicTokenizer",
    "generate",
    "generate_stream",
    "GenerationConfig",
]
