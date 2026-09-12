import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class AtomicConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_head: Optional[int] = None
    d_ffn: Optional[int] = None
    max_seq_len: int = 4096
    dropout: float = 0.0
    bias: bool = False
    rms_norm_eps: float = 1e-6
    decay_min: float = 0.01
    decay_max: float = 0.99
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.d_head is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
            self.d_head = self.d_model // self.n_heads
        else:
            if self.d_model != self.n_heads * self.d_head:
                raise ValueError(f"d_model ({self.d_model}) must equal n_heads * d_head ({self.n_heads * self.d_head})")

        if self.d_ffn is None:
            # SwiGLU standard ratio 8/3 * d_model aligned to multiple of 32
            hidden = int(2 * self.d_model * 4 / 3)
            self.d_ffn = ((hidden + 31) // 32) * 32

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AtomicConfig":
        return cls(**d)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "AtomicConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def nano(cls, vocab_size: int = 4096) -> "AtomicConfig":
        """Ultra-fast nano model for extreme CPU latency (~2.5M params)."""
        return cls(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_head=32,
            d_ffn=344,
            max_seq_len=2048,
        )

    @classmethod
    def micro(cls, vocab_size: int = 4096) -> "AtomicConfig":
        """Balanced micro model (~12M params)."""
        return cls(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_head=32,
            d_ffn=688,
            max_seq_len=4096,
        )

    @classmethod
    def reasoning(cls, vocab_size: int = 4096) -> "AtomicConfig":
        """High-density reasoning production model (~32M params)."""
        return cls(
            vocab_size=vocab_size,
            d_model=384,
            n_layers=8,
            n_heads=8,
            d_head=48,
            d_ffn=1024,
            max_seq_len=4096,
        )

    @classmethod
    def base(cls, vocab_size: int = 8192) -> "AtomicConfig":
        """Standard base capacity model (~125M params)."""
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_head=64,
            d_ffn=2048,
            max_seq_len=8192,
        )
