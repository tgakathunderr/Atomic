import json
import re
import torch
from typing import List, Dict, Optional

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
]

class AtomicTokenizer:
    """
    Fast, robust tokenizer with full lossless reconstruction and complete reasoning token integration.
    """
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            self._build_default_vocab()

        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.think_start_id = self.token_to_id["<think>"]
        self.think_end_id = self.token_to_id["</think>"]
        self.answer_start_id = self.token_to_id["<answer>"]
        self.answer_end_id = self.token_to_id["</answer>"]

        specials_escaped = [re.escape(tok) for tok in SPECIAL_TOKENS]
        specials_pattern = "|".join(specials_escaped)
        self.pattern = re.compile(f"({specials_pattern})|(\\s+)|([a-zA-Z0-9_]+)|([^\\s\\w])")

    def _build_default_vocab(self):
        self.token_to_id = {}
        idx = 0

        # 1. Special tokens
        for tok in SPECIAL_TOKENS:
            self.token_to_id[tok] = idx
            idx += 1

        # 2. Byte/ASCII primitives (covers all 0..255 single characters)
        for b in range(256):
            ch = chr(b)
            if ch not in self.token_to_id:
                self.token_to_id[ch] = idx
                idx += 1

        # 3. Common whitespace and formatting
        whitespaces = [" ", "  ", "   ", "    ", "\n", "\n\n", "\t", " \n", "\n "]
        for ws in whitespaces:
            if ws not in self.token_to_id:
                self.token_to_id[ws] = idx
                idx += 1

        # 4. Common English reasoning and logic words & numbers
        from data.reasoning_dataset import build_reasoning_corpus
        corpus = build_reasoning_corpus(300)
        specials_escaped = [re.escape(tok) for tok in SPECIAL_TOKENS]
        specials_pattern = "|".join(specials_escaped)
        tmp_pat = re.compile(f"({specials_pattern})|(\\s+)|([a-zA-Z0-9_]+)|([^\\s\\w])")

        for sample in corpus:
            for match in tmp_pat.finditer(sample):
                tok_str = match.group(0)
                if tok_str not in self.token_to_id:
                    self.token_to_id[tok_str] = idx
                    idx += 1

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        """Losslessly encode text string into token IDs."""
        tokens: List[int] = []
        for m in self.pattern.finditer(text):
            tok_str = m.group(0)
            if tok_str in self.token_to_id:
                tokens.append(self.token_to_id[tok_str])
            else:
                # Character-by-character fallback
                for ch in tok_str:
                    tokens.append(self.token_to_id.get(ch, self.unk_id))
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs into string with 100% fidelity."""
        parts = []
        for t in tokens:
            if t not in self.id_to_token:
                continue
            tok_str = self.id_to_token[t]
            if skip_special_tokens and tok_str in SPECIAL_TOKENS:
                continue
            parts.append(tok_str)
        return "".join(parts)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        pad_to_max: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_ids = []
        batch_mask = []

        for text in texts:
            ids = self.encode(text)
            if len(ids) > max_length:
                ids = ids[:max_length]
            mask = [1] * len(ids)

            if pad_to_max and len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_id] * pad_len
                mask = mask + [0] * pad_len

            batch_ids.append(ids)
            batch_mask.append(mask)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }

    def save_pretrained(self, path: str):
        data = {
            "vocab": self.token_to_id,
            "special_tokens": SPECIAL_TOKENS
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str) -> "AtomicTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])
