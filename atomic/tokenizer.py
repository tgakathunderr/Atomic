import json
import re
import torch
from typing import List, Dict, Optional, Union

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
    Fast, robust tokenizer for reasoning language models.
    Supports special reasoning delimiters, full ASCII byte coverage (zero UNK tokens for ASCII),
    and fast token-level manipulation.
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

        # Compile regex pattern for tokenization
        # Special tokens are matched first as whole words
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

        # 3. Common English reasoning and logic subwords
        common_words = [
            "the", "of", "and", "to", "a", "in", "is", "that", "for", "it",
            "as", "was", "with", "be", "by", "on", "not", "he", "i", "this",
            "are", "or", "from", "at", "which", "but", "more", "an", "they",
            "one", "we", "if", "would", "all", "so", "has", "there", "their",
            "what", "when", "can", "said", "use", "do", "how", "each", "which",
            "then", "now", "find", "only", "first", "also", "after", "back",
            "Question", "Answer", "Explanation", "Problem", "Solution", "Step",
            "Let", "Since", "Therefore", "Because", "Thus", "Hence", "Given",
            "Suppose", "Assume", "True", "False", "Yes", "No", "equal", "greater",
            "less", "add", "subtract", "multiply", "divide", "result", "sum",
            "difference", "product", "quotient", "calculate", "logic", "deduce",
            "implies", "premise", "conclusion", "valid", "invalid", "rule",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "+", "-", "*", "/", "=", "<", ">", "<=", ">=", "!=", "==",
            "  ", "   ", "    ", "\n\n", "\t", "->", "=>", "::", "==", "!="
        ]
        for w in common_words:
            if w not in self.token_to_id:
                self.token_to_id[w] = idx
                idx += 1

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        """Encode text string into token IDs."""
        tokens: List[int] = []
        pos = 0
        while pos < len(text):
            # Check for special tokens first
            found_special = False
            for spec in SPECIAL_TOKENS:
                if text.startswith(spec, pos):
                    tokens.append(self.token_to_id[spec])
                    pos += len(spec)
                    found_special = True
                    break
            if found_special:
                continue

            # Greedy match longest token from vocab
            # Try matching longest known token up to 16 chars
            matched = False
            for length in range(min(16, len(text) - pos), 1, -1):
                sub = text[pos:pos + length]
                if sub in self.token_to_id:
                    tokens.append(self.token_to_id[sub])
                    pos += length
                    matched = True
                    break

            if not matched:
                # Fallback to single character/byte
                ch = text[pos]
                tokens.append(self.token_to_id.get(ch, self.unk_id))
                pos += 1

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs into string."""
        chars = []
        for t in tokens:
            if t not in self.id_to_token:
                continue
            tok_str = self.id_to_token[t]
            if skip_special_tokens and tok_str in SPECIAL_TOKENS:
                continue
            chars.append(tok_str)
        return "".join(chars)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        pad_to_max: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Batch encode with padding and attention mask."""
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
        """Save vocabulary to JSON."""
        data = {
            "vocab": self.token_to_id,
            "special_tokens": SPECIAL_TOKENS
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str) -> "AtomicTokenizer":
        """Load vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])
