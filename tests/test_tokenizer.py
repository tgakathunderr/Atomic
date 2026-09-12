import os
import pytest
from atomic.tokenizer import AtomicTokenizer

def test_special_tokens_and_ids():
    tok = AtomicTokenizer()
    assert tok.pad_id is not None
    assert tok.eos_id is not None
    assert tok.think_start_id is not None
    assert tok.think_end_id is not None
    assert tok.answer_start_id is not None
    assert tok.answer_end_id is not None

    # Encoding special token alone gives single token ID
    ids = tok.encode("<think>")
    assert len(ids) == 1
    assert ids[0] == tok.think_start_id

def test_encode_decode_roundtrip():
    tok = AtomicTokenizer()
    text = "Question: What is 15 + 27? <think> First add 10 to 27 = 37, then 5 = 42. </think> <answer> 42 </answer>"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text

def test_batch_encoding_and_padding():
    tok = AtomicTokenizer()
    texts = [
        "Short sentence.",
        "A much longer sentence with step-by-step reasoning <think> compute </think> <answer> 1 </answer>"
    ]
    batch = tok.batch_encode(texts, max_length=32, pad_to_max=True)
    assert batch["input_ids"].shape == (2, 32)
    assert batch["attention_mask"].shape == (2, 32)
    # Check padding on shorter sequence
    assert (batch["input_ids"][0] == tok.pad_id).sum() > 0

def test_save_and_load_tokenizer():
    tok = AtomicTokenizer()
    path = "tests/_tmp_test_tokenizer.json"
    try:
        tok.save_pretrained(path)
        loaded_tok = AtomicTokenizer.from_pretrained(path)
        text = "<think> logic test </think> <answer> True </answer>"
        assert loaded_tok.encode(text) == tok.encode(text)
    finally:
        if os.path.exists(path):
            os.remove(path)
