import pytest
from atomic.tokenizer import AtomicTokenizer
from data.reasoning_dataset import (
    generate_arithmetic_reasoning,
    generate_logic_reasoning,
    generate_algorithmic_reasoning,
    build_reasoning_corpus,
    ReasoningDataset,
    collate_reasoning_fn
)

def test_generators_format():
    for gen in [generate_arithmetic_reasoning, generate_logic_reasoning, generate_algorithmic_reasoning]:
        sample = gen()
        assert "Question:" in sample
        assert "<think>" in sample
        assert "</think>" in sample
        assert "<answer>" in sample
        assert "</answer>" in sample

def test_dataset_and_collate():
    tok = AtomicTokenizer()
    samples = build_reasoning_corpus(num_samples=10)
    dataset = ReasoningDataset(samples, tok, max_length=128)
    assert len(dataset) == 10

    inp, tgt = dataset[0]
    assert inp.shape == tgt.shape
    assert (inp[1:] == tgt[:-1]).all()

    # Test collate
    batch = [dataset[0], dataset[1]]
    padded_inps, padded_tgts = collate_reasoning_fn(batch, tok.pad_id)
    assert padded_inps.dim() == 2
    assert padded_tgts.dim() == 2
    assert padded_inps.shape == padded_tgts.shape
