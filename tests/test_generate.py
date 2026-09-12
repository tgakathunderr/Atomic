import pytest
import torch
from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from atomic.tokenizer import AtomicTokenizer
from atomic.generate import generate, generate_stream, GenerationConfig

def test_generation_greedy():
    tokenizer = AtomicTokenizer()
    config = AtomicConfig.nano(vocab_size=tokenizer.vocab_size)
    model = AtomicForCausalLM(config)
    model.eval()

    prompt = "Question: Test"
    
    gen_cfg = GenerationConfig(
        max_new_tokens=20,
        temperature=0.0,
        repetition_penalty=1.0
    )

    output = generate(model, tokenizer, prompt, gen_cfg)
    assert isinstance(output, str)
    assert output.startswith(prompt)
    assert len(output) > len(prompt)

def test_generation_streaming():
    tokenizer = AtomicTokenizer()
    config = AtomicConfig.nano(vocab_size=tokenizer.vocab_size)
    model = AtomicForCausalLM(config)
    model.eval()

    prompt = "Question: Stream"
    
    gen_cfg = GenerationConfig(
        max_new_tokens=15,
        temperature=0.0
    )

    streamed_tokens = list(generate_stream(model, tokenizer, prompt, gen_cfg))
    assert len(streamed_tokens) > 0
    full_text = prompt + "".join(streamed_tokens)
    
    # Compare with non-streaming
    direct_text = generate(model, tokenizer, prompt, gen_cfg)
    assert full_text == direct_text

def test_generation_stop_sequence():
    tokenizer = AtomicTokenizer()
    config = AtomicConfig.nano(vocab_size=tokenizer.vocab_size)
    model = AtomicForCausalLM(config)
    model.eval()

    prompt = "Question: Stop"
    
    gen_cfg = GenerationConfig(
        max_new_tokens=50,
        temperature=0.7,
        stop_strings=["</answer>"]
    )
    
    output = generate(model, tokenizer, prompt, gen_cfg)
    assert isinstance(output, str)
