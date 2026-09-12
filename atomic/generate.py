import math
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Union
import torch
import torch.nn.functional as F

from atomic.model import AtomicForCausalLM
from atomic.tokenizer import AtomicTokenizer


@dataclass
class GenerationConfig:
    """Configuration parameters for text generation."""
    max_new_tokens: int = 150
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    stop_strings: List[str] = field(default_factory=lambda: ["</answer>", "<eos>"])
    stop_token_ids: List[int] = field(default_factory=list)
    device: str = "cpu"


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0
) -> torch.Tensor:
    """
    Sample next token index from logits.
    
    Supports greedy argmax (temperature == 0), temperature scaling,
    top-k truncation, and top-p (nucleus) filtering.
    """
    if logits.dim() == 2:
        logits = logits[0]  # (vocab_size,)

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def generate_stream(
    model: AtomicForCausalLM,
    tokenizer: AtomicTokenizer,
    prompt: Union[str, List[int]],
    config: Optional[GenerationConfig] = None
) -> Generator[str, None, None]:
    """
    Yields newly generated token strings one by one in real-time.
    Uses ATOMIC's O(1) recurrent step without any KV-cache expansion.
    """
    if config is None:
        config = GenerationConfig()

    device = torch.device(config.device)
    model.to(device)
    model.eval()

    if isinstance(prompt, str):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        prompt_tokens = list(prompt)

    if len(prompt_tokens) == 0:
        return

    stop_token_ids = set(config.stop_token_ids)
    if tokenizer.eos_id is not None:
        stop_token_ids.add(tokenizer.eos_id)

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # 1. Prefill / Prompt Processing via parallel scan
    with torch.no_grad():
        logits, _, (states, prev_xs) = model(input_ids)
        next_token_logits = logits[:, -1, :].clone()

    # Apply repetition penalty to initial candidate
    if config.repetition_penalty != 1.0:
        for past_id in set(prompt_tokens[-50:]):
            next_token_logits[0, past_id] /= config.repetition_penalty

    curr_token = _sample_next_token(
        next_token_logits,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p
    ).squeeze(-1)

    generated_ids = [curr_token.item()]
    token_str = tokenizer.decode([curr_token.item()])
    accumulated_str = token_str

    if curr_token.item() in stop_token_ids:
        return

    yield token_str

    # 2. O(1) Recurrent Autoregressive Generation
    with torch.no_grad():
        for _ in range(config.max_new_tokens - 1):
            logits, states, prev_xs = model.step(curr_token, states, prev_xs)
            step_logits = logits.clone()

            # Apply repetition penalty against recent tokens
            if config.repetition_penalty != 1.0:
                recent_window = (prompt_tokens + generated_ids)[-60:]
                for past_id in set(recent_window):
                    step_logits[0, past_id] /= config.repetition_penalty

            next_token = _sample_next_token(
                step_logits,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p
            ).squeeze(-1)

            next_id = next_token.item()
            generated_ids.append(next_id)
            curr_token = next_token

            if next_id in stop_token_ids:
                break

            chunk = tokenizer.decode([next_id])
            accumulated_str += chunk
            yield chunk

            # Check textual stop strings
            if any(s in accumulated_str for s in config.stop_strings):
                break


def generate(
    model: AtomicForCausalLM,
    tokenizer: AtomicTokenizer,
    prompt: Union[str, List[int]],
    config: Optional[GenerationConfig] = None
) -> str:
    """
    Generate text autoregressively and return the full prompt + generation.
    """
    stream = generate_stream(model, tokenizer, prompt, config)
    completion = "".join(list(stream))
    if isinstance(prompt, str):
        return prompt + completion
    return tokenizer.decode(prompt) + completion
