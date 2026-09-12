import os
import sys
import time
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from atomic.tokenizer import AtomicTokenizer
from atomic.generate import generate_stream, GenerationConfig

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_GRAY = "\033[90m"
COLOR_BOLD = "\033[1m"
COLOR_MAGENTA = "\033[35m"


def print_banner(config: AtomicConfig, num_params: int):
    print(f"{COLOR_CYAN}{COLOR_BOLD}")
    print("=" * 72)
    print("      ATOMIC: Adaptive Token Operator for Memory-efficient Inference")
    print("                  Interactive CPU Reasoning Console                  ")
    print("=" * 72)
    print(f"{COLOR_RESET}")
    print(f"  * Architecture:      ATOMIC GLRA + Pre-RMSNorm SwiGLU")
    print(f"  * Model Parameters:  {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"  * Hidden Dimension:  d_model={config.d_model}, layers={config.num_layers}, heads={config.num_heads}")
    print(f"  * Recurrent State:   64.0 KB fixed per stream (O(1) memory)")
    print(f"  * Inference Mode:    Recurrent Step Engine (Zero KV-Cache expansion)")
    print("-" * 72)
    print(f"Commands: {COLOR_YELLOW}exit{COLOR_RESET} to quit | {COLOR_YELLOW}clear{COLOR_RESET} to clear console | {COLOR_YELLOW}info{COLOR_RESET} for specs\n")


def stream_response(
    model: AtomicForCausalLM,
    tokenizer: AtomicTokenizer,
    prompt: str,
    gen_config: GenerationConfig
):
    print(f"\n{COLOR_BOLD}ATOMIC Reasoning Trace:{COLOR_RESET}")

    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0

    in_think_block = False
    in_answer_block = False

    stream = generate_stream(model, tokenizer, prompt, gen_config)

    for chunk in stream:
        token_count += 1
        if first_token_time is None:
            first_token_time = time.perf_counter()

        # Parse reasoning tags for visual styling
        if "<think>" in chunk:
            in_think_block = True
            print(f"{COLOR_YELLOW}<think>{COLOR_GRAY}", end="", flush=True)
            chunk = chunk.replace("<think>", "")
        if "</think>" in chunk:
            in_think_block = False
            print(f"{COLOR_RESET}{COLOR_YELLOW}</think>{COLOR_RESET}", end="", flush=True)
            chunk = chunk.replace("</think>", "")
        if "<answer>" in chunk:
            in_answer_block = True
            print(f"\n{COLOR_GREEN}{COLOR_BOLD}<answer>", end="", flush=True)
            chunk = chunk.replace("<answer>", "")
        if "</answer>" in chunk:
            in_answer_block = False
            print(f"</answer>{COLOR_RESET}", end="", flush=True)
            chunk = chunk.replace("</answer>", "")

        if in_think_block:
            print(f"{COLOR_GRAY}{chunk}", end="", flush=True)
        elif in_answer_block:
            print(f"{COLOR_GREEN}{COLOR_BOLD}{chunk}", end="", flush=True)
        else:
            print(chunk, end="", flush=True)

    t_end = time.perf_counter()
    total_time = t_end - t0
    gen_time = (t_end - first_token_time) if first_token_time else total_time
    tok_per_sec = (token_count / gen_time) if gen_time > 0 else 0.0

    print(f"\n\n{COLOR_MAGENTA}[Metrics: {token_count} tokens generated in {total_time:.2f}s | Speed: {tok_per_sec:.1f} tok/s | Recurrent Cache: 64 KB fixed]{COLOR_RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="ATOMIC Interactive Reasoning Console")
    parser.add_argument("--model-dir", type=str, default="models/atomic-reasoning-prod", help="Path to model directory")
    parser.add_argument("--prompt", type=str, default=None, help="Direct prompt for single inference")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max-tokens", type=int, default=150, help="Maximum new tokens to generate")
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, "config.json")
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    weights_path = os.path.join(args.model_dir, "best_model.pt")

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    config = AtomicConfig.from_json(config_path)
    tokenizer = AtomicTokenizer.from_pretrained(tokenizer_path)
    model = AtomicForCausalLM(config)

    # Load weights
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    model.eval()

    num_params = model.num_parameters()
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=1.1,
        stop_strings=["</answer>", "<eos>"]
    )

    if args.prompt:
        formatted_prompt = f"Question: {args.prompt}\n<think>\n" if "<think>" not in args.prompt else args.prompt
        stream_response(model, tokenizer, formatted_prompt, gen_config)
        return

    print_banner(config, num_params)

    while True:
        try:
            user_input = input(f"{COLOR_CYAN}{COLOR_BOLD}Prompt > {COLOR_RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting ATOMIC Console. Goodbye!")
                break
            if user_input.lower() in ("clear", "cls"):
                os.system("cls" if os.name == "nt" else "clear")
                print_banner(config, num_params)
                continue
            if user_input.lower() == "info":
                print_banner(config, num_params)
                continue

            formatted_prompt = f"Question: {user_input}\n<think>\n"
            stream_response(model, tokenizer, formatted_prompt, gen_config)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting ATOMIC Console. Goodbye!")
            break


if __name__ == "__main__":
    main()
