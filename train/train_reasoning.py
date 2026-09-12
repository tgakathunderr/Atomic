import os
import math
import time
import json
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from atomic.tokenizer import AtomicTokenizer
from data.reasoning_dataset import build_reasoning_corpus, ReasoningDataset, collate_reasoning_fn

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_production_reasoning_model(
    output_dir: str = "models/atomic-reasoning-prod",
    num_train_samples: int = 600,
    num_val_samples: int = 100,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2.5e-3,
    warmup_steps: int = 25,
    max_seq_len: int = 160,
    eval_every_steps: int = 25,
    device: str = "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    print("=" * 70, flush=True)
    print("      TRAINING ATOMIC PRODUCTION REASONING LANGUAGE MODEL (CPU)", flush=True)
    print(f"      Cores: 14 | Threads: {torch.get_num_threads()} | Target Output: {output_dir}", flush=True)
    print("=" * 70, flush=True)

    # 1. Initialize Tokenizer & Corpus
    print("[1/5] Initializing Tokenizer and generating reasoning corpus...", flush=True)
    tokenizer = AtomicTokenizer()
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer.json"))

    train_samples = build_reasoning_corpus(num_train_samples)
    val_samples = build_reasoning_corpus(num_val_samples)

    train_ds = ReasoningDataset(train_samples, tokenizer, max_length=max_seq_len)
    val_ds = ReasoningDataset(val_samples, tokenizer, max_length=max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_fn(b, tokenizer.pad_id)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_reasoning_fn(b, tokenizer.pad_id)
    )

    print(f"      Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Vocab: {tokenizer.vocab_size}", flush=True)

    # 2. Build Model
    print("[2/5] Initializing ATOMIC Reasoning Architecture...", flush=True)
    config = AtomicConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_head=32,
        d_ffn=512,
        max_seq_len=max_seq_len,
        dropout=0.0
    )
    config.to_json(os.path.join(output_dir, "config.json"))

    model = AtomicForCausalLM(config).to(device)
    total_params = model.num_parameters()
    print(f"      Total parameters: {total_params:,} (~{total_params/1e6:.2f}M)", flush=True)

    # 3. Setup Optimizer and Scheduler
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0}
    ], lr=lr, betas=(0.9, 0.98), eps=1e-8)

    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # 4. Training Loop
    print(f"[3/5] Starting production training across {epochs} epochs ({total_steps} total steps)...", flush=True)
    history = []
    best_val_loss = float("inf")
    global_step = 0
    t_start = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            global_step += 1
            inps, tgts = inps.to(device), tgts.to(device)

            optimizer.zero_grad()
            logits, loss, _ = model(inps, targets=tgts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % 10 == 0 or global_step == total_steps:
                print(f"Epoch {epoch}/{epochs} | Step {global_step:3d}/{total_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

            # Periodic Evaluation
            if global_step % eval_every_steps == 0 or global_step == total_steps:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for v_inps, v_tgts in val_loader:
                        v_inps, v_tgts = v_inps.to(device), v_tgts.to(device)
                        _, v_loss, _ = model(v_inps, targets=v_tgts)
                        val_losses.append(v_loss.item())

                mean_val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(min(mean_val_loss, 20.0))
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start

                step_info = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(mean_val_loss, 4),
                    "val_ppl": round(val_ppl, 2),
                    "lr": current_lr,
                    "elapsed_sec": round(elapsed, 1)
                }
                history.append(step_info)

                print(
                    f"--> [EVAL] Step {global_step:3d} | Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {mean_val_loss:.4f} | Val PPL: {val_ppl:6.2f} | Time: {elapsed:.1f}s",
                    flush=True
                )

                # Save best checkpoint
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_path = os.path.join(output_dir, "best_model.pt")
                    torch.save(model.state_dict(), best_path)

                model.train()

        epoch_time = time.time() - epoch_start
        print(f"--- Completed Epoch {epoch} in {epoch_time:.1f}s ---", flush=True)

    # 5. Export Final Checkpoint & Safetensors
    print("[4/5] Saving final model artifacts and safetensors...", flush=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    try:
        from safetensors.torch import save_model
        save_model(model, os.path.join(output_dir, "model.safetensors"))
        print("      Exported safetensors weights: model.safetensors", flush=True)
    except Exception as e:
        print(f"      Safetensors export note: {e}", flush=True)

    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    total_training_time = time.time() - t_start
    print(f"[5/5] Training Complete! Total time: {total_training_time:.1f}s | Best Val Loss: {best_val_loss:.4f}", flush=True)
    return model, history


if __name__ == "__main__":
    train_production_reasoning_model()
