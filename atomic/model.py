import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from atomic.config import AtomicConfig
from atomic.layers import RMSNorm, AtomicBlock

class AtomicModel(nn.Module):
    """
    Core ATOMIC Model: Embedding -> Stack of Atomic Blocks -> Final RMSNorm.
    Supports dual-mode execution:
      1. Parallel forward for training (O(T) time).
      2. Recurrent step for streaming CPU inference (O(1) time & space).
    """
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            AtomicBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])
        
        self.ln_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self._init_weights()

    def _init_weights(self):
        # Gaussian init with std=initializer_range
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=self.config.initializer_range)
        for name, p in self.named_parameters():
            if "w_q" in name or "w_k" in name or "w_v" in name or "w_o" in name:
                nn.init.normal_(p, mean=0.0, std=self.config.initializer_range)
            elif "w_gate" in name or "w_up" in name:
                nn.init.normal_(p, mean=0.0, std=self.config.initializer_range)
            elif "w_down" in name:
                # Scaled initialization for residual projection
                std = self.config.initializer_range / math.sqrt(2 * self.config.n_layers)
                nn.init.normal_(p, mean=0.0, std=std)

    def init_states(self, batch_size: int, device=None, dtype=None) -> List[torch.Tensor]:
        """Initialize GLRA recurrent state matrices across all layers."""
        return [block.init_state(batch_size, device=device, dtype=dtype) for block in self.blocks]

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        prev_xs: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Parallel sequence forward pass.
        
        Args:
            input_ids: (B, T)
            states: list of layer states (B, H, dk, dv)
            prev_xs: list of (prev_glra_x, prev_ffn_x) for each layer
            
        Returns:
            hidden_states: (B, T, d_model)
            new_states: list of updated states
            new_prev_xs: list of updated (last_glra_x, last_ffn_x)
        """
        B, T = input_ids.shape
        x = self.drop(self.embeddings(input_ids))

        if states is None:
            states = [None] * len(self.blocks)
        if prev_xs is None:
            prev_xs = [(None, None)] * len(self.blocks)

        new_states = []
        new_prev_xs = []

        for i, block in enumerate(self.blocks):
            state_i = states[i]
            prev_glra_x, prev_ffn_x = prev_xs[i]
            x, new_s, last_glra_x, last_ffn_x = block(
                x, state=state_i, prev_glra_x=prev_glra_x, prev_ffn_x=prev_ffn_x
            )
            new_states.append(new_s)
            new_prev_xs.append((last_glra_x, last_ffn_x))

        x = self.ln_f(x)
        return x, new_states, new_prev_xs

    def step(
        self,
        token_id: torch.Tensor,
        states: List[torch.Tensor],
        prev_xs: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        O(1) Recurrent step for single-token CPU inference.
        
        Args:
            token_id: (B,)
            states: list of layer states (B, H, dk, dv)
            prev_xs: list of (prev_glra_x, prev_ffn_x) for each layer
            
        Returns:
            hidden_t: (B, d_model)
            new_states: updated states
            new_prev_xs: updated previous token vectors
        """
        x_t = self.embeddings(token_id)

        if prev_xs is None:
            prev_xs = [(None, None)] * len(self.blocks)

        new_states = []
        new_prev_xs = []

        for i, block in enumerate(self.blocks):
            state_i = states[i]
            prev_glra_x, prev_ffn_x = prev_xs[i]
            x_t, new_s, curr_glra_x, curr_ffn_x = block.step(
                x_t, state=state_i, prev_glra_x=prev_glra_x, prev_ffn_x=prev_ffn_x
            )
            new_states.append(new_s)
            new_prev_xs.append((curr_glra_x, curr_ffn_x))

        x_t = self.ln_f(x_t)
        return x_t, new_states, new_prev_xs


class AtomicForCausalLM(nn.Module):
    """
    Atomic Causal Language Model with LM Head.
    Ideal for production CPU reasoning and generation.
    """
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.config = config
        self.model = AtomicModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.weight

    def init_states(self, batch_size: int, device=None, dtype=None) -> List[torch.Tensor]:
        return self.model.init_states(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
        prev_xs: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Training forward pass.
        
        Args:
            input_ids: (B, T)
            targets: (B, T) optional next-token targets
            states: optional recurrence states
            prev_xs: optional token-shift buffers
            
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss if targets provided, else None
            cache: (new_states, new_prev_xs)
        """
        hidden_states, new_states, new_prev_xs = self.model(
            input_ids, states=states, prev_xs=prev_xs
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            # Flatten to compute CrossEntropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss, (new_states, new_prev_xs)

    def step(
        self,
        token_id: torch.Tensor,
        states: List[torch.Tensor],
        prev_xs: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Inference step: (B,) -> next-token logits (B, vocab_size) in O(1) time.
        """
        hidden_t, new_states, new_prev_xs = self.model.step(
            token_id, states=states, prev_xs=prev_xs
        )
        logits = self.lm_head(hidden_t)
        return logits, new_states, new_prev_xs

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count parameters."""
        if exclude_embeddings:
            emb_params = sum(p.numel() for p in self.model.embeddings.parameters())
            return sum(p.numel() for p in self.parameters()) - emb_params
        return sum(p.numel() for p in self.parameters())
