// architecture/routing.py — Decision 5: Optional Recurrent Component
// Two drop-in options for Stage 3. Pick one; do not combine both in the same model.

const code = `# architecture/routing.py
#
# Decision 5: Optional recurrent component (Stage 3 only)
#
# Two options are provided:
#   Option A — RWKVTimeMix:  a lightweight time-mixing layer in the style of RWKV-4.
#              Adds explicit recurrent state that MASSIF can track separately from the
#              transformer hidden state. Insert one every 4 transformer blocks.
#
#   Option B — ExpertRouter:  a simple 2-expert mixture-of-experts routing layer.
#              Gives MASSIF a clean routing signal to track (which expert is active,
#              and whether routing changes at the flip boundary).
#
# Integration in architecture/model.py:
#   from architecture.routing import RWKVTimeMix, ExpertRouter
#
#   In MASSIFModel.__init__(), after building self.blocks:
#     if config.use_rwkv_mixing:
#         self.time_mix_layers = nn.ModuleList([
#             RWKVTimeMix(config.d_model)
#             for _ in range(config.n_layers // 4)
#         ])
#
#   In MASSIFModel.forward(), after every 4th block:
#     if config.use_rwkv_mixing and layer_idx % 4 == 3:
#         x = self.time_mix_layers[layer_idx // 4](x, state)
#
# Config flags to add to architecture/config.py:
#   use_rwkv_mixing: bool = False   # Option A
#   use_moe_routing: bool = False   # Option B
#   n_experts: int = 2              # Option B: number of experts
#
# MASSIF telemetry hooks for Option A:
#   The time-mixed state (self._last_time_state) is a separate observable from
#   the transformer hidden state. Run parallel MASSIF sweeps on both to test
#   whether the recurrent state dynamics predict collapse earlier than the
#   transformer hidden state dynamics.
#
# MASSIF telemetry hooks for Option B:
#   Log self._last_routing_weights per step. A routing weight flip (expert A
#   dominant -> expert B dominant) near the persistence flip time is a
#   candidate mechanistic correlate of collapse.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Option A: RWKV-style time-mixing layer
# =============================================================================

class RWKVTimeMix(nn.Module):
    """
    Lightweight RWKV-4 time-mixing layer for hybrid Transformer + recurrence.

    Placed every 4 transformer blocks. Mixes the current token's hidden state
    with an exponentially-decaying memory of previous states using learnable
    time-decay and time-first parameters.

    MASSIF relevance:
        self._last_time_state stores the recurrent state after each forward pass.
        Track this separately from the transformer hidden state in massif_sweep.py
        to test whether recurrent state dynamics predict collapse earlier.

    Args:
        d_model (int): Hidden dimension (must match MASSIFBlock.d_model).
        expand (float): Inner dimension multiplier. Default 1.0 (no expansion).
    """

    def __init__(self, d_model: int, expand: float = 1.0):
        super().__init__()
        d_inner = int(d_model * expand)

        # Learnable time-decay: one value per channel, initialized to slow decay
        self.time_decay = nn.Parameter(torch.zeros(d_inner))

        # Learnable time-first: bonus applied at t=0
        self.time_first = nn.Parameter(torch.full((d_inner,), fill_value=-3.0))

        # Time-mixing coefficients: how much of x_t vs x_{t-1} to blend
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))

        # Projections
        self.key   = nn.Linear(d_model, d_inner, bias=False)
        self.value = nn.Linear(d_model, d_inner, bias=False)
        self.receptance = nn.Linear(d_model, d_inner, bias=False)
        self.output = nn.Linear(d_inner, d_model, bias=False)

        # Layer norm on output
        self.ln = nn.LayerNorm(d_model)

        # MASSIF telemetry: stores recurrent state for external inspection
        self._last_time_state = None

    def forward(self, x: torch.Tensor,
                state: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:     (B, T, d_model) hidden states from transformer block
            state: (B, d_model) recurrent state from previous call, or None

        Returns:
            x_out: (B, T, d_model) time-mixed hidden states
        """
        B, T, D = x.shape

        # Initialize state if not provided
        if state is None:
            state = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        out_list = []
        for t in range(T):
            x_t = x[:, t, :]    # (B, D)
            x_prev = state       # (B, D) — previous token's hidden state

            # Blend current and previous token for key, value, receptance
            xk = x_t * self.time_mix_k[0, 0] + x_prev * (1 - self.time_mix_k[0, 0])
            xv = x_t * self.time_mix_v[0, 0] + x_prev * (1 - self.time_mix_v[0, 0])
            xr = x_t * self.time_mix_r[0, 0] + x_prev * (1 - self.time_mix_r[0, 0])

            k = self.key(xk)           # (B, d_inner)
            v = self.value(xv)         # (B, d_inner)
            r = torch.sigmoid(self.receptance(xr))  # (B, d_inner), gate in [0,1]

            # RWKV WKV mechanism: decay-weighted sum
            # time_decay is negative log of decay, so exp(time_decay) in (0,1)
            decay = torch.exp(-torch.exp(self.time_decay))  # (d_inner,)
            wkv = (torch.exp(self.time_first) * v +
                   decay.unsqueeze(0) * k * v) / \
                  (torch.exp(self.time_first) + decay.unsqueeze(0) * k + 1e-8)

            out_t = self.output(r * wkv)   # (B, d_model)
            out_list.append(out_t)

            state = x_t  # update recurrent state to current token

        # Store final recurrent state for MASSIF telemetry
        self._last_time_state = state.detach()

        x_mixed = torch.stack(out_list, dim=1)  # (B, T, d_model)

        # Residual + layer norm (Pre-LN style)
        return self.ln(x + x_mixed)


# =============================================================================
# Option B: Simple Mixture-of-Experts routing layer
# =============================================================================

class ExpertFFN(nn.Module):
    """Single SwiGLU feed-forward expert."""

    def __init__(self, d_model: int, d_inner: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_inner * 2, bias=False)
        self.proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, h = self.gate(x).chunk(2, dim=-1)
        return self.proj(F.silu(g) * h)


class ExpertRouter(nn.Module):
    """
    Simple top-1 Mixture-of-Experts routing layer.

    Routes each token to one of n_experts feed-forward experts based on a
    learned routing function. Designed as a drop-in replacement for
    architecture/ffn.py SwiGLU in selected MASSIFBlock positions.

    MASSIF relevance:
        self._last_routing_weights stores the per-token routing distribution
        at each forward pass. A routing weight flip near the persistence flip
        time is a candidate mechanistic correlate of collapse. Log these weights
        in massif_sweep.py alongside the standard MASSIF observables.

    Args:
        d_model (int):   Hidden dimension.
        n_experts (int): Number of experts. Default 2.
        expand (int):    FFN expansion factor per expert. Default 4.
    """

    def __init__(self, d_model: int, n_experts: int = 2, expand: int = 4):
        super().__init__()
        self.n_experts = n_experts
        d_inner = int(d_model * expand * 2/3)  # SwiGLU canonical sizing

        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_inner) for _ in range(n_experts)
        ])

        # Router: linear projection to n_experts logits
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Load balancing auxiliary loss weight
        self.aux_loss_weight = 0.01

        # MASSIF telemetry: stores routing weights for external inspection
        self._last_routing_weights = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model) hidden states

        Returns:
            out:      (B, T, d_model) expert-mixed output
            aux_loss: scalar load-balancing loss (add to total_loss during training)
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)   # (B*T, D)

        # Compute routing weights
        logits = self.router(x_flat)           # (B*T, n_experts)
        weights = F.softmax(logits, dim=-1)    # (B*T, n_experts)

        # Store for MASSIF telemetry (reshape to (B, T, n_experts))
        self._last_routing_weights = weights.view(B, T, self.n_experts).detach()

        # Top-1 routing: select the expert with highest weight per token
        top1_idx = weights.argmax(dim=-1)      # (B*T,)

        # Compute expert outputs
        out = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            mask = (top1_idx == expert_idx)    # (B*T,) boolean
            if mask.any():
                expert_input = x_flat[mask]    # (n_selected, D)
                expert_out = expert(expert_input)
                out[mask] = expert_out

        # Load balancing auxiliary loss (encourages equal expert utilization)
        # Reference: Switch Transformer (Fedus et al., 2021)
        fraction_routed = weights.mean(dim=0)          # (n_experts,) mean routing prob
        fraction_tokens = (F.one_hot(top1_idx, self.n_experts)
                           .float().mean(dim=0))       # (n_experts,) fraction of tokens
        aux_loss = self.aux_loss_weight * self.n_experts * \
                   (fraction_routed * fraction_tokens).sum()

        return out.view(B, T, D), aux_loss


# =============================================================================
# Integration example for architecture/model.py
# =============================================================================
#
# To use Option A (RWKVTimeMix), modify MASSIFModel as follows:
#
#   class MASSIFModel(nn.Module):
#       def __init__(self, config):
#           ...
#           if config.use_rwkv_mixing:
#               self.time_mixers = nn.ModuleList([
#                   RWKVTimeMix(config.d_model)
#                   for _ in range(config.n_layers // 4)
#               ])
#
#       def forward(self, input_ids, mask=None):
#           x = self.embedding(input_ids)
#           recurrent_state = None
#           for i, block in enumerate(self.blocks):
#               x = block(x, mask)
#               if self.config.use_rwkv_mixing and (i + 1) % 4 == 0:
#                   mixer_idx = i // 4
#                   x = self.time_mixers[mixer_idx](x, recurrent_state)
#                   # Update recurrent state from mixer's stored state
#                   recurrent_state = self.time_mixers[mixer_idx]._last_time_state
#           x = self.final_norm(x)
#           return self.lm_head(x)
#
# To use Option B (ExpertRouter), replace SwiGLU in selected MASSIFBlock layers:
#
#   class MASSIFBlock(nn.Module):
#       def __init__(self, config, use_moe=False):
#           ...
#           if use_moe:
#               self.ffn = ExpertRouter(config.d_model, config.n_experts)
#           else:
#               self.ffn = SwiGLU(config.d_model)
#           self.use_moe = use_moe
#
#       def forward(self, x, mask=None):
#           x = x + self.alpha_attn * self.attn(self.norm1(x), mask)
#           if self.use_moe:
#               ffn_out, aux_loss = self.ffn(self.norm2(x))
#               self._last_aux_loss = aux_loss   # retrieve in training loop
#           else:
#               ffn_out = self.ffn(self.norm2(x))
#               self._last_aux_loss = None
#           x = x + self.alpha_ffn * ffn_out
#           self._hidden_state = x.detach()
#           return x
#
# In training/train.py, collect and add aux_loss from MoE blocks:
#
#   moe_aux = sum(
#       b._last_aux_loss for b in model.blocks
#       if b.use_moe and b._last_aux_loss is not None
#   )
#   total_loss = lm_loss + reg_loss + moe_aux
`;

document.getElementById('code-out').textContent = code;
