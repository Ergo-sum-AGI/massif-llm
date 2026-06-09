# architecture/routing.py
#
# Decision 5: Optional recurrent component (Stage 3 only)
#
# Two drop-in options are provided. Pick ONE; do not combine both in the same model.
#
#   Option A — RWKVTimeMix:  lightweight RWKV-4 time-mixing layer.
#              Adds explicit recurrent state that MASSIF can track separately
#              from the transformer hidden state.
#              Insert one every 4 transformer blocks.
#
#   Option B — ExpertRouter: simple top-1 mixture-of-experts routing layer.
#              Gives MASSIF a clean routing signal (which expert is active,
#              and whether routing flips near the persistence flip boundary).
#              Replaces SwiGLU in selected MASSIFBlock positions.
#
# Config flags to add to architecture/config.py:
#   use_rwkv_mixing: bool = False   # Option A
#   use_moe_routing: bool = False   # Option B
#   n_experts: int = 2              # Option B only

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Option A: RWKV-style time-mixing layer
# ===========================================================================

class RWKVTimeMix(nn.Module):
    """
    Lightweight RWKV-4 time-mixing layer for hybrid Transformer + recurrence.

    Placed every 4 transformer blocks. Mixes the current token's hidden state
    with an exponentially-decaying memory of previous states using learnable
    time-decay and time-first parameters.

    File: architecture/routing.py
    Used in: architecture/model.py — see integration notes at bottom of file.

    MASSIF telemetry:
        self._last_time_state stores the recurrent state after each forward pass.
        Track this alongside the transformer hidden state in experiments/massif_sweep.py
        to test whether recurrent dynamics predict collapse earlier than
        the standard MASSIF observables.

    Args:
        d_model (int):   Hidden dimension. Must match MASSIFBlock.d_model.
        expand (float):  Inner dimension multiplier. Default 1.0.
    """

    def __init__(self, d_model: int, expand: float = 1.0):
        super().__init__()
        d_inner = int(d_model * expand)

        # Learnable time-decay: one value per inner channel
        # Initialized to zero; exponentiated in forward, so effective decay in (0, 1)
        self.time_decay = nn.Parameter(torch.zeros(d_inner))

        # Learnable time-first: bonus weight applied at t=0
        self.time_first = nn.Parameter(torch.full((d_inner,), fill_value=-3.0))

        # Time-mixing coefficients: blend of x_t vs x_{t-1} for each projection
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))

        # Projections (no bias, consistent with Pre-LN design)
        self.key        = nn.Linear(d_model, d_inner, bias=False)
        self.value      = nn.Linear(d_model, d_inner, bias=False)
        self.receptance = nn.Linear(d_model, d_inner, bias=False)
        self.output     = nn.Linear(d_inner, d_model, bias=False)

        # Pre-LN on output (consistent with MASSIFBlock normalization convention)
        self.ln = nn.LayerNorm(d_model)

        # MASSIF telemetry: recurrent state after last forward pass
        self._last_time_state = None

    def forward(self, x: torch.Tensor,
                state: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:     (B, T, d_model) — hidden states from preceding transformer block
            state: (B, d_model)    — recurrent state from previous step, or None

        Returns:
            (B, T, d_model) — time-mixed hidden states (residual + LN applied)
        """
        B, T, D = x.shape

        if state is None:
            state = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        out_list = []
        for t in range(T):
            x_t    = x[:, t, :]   # current token  (B, D)
            x_prev = state         # previous token (B, D)

            # Blend current and previous for key, value, receptance
            xk = x_t * self.time_mix_k[0, 0] + x_prev * (1 - self.time_mix_k[0, 0])
            xv = x_t * self.time_mix_v[0, 0] + x_prev * (1 - self.time_mix_v[0, 0])
            xr = x_t * self.time_mix_r[0, 0] + x_prev * (1 - self.time_mix_r[0, 0])

            k = self.key(xk)                           # (B, d_inner)
            v = self.value(xv)                         # (B, d_inner)
            r = torch.sigmoid(self.receptance(xr))     # (B, d_inner), gate in [0,1]

            # RWKV WKV: decay-weighted value accumulation
            decay = torch.exp(-torch.exp(self.time_decay))   # (d_inner,) in (0,1)
            wkv   = (
                torch.exp(self.time_first) * v
                + decay.unsqueeze(0) * k * v
            ) / (
                torch.exp(self.time_first)
                + decay.unsqueeze(0) * k
                + 1e-8
            )

            out_list.append(self.output(r * wkv))   # (B, d_model)
            state = x_t                              # advance recurrent state

        # Store final recurrent state for MASSIF telemetry
        self._last_time_state = state.detach()

        x_mixed = torch.stack(out_list, dim=1)       # (B, T, d_model)
        return self.ln(x + x_mixed)                  # residual + Pre-LN


# ===========================================================================
# Option B: Simple Mixture-of-Experts routing layer
# ===========================================================================

class ExpertFFN(nn.Module):
    """Single SwiGLU expert. Internal to ExpertRouter."""

    def __init__(self, d_model: int, d_inner: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_inner * 2, bias=False)
        self.proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, h = self.gate(x).chunk(2, dim=-1)
        return self.proj(F.silu(g) * h)


class ExpertRouter(nn.Module):
    """
    Top-1 Mixture-of-Experts routing layer (drop-in replacement for SwiGLU).

    Routes each token to one of n_experts feed-forward experts based on a
    learned routing function. Intended as a replacement for architecture/ffn.py
    SwiGLU in selected MASSIFBlock positions (e.g. every 4th block).

    File: architecture/routing.py
    Used in: architecture/model.py — see integration notes at bottom of file.

    MASSIF telemetry:
        self._last_routing_weights stores the per-token routing distribution
        after each forward pass, shape (B, T, n_experts).
        Log these alongside standard MASSIF observables in massif_sweep.py.
        A routing weight flip near the persistence flip time is a candidate
        mechanistic correlate of collapse onset.

    Args:
        d_model (int):   Hidden dimension.
        n_experts (int): Number of experts. Default 2 (binary routing).
        expand (int):    FFN expansion factor per expert. Default 4.
    """

    def __init__(self, d_model: int, n_experts: int = 2, expand: int = 4):
        super().__init__()
        self.n_experts = n_experts
        d_inner = int(d_model * expand * 2 / 3)   # SwiGLU canonical sizing

        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_inner) for _ in range(n_experts)
        ])

        # Router: linear projection from hidden state to expert logits
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Load balancing loss weight (Switch Transformer, Fedus et al. 2021)
        self.aux_loss_weight = 0.01

        # MASSIF telemetry: routing weights (B, T, n_experts)
        self._last_routing_weights = None

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, T, d_model) hidden states

        Returns:
            out:      (B, T, d_model) — expert-mixed output
            aux_loss: scalar — load-balancing loss; add to total_loss in train.py
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Routing distribution
        logits  = self.router(x_flat)              # (B*T, n_experts)
        weights = F.softmax(logits, dim=-1)        # (B*T, n_experts)

        # Store for MASSIF telemetry
        self._last_routing_weights = weights.view(B, T, self.n_experts).detach()

        # Top-1 dispatch
        top1_idx = weights.argmax(dim=-1)          # (B*T,)
        out = torch.zeros_like(x_flat)
        for idx, expert in enumerate(self.experts):
            mask = (top1_idx == idx)
            if mask.any():
                out[mask] = expert(x_flat[mask])

        # Load balancing auxiliary loss
        # Penalizes routing collapse (all tokens to one expert)
        frac_prob   = weights.mean(dim=0)                              # (n_experts,)
        frac_tokens = F.one_hot(top1_idx, self.n_experts).float().mean(dim=0)
        aux_loss = self.aux_loss_weight * self.n_experts * (frac_prob * frac_tokens).sum()

        return out.view(B, T, D), aux_loss


# ===========================================================================
# Integration notes for architecture/model.py
# ===========================================================================
#
# OPTION A — Insert RWKVTimeMix every 4 transformer blocks:
#
#   In MASSIFModel.__init__():
#       if config.use_rwkv_mixing:
#           self.time_mixers = nn.ModuleList([
#               RWKVTimeMix(config.d_model)
#               for _ in range(config.n_layers // 4)
#           ])
#
#   In MASSIFModel.forward():
#       recurrent_state = None
#       for i, block in enumerate(self.blocks):
#           x = block(x, mask)
#           if self.config.use_rwkv_mixing and (i + 1) % 4 == 0:
#               x = self.time_mixers[i // 4](x, recurrent_state)
#               recurrent_state = self.time_mixers[i // 4]._last_time_state
#
# OPTION B — Replace SwiGLU with ExpertRouter in selected MASSIFBlock layers:
#
#   In MASSIFBlock.__init__(config, use_moe=False):
#       self.ffn     = ExpertRouter(config.d_model, config.n_experts) if use_moe \
#                      else SwiGLU(config.d_model)
#       self.use_moe = use_moe
#
#   In MASSIFBlock.forward():
#       if self.use_moe:
#           ffn_out, self._last_aux_loss = self.ffn(self.norm2(x))
#       else:
#           ffn_out = self.ffn(self.norm2(x))
#           self._last_aux_loss = None
#       x = x + self.alpha_ffn * ffn_out
#
#   In training/train.py, collect aux_loss across MoE blocks:
#       moe_aux = sum(
#           b._last_aux_loss for b in model.blocks
#           if b.use_moe and b._last_aux_loss is not None
#       )
#       total_loss = lm_loss + reg_loss + moe_aux
#
# MASSIF sweep additions for both options (experiments/massif_sweep.py):
#
#   Option A — after each generation step, record recurrent state norm:
#       if hasattr(model, 'time_mixers'):
#           for mixer in model.time_mixers:
#               if mixer._last_time_state is not None:
#                   recurrent_norms.append(
#                       mixer._last_time_state.norm(dim=-1).mean().item()
#                   )
#
#   Option B — after each generation step, record routing entropy:
#       for block in model.blocks:
#           if block.use_moe and block.ffn._last_routing_weights is not None:
#               w = block.ffn._last_routing_weights[0, -1, :]  # last token
#               entropy = -(w * (w + 1e-8).log()).sum().item()
#               routing_entropies.append(entropy)
#       # Routing entropy near zero = collapsed to one expert.
#       # Compare entropy timeline against persistence flip time (t_flip).
