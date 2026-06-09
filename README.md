# Building a MASSIF-Optimized Language Model: A Step-by-Step Research Guide

**Daniel Solis / Ergo Sum AGI Safety Systems**
**June 2026**

---

## Preamble: What "MASSIF-Optimized" Actually Means

A standard language model is optimized for next-token prediction loss. A MASSIF-optimized model has a second design objective: its hidden-state dynamics should be *measurable, interpretable, and controllable* under the MASSIF observables (persistence, directional alignment, curvature, radial variance, lead-lag).

This means three things in practice:

1. **Geometric legibility**: hidden states should evolve in ways that make velocity, curvature, and norm meaningful signals rather than noise artefacts of normalization choices.
2. **Dynamical regularity**: the model should have predictable, architecture-determined collapse behavior, not stochastic or norm-runaway behavior.
3. **Interventionability**: the architecture should expose handles (temperature, routing weights, recurrence gates) that allow external control of dynamical regime.

The goal is not a better language model in the MMLU/HellaSwag sense. It is a model that serves as a *dynamical laboratory* for MASSIF research, with language capability sufficient to generate meaningful recursive outputs.

---

## Stage 0: Compute Strategy and Environment Setup

### Hardware allocation

| Task | Where | Why |
|---|---|---|
| Architecture prototyping, debugging | Local HP notebook (CPU only) | Fast iteration, no cost |
| Small-scale training runs (100M, N=50 MASSIF sweeps) | SageMaker T4 16GB | Adequate for models up to ~500M |
| Full 512M training runs | SageMaker ml.g4dn.2xlarge or ml.g5.xlarge | 16-24GB VRAM, cost-effective |
| MASSIF evaluation sweeps | SageMaker T4 | Same environment as existing experiments |

### Local environment (HP notebook, 8GB RAM, 512MB GPU)

Your GPU (likely Intel integrated or a very small discrete) cannot run even 100M parameter inference in float32. Use it exclusively for CPU-based prototyping.

```bash
# Create a dedicated conda environment
conda create -n massif_llm python=3.10
conda activate massif_llm

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets tokenizers accelerate
pip install einops rotary-embedding-torch
pip install wandb  # for experiment tracking
pip install pytest  # for unit testing architecture components
```

On SageMaker, use the `pytorch-training:2.1.0-gpu-py310` base image. All pip installs are identical except torch pulls the CUDA variant automatically.

### Repository structure

Set this up locally from day one. Everything goes to GitHub.

```
massif-llm/ (crossed where populated)
│
├── architecture/                         # All model building blocks
│ x ├── config.py                         # MASSIFConfig dataclass: d_model, n_layers,
│   │                                     #   n_heads, vocab_size, track_norms, etc.
│ x ├── normalization.py                  # Decision 1: RMSNorm + NormTracked
│ x ├── attention.py                      # Decision 3: MASSIFAttention + RoPE hook
│ x ├── ffn.py                            # Decision 4: SwiGLU feed-forward block
│ x ├── model.py                          # Decision 2: MASSIFBlock (residual alpha)
│   │                                     #   + MASSIFModel (full stack, weight tying,
│   │                                     #   get_hidden_states())
│ x ├── routing.py (phase 3)              # Decision 5 (Stage 3 only): optional MoE
│ x └── architecture_configuration.yaml
│
├── massif/                               # MASSIF telemetry and training objectives
│ x ├── observables.py                    # Pure functions: persistence, flip detection,
│   │                                     #   R_t, curvature, radial variance, lead-lag
│ x ├── losses.py                         # MASSIFRegularizer + config dataclasses
│ x ├── telemetry.py                      # Forward hooks for any model layer
│ x └── taxonomy.py                       # classify_model(): metric dict -> class label
│
├── training/
│ x ├── train.py                          # Main loop: LM loss + MASSIFRegularizer
│ x ├── data.py                           # FineWeb-Edu / TinyStories dataloaders
│ x ├── scheduler.py                      # CosineAnnealingLR + warmup
│ x └── tinystories.py                    # TinyStories loader (for skeleton validation)
│
├── experiments/
│ x ├── massif_sweep.py                   # run_massif_sweep(): N=50 protocol
│   └── fourier_sweep.py                  # Frequency response: 0.1-1.0 Hz sweep
│
├── configs/
│ x ├── skeleton_100m.yaml                # 12 layers, d_model=512, ~85M params
│ x ├── full_512m.yaml                    # 24 layers, d_model=1024, ~512M params
│ x ├── ablation_none.yaml
│   ├── ablation_persist_only.yaml
│   ├── ablation_norm_only.yaml
│   ├── ablation_kappa_only.yaml
│   └── ablation_full.yaml
│
├── notebooks/
│   └── analysis.ipynb                    # Post-hoc MASSIF plots, taxonomy table
│
└── tests/
    ├── test_normalization.py
    ├── test_attention.py
    ├── test_ffn.py
    ├── test_model.py
    └── test_observables.py
```

**Which file each Stage 1 decision belongs to:**

| Decision | File | What goes there |
|---|---|---|
| 1: Pre-LN + RMSNorm | `architecture/normalization.py` | `RMSNorm`, `NormTracked` classes |
| 2: Residual alpha scaling | `architecture/model.py` | `self.alpha_attn`, `self.alpha_ffn` inside `MASSIFBlock` |
| 3: Attention + RoPE | `architecture/attention.py` | `MASSIFAttention` class |
| 4: SwiGLU FFN | `architecture/ffn.py` | `SwiGLU` class |
| 5: Optional recurrence | `architecture/routing.py` | RWKV time-mixing or MoE (Stage 3 only) |

---

## Stage 1: Architecture Design

### 1.1 The core design choices

Five architectural decisions determine MASSIF behavior. Make them deliberately.

**Decision 1: Normalization placement**

Your MASSIF data shows this is the single most important architectural variable. Pre-LayerNorm (before the sublayer) with RMSNorm gives norm-constrained hidden states (CV < 0.05). Post-LayerNorm (after the sublayer) gives norm-expanding states (CV > 0.15) and Runaway risk.

**Choose Pre-LN + RMSNorm. This is non-negotiable for a MASSIF-optimized design.**

```python
# architecture/normalization.py

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization: no mean subtraction, only scale
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

class NormTracked(RMSNorm):
    """RMSNorm with optional norm logging for MASSIF radial variance tracking."""
    def __init__(self, dim: int, eps: float = 1e-6, track: bool = False):
        super().__init__(dim, eps)
        self.track = track
        self.last_norm = None

    def forward(self, x):
        if self.track:
            self.last_norm = x.norm(dim=-1).detach()
        return super().forward(x)
```

**Decision 2: Residual connection scaling**

Standard residual connections (x + sublayer(x)) allow norm growth. Add a learnable scalar alpha per layer, initialized to 1.0, to give the model explicit control over how much each layer contributes to norm growth.

```python
# architecture/model.py  →  inside MASSIFBlock.__init__() and forward()

# In __init__:
self.alpha_attn = nn.Parameter(torch.ones(1))  # learnable residual scale
self.alpha_ffn  = nn.Parameter(torch.ones(1))

# In forward:
x = x + self.alpha_attn * self.attn(self.norm1(x), mask)
x = x + self.alpha_ffn  * self.ffn(self.norm2(x))
```

This single addition gives MASSIF a direct handle on the norm regime. During training, monitor alpha values per layer to see where the model is choosing to amplify or suppress.

**Decision 3: Attention mechanism**

Use standard multi-head attention with rotary position embeddings (RoPE). Do not use ALiBi or learned positional embeddings: RoPE gives clean rotational geometry in attention space that interacts predictably with MASSIF velocity vectors.

Add a MASSIF telemetry hook at the attention output, before the residual addition.

```python
# architecture/attention.py

from rotary_embedding_torch import RotaryEmbedding
import torch.nn as nn
import torch

class MASSIFAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.d_head)

        # MASSIF hook: stores attention output before residual
        self.massif_hook = None

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).split(D, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1,2)
                   for t in qkv]

        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)

        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2,-1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(-1)

        out = (attn @ v).transpose(1,2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        # Store for MASSIF telemetry
        if self.massif_hook is not None:
            self.massif_hook(out.detach())

        return out
```

**Decision 4: Feed-forward activation**

Use SwiGLU (as in LLaMA/DeepSeek) rather than GELU or ReLU. SwiGLU gives smoother activation landscapes that produce lower curvature trajectories, which from your MASSIF data correlates with more predictable dynamics.

```python
# architecture/ffn.py

import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expand: int = 4):
        super().__init__()
        d_inner = int(d_model * expand * 2/3)  # SwiGLU canonical sizing
        self.gate = nn.Linear(d_model, d_inner * 2, bias=False)
        self.proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        g, h = self.gate(x).chunk(2, dim=-1)
        return self.proj(torch.nn.functional.silu(g) * h)
```

**Decision 5: Optional recurrent component**

For a pure transformer skeleton (recommended for Stage 1), skip this. For Stage 3, consider inserting one RWKV-style time-mixing layer every 4 transformer layers. This gives the model explicit recurrent state that MASSIF can track separately from the transformer hidden state, and tests whether the hybrid regime (like Nemotron's decoupled norm/dynamics) is architecturally replicable.

### 1.2 The MASSIF transformer block

```python
# model.py

class MASSIFBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = NormTracked(config.d_model, track=config.track_norms)
        self.norm2 = NormTracked(config.d_model, track=config.track_norms)
        self.attn = MASSIFAttention(config)
        self.ffn = SwiGLU(config.d_model)
        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.alpha_ffn = nn.Parameter(torch.ones(1))

        # MASSIF hidden state storage
        self._hidden_state = None

    def forward(self, x, mask=None):
        # Pre-LN attention with scaled residual
        x = x + self.alpha_attn * self.attn(self.norm1(x), mask)
        # Pre-LN FFN with scaled residual
        x = x + self.alpha_ffn * self.ffn(self.norm2(x))

        # Store final hidden state for MASSIF extraction
        self._hidden_state = x.detach()
        return x


class MASSIFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([MASSIFBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (embedding and lm_head share weights)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    def get_hidden_states(self):
        """Extract final-layer hidden states for MASSIF telemetry."""
        return self.blocks[-1]._hidden_state
```

### 1.3 Architecture configurations

```yaml
# configs/skeleton_100m.yaml
d_model: 512
n_layers: 12
n_heads: 8
vocab_size: 32000
max_seq_len: 512
track_norms: true
target_params: ~85M

# configs/full_512m.yaml
d_model: 1024
n_layers: 24
n_heads: 16
vocab_size: 32000
max_seq_len: 1024
track_norms: true
target_params: ~512M
```

Parameter count formula (approximate):
`params ≈ 12 * n_layers * d_model^2`

For d_model=1024, n_layers=24: 12 * 24 * 1024^2 ≈ 302M (transformer blocks only). Add embedding, head, and intermediate projections to reach ~512M.

---

## Stage 2: The MASSIF Telemetry Layer

This is what distinguishes your model from any other small transformer. The telemetry layer is built in from the start, not added later.

### 2.1 Core observables

```python
# massif/observables.py

import torch
import numpy as np

def compute_persistence(h_sequence):
    """
    Compute persistence I_t for a sequence of hidden states.
    h_sequence: tensor of shape (T, d) - sequence of hidden states for one run
    Returns: tensor of shape (T-2,) - persistence values
    """
    eps = 1e-8
    # Normalize states
    h_norm = h_sequence / (h_sequence.norm(dim=-1, keepdim=True) + eps)
    # Velocity vectors
    v = h_norm[1:] - h_norm[:-1]
    v_norm = v / (v.norm(dim=-1, keepdim=True) + eps)
    # Persistence: cosine similarity of consecutive velocities
    persistence = (v_norm[1:] * v_norm[:-1]).sum(dim=-1)
    return persistence

def detect_flip(persistence, window=3):
    """
    Detect persistence flip: first step where mean of last `window` values > 0.
    Returns flip step index, or None if no flip detected.
    """
    for t in range(window, len(persistence)+1):
        if persistence[t-window:t].mean() > 0:
            return t
    return None

def compute_directional_alignment(h_ensemble):
    """
    Compute R_t (Kuramoto-inspired directional alignment) across ensemble.
    h_ensemble: tensor of shape (N, T, d) - N runs, T steps, d dimensions
    Returns: tensor of shape (T-2,) - R_t values
    """
    eps = 1e-8
    N, T, d = h_ensemble.shape
    h_norm = h_ensemble / (h_ensemble.norm(dim=-1, keepdim=True) + eps)
    v = h_norm[:, 1:] - h_norm[:, :-1]  # (N, T-1, d)
    u = v / (v.norm(dim=-1, keepdim=True) + eps)  # unit velocity (N, T-1, d)
    mean_u = u.mean(dim=0)  # (T-1, d)
    R_t = mean_u.norm(dim=-1)  # (T-1,)
    return R_t

def compute_curvature(h_sequence):
    """
    Compute mean curvature kappa for a hidden state sequence.
    Returns scalar mean curvature in radians.
    """
    persistence = compute_persistence(h_sequence)
    # Clamp to valid arccos range
    persistence_clamped = persistence.clamp(-1+1e-6, 1-1e-6)
    curvature = torch.arccos(persistence_clamped)
    return curvature.mean().item()

def compute_radial_variance(h_sequence):
    """
    Compute radial variance sigma_rho and CV for a hidden state sequence.
    """
    norms = h_sequence.norm(dim=-1)
    sigma = norms.std().item()
    mu = norms.mean().item()
    cv = sigma / (mu + 1e-8)
    return sigma, cv

def compute_lead_lag(R_t, flip_times):
    """
    Compute Delta_t = t_V_peak - mean_flip_time.
    R_t: directional alignment sequence
    flip_times: list of flip step indices across ensemble
    Returns: Delta_t scalar
    """
    V_t = torch.diff(R_t)
    t_V_peak = V_t.argmax().item()
    mean_flip = np.mean([f for f in flip_times if f is not None]) if any(
        f is not None for f in flip_times) else None
    if mean_flip is None:
        return None
    return t_V_peak - mean_flip

def compute_tau_eff(tau_collapse, flip_rate):
    """Effective warning: actionable precursor steps weighted by reliability."""
    return tau_collapse * (1 - flip_rate)
```

### 2.2 Full MASSIF sweep

```python
# experiments/massif_sweep.py

import torch
from massif.observables import (compute_persistence, detect_flip,
                                  compute_directional_alignment,
                                  compute_curvature, compute_radial_variance,
                                  compute_lead_lag, compute_tau_eff)

def run_massif_sweep(model, tokenizer, prompt, n_runs=50, max_tokens=30,
                     temperature=0.5, device='cuda'):
    """
    Full MASSIF N=50 sweep for a single prompt.
    Returns complete taxonomy metrics.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    all_hidden_states = []  # (N, T, d)
    all_flip_times = []
    all_curvatures = []
    all_norms = []

    with torch.no_grad():
        for run in range(n_runs):
            torch.manual_seed(run)  # independent seeds
            hidden_sequence = []
            ids = input_ids.clone()

            for step in range(max_tokens):
                logits = model(ids)
                last_logit = logits[0, -1, :] / temperature
                probs = torch.softmax(last_logit, dim=-1)
                next_token = torch.multinomial(probs, 1)
                ids = torch.cat([ids, next_token.unsqueeze(0)], dim=-1)

                h = model.get_hidden_states()
                if h is not None:
                    hidden_sequence.append(h[0, -1, :].cpu())

            if len(hidden_sequence) < 3:
                continue

            h_seq = torch.stack(hidden_sequence)  # (T, d)
            all_hidden_states.append(h_seq)

            # Per-run metrics
            persistence = compute_persistence(h_seq)
            flip_t = detect_flip(persistence)
            all_flip_times.append(flip_t)
            all_curvatures.append(compute_curvature(h_seq))
            sigma, cv = compute_radial_variance(h_seq)
            all_norms.append((sigma, cv))

    # Ensemble metrics
    h_ensemble = torch.stack(all_hidden_states)  # (N, T, d)
    R_t = compute_directional_alignment(h_ensemble)
    delta_t = compute_lead_lag(R_t, all_flip_times)

    flip_rate = sum(1 for f in all_flip_times if f is not None) / n_runs
    mean_kappa = sum(all_curvatures) / len(all_curvatures)
    mean_sigma = sum(s for s,_ in all_norms) / len(all_norms)
    mean_cv = sum(cv for _,cv in all_norms) / len(all_norms)
    max_norm = max(h_ensemble.norm(dim=-1).max().item() for _ in [1])

    # Effective warning
    valid_flips = [f for f in all_flip_times if f is not None]
    tau_collapse = sum(valid_flips) / len(valid_flips) if valid_flips else 0
    tau_eff = compute_tau_eff(tau_collapse, flip_rate)

    return {
        'flip_rate': flip_rate,
        'delta_t': delta_t,
        'tau_eff': tau_eff,
        'mean_kappa': mean_kappa,
        'mean_sigma': mean_sigma,
        'mean_cv': mean_cv,
        'max_norm': max_norm,
        'R_t': R_t.numpy(),
        'flip_times': all_flip_times,
    }
```

---

## Stage 3: The Dynamical Regularization Loss

This is the key training innovation. Standard LM training uses cross-entropy loss only. MASSIF-optimized training adds auxiliary losses that directly shape the dynamical regime.

### 3.1 Three auxiliary losses

```python
# massif/losses.py

import torch
import torch.nn.functional as F

def persistence_smoothness_loss(hidden_states_sequence):
    """
    Penalize abrupt persistence flips during training.
    Encourages gradual transitions rather than sudden attractor lock-in.
    hidden_states_sequence: (B, T, d) hidden states for a batch
    """
    eps = 1e-8
    h = hidden_states_sequence
    h_norm = h / (h.norm(dim=-1, keepdim=True) + eps)
    v = h_norm[:, 1:] - h_norm[:, :-1]  # (B, T-1, d)
    v_unit = v / (v.norm(dim=-1, keepdim=True) + eps)

    # Persistence: cosine similarity of consecutive velocities
    persistence = (v_unit[:, 1:] * v_unit[:, :-1]).sum(dim=-1)  # (B, T-2)

    # Penalize the magnitude of persistence (push toward zero: corrective regime)
    # Soft penalty: allow small positive persistence, heavily penalize large values
    loss = F.relu(persistence - 0.3).pow(2).mean()
    return loss

def norm_stability_loss(hidden_states_sequence):
    """
    Penalize norm growth across the sequence.
    Encourages norm-constrained (low sigma_rho) dynamics.
    """
    norms = hidden_states_sequence.norm(dim=-1)  # (B, T)
    # Penalize standard deviation of norms across time steps
    norm_std = norms.std(dim=-1).mean()
    # Penalize absolute norm growth
    norm_growth = (norms[:, 1:] - norms[:, :-1]).abs().mean()
    return norm_std + 0.5 * norm_growth

def curvature_target_loss(hidden_states_sequence, target_kappa=1.5):
    """
    Encourage curvature toward a target value.
    target_kappa=1.5 rad (~86 degrees) sits between persistent (low) and
    meandering (high), in the zone associated with Accelerator dynamics
    in the MASSIF taxonomy.
    """
    eps = 1e-8
    h = hidden_states_sequence
    h_norm = h / (h.norm(dim=-1, keepdim=True) + eps)
    v = h_norm[:, 1:] - h_norm[:, :-1]
    v_unit = v / (v.norm(dim=-1, keepdim=True) + eps)
    persistence = (v_unit[:, 1:] * v_unit[:, :-1]).sum(dim=-1).clamp(-1+eps, 1-eps)
    kappa = torch.arccos(persistence)
    return (kappa.mean() - target_kappa).pow(2)

def massif_regularization_loss(hidden_states, lambda_persist=0.01,
                                lambda_norm=0.005, lambda_kappa=0.01):
    """
    Combined MASSIF regularization loss.
    Weights are small: LM loss dominates, these shape dynamics at the margin.
    """
    l_persist = persistence_smoothness_loss(hidden_states)
    l_norm = norm_stability_loss(hidden_states)
    l_kappa = curvature_target_loss(hidden_states)

    return (lambda_persist * l_persist +
            lambda_norm * l_norm +
            lambda_kappa * l_kappa)
```

### 3.2 Recommended lambda values

Start with very small lambdas. The LM loss must dominate or the model will learn to minimize curvature at the expense of coherent language generation.

| Phase | lambda_persist | lambda_norm | lambda_kappa |
|---|---|---|---|
| Warmup (first 1K steps) | 0.0 | 0.0 | 0.0 |
| Early training (1K-10K) | 0.001 | 0.001 | 0.001 |
| Main training (10K+) | 0.01 | 0.005 | 0.01 |

Run a MASSIF sweep every 5K steps to monitor whether the regularization is shifting the dynamical regime in the intended direction.

---

## Stage 4: Training

### 4.1 Data

For a research instrument, not a deployed product, use a small, clean text corpus. Recommended:

- **FineWeb-Edu** (10B token subset via Hugging Face): clean, filtered web text
- **The Pile** (subset): diverse domains
- **TinyStories** (for skeleton validation): very small, coherent narratives

Total recommended: 5-10B tokens for the 512M model. This is a fraction of what production models use, but sufficient for the dynamical properties to emerge.

```python
# training/data.py
from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataloader(config):
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",
                           name="sample-10BT",
                           split="train",
                           streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    # Or train your own BPE tokenizer on the dataset for cleaner vocab
    ...
```

### 4.2 Training loop with MASSIF monitoring

```python
# training/train.py

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(model, dataloader, config, device='cuda'):
    optimizer = AdamW(model.parameters(), lr=3e-4,
                      betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_steps)

    model.train()
    step = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = input_ids[:, 1:].contiguous()

        # Forward pass
        logits = model(input_ids[:, :-1])

        # LM loss
        lm_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1)
        )

        # MASSIF regularization (from stored hidden states)
        hidden = model.get_hidden_states()  # (B, T, d)
        if hidden is not None and step > config.warmup_steps:
            reg_loss = massif_regularization_loss(hidden)
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total_loss = lm_loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log
        if step % 100 == 0:
            print(f"Step {step} | LM: {lm_loss:.4f} | Reg: {reg_loss:.4f}")

        # MASSIF sweep every 5K steps
        if step % 5000 == 0 and step > 0:
            run_checkpoint_massif_eval(model, config, step)

        step += 1
```

### 4.3 Compute estimate for 512M model

| Stage | Steps | Tokens | Time (T4 16GB) | Cost (est.) |
|---|---|---|---|---|
| Skeleton 100M validation | 10K | 10M | ~2 hours | ~$2 |
| 512M warmup | 5K | 20M | ~4 hours | ~$4 |
| 512M main training | 100K | 400M | ~80 hours | ~$80 |
| MASSIF evaluation sweeps | per checkpoint | N/A | ~1 hour each | ~$1 each |

Total realistic budget for a research-grade (not production-grade) 512M model: **$100-150 in SageMaker compute**. Well within AWS credit allocation.

---

## Stage 5: Evaluation Protocol

At each checkpoint (every 5K steps), run the full MASSIF protocol:

1. N=50 sweep on "I I I I" prompt (main taxonomy)
2. N=30 sweep on four prompt classes (invariance check)
3. Frequency sweep at 5 periodicities (spectral fingerprint)
4. Record: flip_rate, delta_t, tau_eff, mean_kappa, sigma_rho, CV, max_norm

Plot all metrics against training step. You expect to see:

- CV (radial variance coefficient) decreasing as Pre-LN + RMSNorm takes effect
- Mean curvature converging toward target_kappa (~1.5 rad) as regularization shapes dynamics
- Delta_t becoming more negative (Accelerator regime) as training progresses
- Flip rate decreasing

If the regularization is working, by checkpoint 50K you should see the model settling into an Accelerator regime with low flip rate and measurable positive tau_eff.

---

## Recommended Execution Sequence

**Week 1 (local, CPU):**
- Implement architecture components and unit-test each one
- Verify hidden state extraction returns correct shapes
- Verify MASSIF observables compute correctly on random tensors
- Train a 10M parameter toy version for 1K steps on TinyStories to verify the pipeline end-to-end

**Week 2 (SageMaker T4):**
- Train 100M skeleton for 10K steps
- Run first full MASSIF sweep
- Tune lambda values based on observed regime
- Verify the model is an Accelerator or adjust regularization

**Week 3-4 (SageMaker, sustained):**
- Scale to 512M
- Train for 50K-100K steps with periodic MASSIF checkpoints
- Compare MASSIF profile of your model against the existing 13-architecture taxonomy

**Ongoing:**
- At completion, add your model as architecture #14 in the MASSIF taxonomy
- The key research question: does a model explicitly trained toward Accelerator dynamics achieve lower flip rates and higher tau_eff than architecturally similar models trained without MASSIF regularization?

---

## The Research Question This Answers

The MASSIF taxonomy describes what architectures *are*, dynamically. The MASSIF-optimized model asks whether you can *engineer* a target dynamical class through training. If yes, that is a significant result: it means the MASSIF framework is not just descriptive but prescriptive. Architectural and training choices can be deliberately steered toward collapse-predictable regimes.

That result would be worth a follow-on paper.

---

## Success Criteria

Commit to these targets before running the experiment. They are derived from the best observed Accelerator models in the existing 13-architecture taxonomy (DeepSeek-1.3B and Nemotron-Mini-4B), scaled conservatively to account for the smaller training budget and the fact that no model has previously been trained with explicit dynamical regularization.

### Primary criteria (must achieve for the experiment to be considered successful)

| Metric | Target | Reference (best observed) | Meaning |
|---|---|---|---|
| Dynamical class | Accelerator | DeepSeek, Nemotron | Delta_t < -2, consistent |
| Delta_t | < -8.0 | -11.0 (DeepSeek, Nemotron) | Strong advance warning |
| tau_eff | > 5.0 steps | 10.3 (Nemotron), 7.7 (DeepSeek) | Actionable intervention window |
| Flip rate | < 25% | 6% (Nemotron), 30% (DeepSeek) | Collapse is the exception |
| CV (radial variance) | < 0.06 | 0.039 (DeepSeek), 0.041 (TinyLlama) | Norm-constrained regime |

### Secondary criteria (strengthens the result)

| Metric | Target | Meaning |
|---|---|---|
| Prompt invariance | Accelerator on >= 3 of 4 prompt classes | Dynamical class is architectural, not prompt-specific |
| Mean curvature (kappa) | 1.4 - 2.1 rad | In the zone of observed Accelerator models |
| Max norm | < 100 | Avoids Runaway-norm territory |
| Spectral peak | Identifiable and stable | Model has a characteristic frequency fingerprint |

### Ablation success criterion

If running the four-variant ablation (no regularization, persistence only, norm only, curvature only, all three combined), the full regularization variant should achieve strictly better primary criteria than any single-loss variant. If a single loss achieves the same result as all three combined, one or two losses are redundant and should be removed from the final architecture.

### Diagnostic failure modes

If CV < 0.06 (norm-constrained) but Delta_t > 2 (Decelerator) or tau_eff = 0, the norm regularization is working but the persistence and curvature losses are not shaping the dynamical regime as intended. This is diagnostic, not terminal: it identifies which loss is load-bearing.

If flip rate remains > 70% after 50K steps with full regularization active, the lambda values are too small to overcome the architecture's natural collapse tendency. Double lambda_persist and rerun from the 10K checkpoint.

---

## Configurable Regularization Losses

The three auxiliary losses should be fully configurable via the config file. This is not merely engineering hygiene: it enables the ablation study that makes the training contribution publishable.

### Config structure

```yaml
# configs/full_512m.yaml

# ... architecture params as before ...

massif_regularization:
  enabled: true                    # master switch
  warmup_steps: 1000               # steps before any reg loss activates

  persistence:
    enabled: true
    lambda: 0.01
    threshold: 0.3                 # penalize persistence above this value
    schedule: linear               # ramp from 0 to lambda over 10K steps

  norm_stability:
    enabled: true
    lambda: 0.005
    std_weight: 1.0
    growth_weight: 0.5
    schedule: constant

  curvature:
    enabled: true
    lambda: 0.01
    target_kappa: 1.5              # target curvature in radians (~86 degrees)
    schedule: constant
```

### Config-driven loss factory

```python
# massif/losses.py (updated)

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PersistenceConfig:
    enabled: bool = True
    lambda_weight: float = 0.01
    threshold: float = 0.3
    schedule: str = "linear"

@dataclass
class NormStabilityConfig:
    enabled: bool = True
    lambda_weight: float = 0.005
    std_weight: float = 1.0
    growth_weight: float = 0.5
    schedule: str = "constant"

@dataclass
class CurvatureConfig:
    enabled: bool = True
    lambda_weight: float = 0.01
    target_kappa: float = 1.5
    schedule: str = "constant"

@dataclass
class MASSIFRegConfig:
    enabled: bool = True
    warmup_steps: int = 1000
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    norm_stability: NormStabilityConfig = field(default_factory=NormStabilityConfig)
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)


class MASSIFRegularizer:
    """
    Config-driven MASSIF regularization loss manager.
    Supports per-loss enable/disable and independent lambda scheduling.
    """

    def __init__(self, config: MASSIFRegConfig):
        self.config = config
        self.step = 0

    def get_lambda(self, base_lambda: float, schedule: str,
                   ramp_steps: int = 10000) -> float:
        if schedule == "linear":
            ramp = min(1.0, max(0.0,
                (self.step - self.config.warmup_steps) / ramp_steps))
            return base_lambda * ramp
        return base_lambda  # constant

    def compute(self, hidden_states: torch.Tensor,
                step: Optional[int] = None) -> dict:
        """
        Compute all enabled regularization losses.
        Returns dict with individual loss values and weighted total.
        hidden_states: (B, T, d)
        """
        if step is not None:
            self.step = step

        losses = {'persistence': None, 'norm_stability': None,
                  'curvature': None, 'total': torch.tensor(0.0,
                  device=hidden_states.device)}

        if not self.config.enabled or self.step < self.config.warmup_steps:
            return losses

        eps = 1e-8
        h = hidden_states
        h_norm = h / (h.norm(dim=-1, keepdim=True) + eps)
        v = h_norm[:, 1:] - h_norm[:, :-1]
        v_unit = v / (v.norm(dim=-1, keepdim=True) + eps)
        persistence_vals = (v_unit[:, 1:] * v_unit[:, :-1]).sum(dim=-1)

        if self.config.persistence.enabled:
            cfg = self.config.persistence
            l = F.relu(persistence_vals - cfg.threshold).pow(2).mean()
            lam = self.get_lambda(cfg.lambda_weight, cfg.schedule)
            losses['persistence'] = l.item()
            losses['total'] = losses['total'] + lam * l

        if self.config.norm_stability.enabled:
            cfg = self.config.norm_stability
            norms = h.norm(dim=-1)
            l = (cfg.std_weight * norms.std(dim=-1).mean() +
                 cfg.growth_weight * (norms[:, 1:] - norms[:, :-1]).abs().mean())
            lam = self.get_lambda(cfg.lambda_weight, cfg.schedule)
            losses['norm_stability'] = l.item()
            losses['total'] = losses['total'] + lam * l

        if self.config.curvature.enabled:
            cfg = self.config.curvature
            kappa = torch.arccos(persistence_vals.clamp(-1+eps, 1-eps))
            l = (kappa.mean() - cfg.target_kappa).pow(2)
            lam = self.get_lambda(cfg.lambda_weight, cfg.schedule)
            losses['curvature'] = l.item()
            losses['total'] = losses['total'] + lam * l

        return losses
```

### Five ablation configurations

Train all five variants from the same random seed and data order. Evaluate each at the 50K step checkpoint with the full MASSIF protocol. The comparison table is the core result of the follow-on paper.

```yaml
# configs/ablation_none.yaml
massif_regularization:
  enabled: false

# configs/ablation_persist_only.yaml
massif_regularization:
  enabled: true
  persistence: {enabled: true, lambda: 0.01}
  norm_stability: {enabled: false, lambda: 0.0}
  curvature: {enabled: false, lambda: 0.0}

# configs/ablation_norm_only.yaml
massif_regularization:
  enabled: true
  persistence: {enabled: false, lambda: 0.0}
  norm_stability: {enabled: true, lambda: 0.005}
  curvature: {enabled: false, lambda: 0.0}

# configs/ablation_kappa_only.yaml
massif_regularization:
  enabled: true
  persistence: {enabled: false, lambda: 0.0}
  norm_stability: {enabled: false, lambda: 0.0}
  curvature: {enabled: true, lambda: 0.01}

# configs/ablation_full.yaml
massif_regularization:
  enabled: true
  persistence: {enabled: true, lambda: 0.01}
  norm_stability: {enabled: true, lambda: 0.005}
  curvature: {enabled: true, lambda: 0.01}
```

The ablation table in the follow-on paper will look like this:

| Config | Delta_t | tau_eff | Flip Rate | CV | Class |
|---|---|---|---|---|---|
| No regularization | ? | ? | ? | ? | ? |
| Persistence only | ? | ? | ? | ? | ? |
| Norm stability only | ? | ? | ? | ? | ? |
| Curvature only | ? | ? | ? | ? | ? |
| Full MASSIF reg | ? | ? | ? | ? | ? |
| DeepSeek-1.3B (baseline) | -11.0 | 7.7 | 30% | 0.039 | Accelerator |
| Nemotron-Mini-4B (baseline) | -11.0 | 10.3 | 6% | — | Accelerator* |

The question marks are what the experiment fills in. The baselines give the reader an immediate sense of whether the trained model reaches the level of naturally occurring best-case architectures.

---

*Daniel Solis / Ergo Sum AGI Safety Systems / solis@dubito-ergo.com*
*June 2026 / github.com/Ergo-sum-AGI/MASSIF*
