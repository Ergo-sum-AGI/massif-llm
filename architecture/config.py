# architecture/config.py
#
# Central configuration dataclass for the MASSIF-optimized LLM.
# Every other module imports MASSIFConfig from here.
#
# Usage:
#   from architecture.config import MASSIFConfig
#   config = MASSIFConfig.from_yaml("configs/skeleton_100m.yaml")
#   config = MASSIFConfig.skeleton_100m()   # convenience constructor
#   config = MASSIFConfig.full_512m()       # convenience constructor

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml
import json
import os


@dataclass
class MASSIFConfig:
    """
    Unified configuration for architecture, training, and MASSIF telemetry.

    Sections:
        Architecture    — model structure
        Normalization   — norm placement and tracking
        Recurrence      — optional Stage 3 components (routing.py)
        Training        — optimizer, scheduler, batch
        MASSIF          — telemetry and regularization switches
        Data            — dataset and tokenizer paths
        Logging         — WandB and checkpoint settings
    """

    # -----------------------------------------------------------------------
    # Architecture
    # -----------------------------------------------------------------------
    d_model:      int   = 512       # hidden dimension
    n_layers:     int   = 12        # number of MASSIFBlock layers
    n_heads:      int   = 8         # attention heads (d_model must be divisible)
    vocab_size:   int   = 32000     # tokenizer vocabulary size
    max_seq_len:  int   = 512       # maximum sequence length
    ffn_expand:   int   = 4         # SwiGLU expansion factor
    rope_base:    int   = 10000     # RoPE base frequency

    # -----------------------------------------------------------------------
    # Normalization (Decision 1)
    # -----------------------------------------------------------------------
    norm_eps:     float = 1e-6      # RMSNorm epsilon
    track_norms:  bool  = True      # log hidden state norms via NormTracked

    # -----------------------------------------------------------------------
    # Recurrence — Stage 3 optional components (architecture/routing.py)
    # -----------------------------------------------------------------------
    use_rwkv_mixing: bool = False   # Option A: insert RWKVTimeMix every 4 blocks
    use_moe_routing: bool = False   # Option B: replace SwiGLU with ExpertRouter
    n_experts:       int  = 2       # Option B: number of MoE experts
    moe_every_n:     int  = 4       # Option B: insert MoE every N blocks

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    learning_rate:    float = 3e-4
    weight_decay:     float = 0.1
    beta1:            float = 0.9
    beta2:            float = 0.95
    grad_clip:        float = 1.0
    batch_size:       int   = 32
    grad_accum_steps: int   = 1     # effective batch = batch_size * grad_accum_steps
    total_steps:      int   = 100_000
    warmup_steps:     int   = 1_000
    min_lr_ratio:     float = 0.1   # min LR = learning_rate * min_lr_ratio

    # -----------------------------------------------------------------------
    # MASSIF regularization (massif/losses.py)
    # -----------------------------------------------------------------------
    massif_reg_enabled:    bool  = True
    massif_warmup_steps:   int   = 1_000   # steps before any reg loss activates

    # Persistence loss
    persist_enabled:   bool  = True
    persist_lambda:    float = 0.01
    persist_threshold: float = 0.3    # penalize persistence above this value
    persist_schedule:  str   = "linear"

    # Norm stability loss
    norm_stab_enabled:     bool  = True
    norm_stab_lambda:      float = 0.005
    norm_stab_std_weight:  float = 1.0
    norm_stab_grow_weight: float = 0.5
    norm_stab_schedule:    str   = "constant"

    # Curvature loss
    kappa_enabled:   bool  = True
    kappa_lambda:    float = 0.01
    kappa_target:    float = 1.5      # target curvature in radians (~86 degrees)
    kappa_schedule:  str   = "constant"

    # -----------------------------------------------------------------------
    # MASSIF evaluation protocol
    # -----------------------------------------------------------------------
    massif_eval_every:  int = 5_000   # run MASSIF sweep every N training steps
    massif_n_runs:      int = 50      # ensemble size for taxonomy sweep
    massif_n_runs_inv:  int = 30      # ensemble size for prompt invariance sweep
    massif_n_runs_fft:  int = 20      # ensemble size for Fourier sweep
    massif_prompt:      str = "I " * 4
    massif_temperature: float = 0.5
    massif_max_tokens:  int   = 30

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    dataset_name:   str = "HuggingFaceFW/fineweb-edu"
    dataset_split:  str = "train"
    tokenizer_name: str = "gpt2"      # or path to custom trained tokenizer
    num_workers:    int = 4

    # -----------------------------------------------------------------------
    # Logging and checkpointing
    # -----------------------------------------------------------------------
    project_name:     str  = "massif-llm"
    run_name:         str  = "skeleton_100m"
    checkpoint_dir:   str  = "checkpoints"
    checkpoint_every: int  = 10_000
    log_every:        int  = 100
    use_wandb:        bool = True

    # -----------------------------------------------------------------------
    # Derived properties (read-only, computed from above)
    # -----------------------------------------------------------------------

    @property
    def d_head(self) -> int:
        """Attention head dimension."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads

    @property
    def n_time_mixers(self) -> int:
        """Number of RWKVTimeMix layers inserted (Option A)."""
        return self.n_layers // 4 if self.use_rwkv_mixing else 0

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def param_estimate(self) -> int:
        """Rough parameter count estimate (transformer blocks only)."""
        return 12 * self.n_layers * self.d_model ** 2

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=True)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> MASSIFConfig:
        # Only pass keys that exist as fields to allow forward compatibility
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> MASSIFConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_json(cls, path: str) -> MASSIFConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # -----------------------------------------------------------------------
    # Convenience constructors matching the guide's yaml configs
    # -----------------------------------------------------------------------

    @classmethod
    def skeleton_100m(cls) -> MASSIFConfig:
        """~85M parameter skeleton for local CPU prototyping and pipeline validation."""
        return cls(
            d_model=512,
            n_layers=12,
            n_heads=8,
            vocab_size=32000,
            max_seq_len=512,
            total_steps=10_000,
            batch_size=16,
            run_name="skeleton_100m",
            massif_eval_every=2_000,
        )

    @classmethod
    def full_512m(cls) -> MASSIFConfig:
        """~512M parameter full model for SageMaker T4 training."""
        return cls(
            d_model=1024,
            n_layers=24,
            n_heads=16,
            vocab_size=32000,
            max_seq_len=1024,
            total_steps=100_000,
            batch_size=32,
            run_name="full_512m",
            massif_eval_every=5_000,
        )

    @classmethod
    def ablation_none(cls) -> MASSIFConfig:
        """Full 512M model with all MASSIF regularization disabled."""
        cfg = cls.full_512m()
        cfg.run_name = "ablation_none"
        cfg.massif_reg_enabled = False
        return cfg

    @classmethod
    def ablation_persist_only(cls) -> MASSIFConfig:
        """Full 512M model with persistence loss only."""
        cfg = cls.full_512m()
        cfg.run_name = "ablation_persist_only"
        cfg.norm_stab_enabled = False
        cfg.kappa_enabled = False
        return cfg

    @classmethod
    def ablation_norm_only(cls) -> MASSIFConfig:
        """Full 512M model with norm stability loss only."""
        cfg = cls.full_512m()
        cfg.run_name = "ablation_norm_only"
        cfg.persist_enabled = False
        cfg.kappa_enabled = False
        return cfg

    @classmethod
    def ablation_kappa_only(cls) -> MASSIFConfig:
        """Full 512M model with curvature loss only."""
        cfg = cls.full_512m()
        cfg.run_name = "ablation_kappa_only"
        cfg.persist_enabled = False
        cfg.norm_stab_enabled = False
        return cfg

    @classmethod
    def ablation_full(cls) -> MASSIFConfig:
        """Full 512M model with all three MASSIF losses active."""
        cfg = cls.full_512m()
        cfg.run_name = "ablation_full"
        return cfg

    def __post_init__(self):
        """Validate configuration on construction."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"
        assert self.massif_warmup_steps <= self.warmup_steps or \
               self.massif_warmup_steps <= self.total_steps, \
            "massif_warmup_steps exceeds total_steps"
        assert self.persist_schedule in ("linear", "constant"), \
            f"Unknown schedule: {self.persist_schedule}"
        if self.use_rwkv_mixing and self.use_moe_routing:
            raise ValueError(
                "use_rwkv_mixing and use_moe_routing cannot both be True. "
                "Pick one Stage 3 option from architecture/routing.py."
            )

    def __repr__(self) -> str:
        return (
            f"MASSIFConfig("
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}, vocab={self.vocab_size}, "
            f"~{self.param_estimate/1e6:.0f}M params, "
            f"reg={'full' if self.massif_reg_enabled else 'off'}"
            f")"
        )
