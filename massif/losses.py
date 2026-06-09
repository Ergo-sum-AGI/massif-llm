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