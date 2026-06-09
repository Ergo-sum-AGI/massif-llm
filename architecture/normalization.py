# normalization.py

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