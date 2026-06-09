# architecture/ffn.py (Feed-forward activation)

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