# attention.py

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