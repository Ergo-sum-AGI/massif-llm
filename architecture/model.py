# model.py (residual alpha)
import torch
import torch.nn as nn
from architecture.normalization import RMSNorm, NormTracked
from architecture.attention import MASSIFAttention
from architecture.ffn import SwiGLU

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