"""
Transformer.py — the shared model architecture for every experiment.

Defines GeneralTransformer: a small decoder-style causal Transformer (encoder
layers used autoregressively via a causal mask) with ALiBi relative-position
biases instead of learned/sinusoidal positional embeddings. The same class is reused 
for every pretraining task and for the arithmetic fine-tuning; only `vocab_size` 
changes per task.
"""
import torch
import torch.nn as nn
import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import ModelConfig


class GeneralTransformer(nn.Module):
    """Causal Transformer with ALiBi positional biasing; vocab_size varies per task."""

    def __init__(
        self,
        vocab_size=2,
        d_model=ModelConfig.d_model,
        nhead=ModelConfig.n_heads,
        num_layers=ModelConfig.n_layers,
        dim_feedforward=ModelConfig.dim_feedforward,
        dropout=ModelConfig.dropout
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Token embedding. Only this layer (and fc_out) are vocab-specific, so
        # transfer between tasks drops these two and keeps the transformer body.
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Pre-LN encoder layers (norm_first=True) used as a causal decoder via the
        # additive mask built in forward().
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _get_alibi_causal_mask(self, seq_len, batch_size, device):
        """Build the combined (causal + ALiBi) additive attention mask for one forward pass.

        Returns shape (batch_size*nhead, seq_len, seq_len): a -inf causal triangle
        plus a per-head linear distance penalty (ALiBi). Rebuilt every forward;
        fine for these sequence lengths but is the main avoidable compute cost.
        """
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # ALiBi slopes, one per head. NOTE: this is the power-of-2 formula only.
        def get_slopes(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        slopes = torch.tensor(get_slopes(self.nhead), device=device)

        # Signed token distance i-j; ALiBi penalizes attention by distance * slope.
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)
        distances = i - j

        alibi_bias = -1 * distances.unsqueeze(0) * slopes.view(-1, 1, 1)   # (nhead, S, S)

        combined_mask = alibi_bias + causal_mask.unsqueeze(0)              # (nhead, S, S)

        # Tile per batch element. The ALiBi bias is the SAME for every sample, so
        # repeating the per-head block once per sample is correct here.
        return combined_mask.repeat(batch_size, 1, 1)                      # (batch*nhead, S, S)

    def forward(self, src):
        """Embed tokens, run causal+ALiBi attention, return per-position vocab logits."""
        batch_size, seq_len = src.shape

        # Scale embeddings by sqrt(d_model), standard Transformer convention.
        x = self.embedding(src) * math.sqrt(self.d_model)

        mask = self._get_alibi_causal_mask(seq_len, batch_size, src.device)

        # is_causal=False because causality is supplied explicitly via `mask`.
        out = self.transformer(x, mask=mask, is_causal=False)

        out = self.final_norm(out)
        logits = self.fc_out(out)
        return logits