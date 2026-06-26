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
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
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
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        def get_slopes(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        slopes = torch.tensor(get_slopes(self.nhead), device=device)
        
        i = torch.arange(seq_len, device=device).unsqueeze(1) 
        j = torch.arange(seq_len, device=device).unsqueeze(0) 
        distances = i - j 
        
        alibi_bias = -1 * distances.unsqueeze(0) * slopes.view(-1, 1, 1)
        
        combined_mask = alibi_bias + causal_mask.unsqueeze(0)
        
        return combined_mask.repeat(batch_size, 1, 1)

    def forward(self, src):
        batch_size, seq_len = src.shape
        
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        mask = self._get_alibi_causal_mask(seq_len, batch_size, src.device)
        
        out = self.transformer(x, mask=mask, is_causal=False)
        
        out = self.final_norm(out)
        logits = self.fc_out(out)
        
        return logits