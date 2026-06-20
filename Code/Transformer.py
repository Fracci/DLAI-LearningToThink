import torch
import torch.nn as nn
import math

class Rule30Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size=2,          
        d_model=128,           
        nhead=8,               
        num_layers=4,          
        dim_feedforward=512, 
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 1. Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Hybrid Causal Transformer (Decoder logic using Encoder primitives)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            norm_first=True,    # CRITICAL: Pre-LN configuration for training stability
            batch_first=True    # Expects (Batch, Sequence, Feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Standard final LayerNorm applied after Pre-LN blocks
        self.final_norm = nn.LayerNorm(d_model)
        
        # 3. Task Head
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _get_alibi_causal_mask(self, seq_len, batch_size, device):
        """
        Generates a 3D causal mask combined with ALiBi relative position biases.
        This entirely replaces absolute sinusoidal embeddings, enabling strict
        shift-invariance and Out-of-Distribution (OOD) length extrapolation.
        """
        # 1. Base Causal Mask: Upper triangular matrix of -inf (blocks future tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # 2. ALiBi Slope Generation (Linear progression for localized 1D Automata)
        def get_slopes(n):
            # Creates steep penalties [0.5, 1.0, 1.5, 2.0] for 4 heads.
            # This mathematically isolates local cells and breaks binary symmetry.
            return [0.5 * (i + 1) for i in range(n)]
        
        slopes = torch.tensor(get_slopes(self.nhead), device=device)
        
        # 3. Distance Matrix: Calculates distance between query (i) and key (j)
        i = torch.arange(seq_len, device=device).unsqueeze(1) # Rows (queries)
        j = torch.arange(seq_len, device=device).unsqueeze(0) # Columns (keys)
        distances = i - j # Positive values for past tokens
        
        # 4. ALiBi Bias: -m * distance
        # Shape: (nhead, seq_len, seq_len)
        alibi_bias = -1 * distances.unsqueeze(0) * slopes.view(-1, 1, 1)
        
        # 5. Merge ALiBi with Causal Mask
        # The causal_mask adds -inf to future tokens, while alibi adds smooth penalties to past tokens
        combined_mask = alibi_bias + causal_mask.unsqueeze(0)
        
        # 6. PyTorch requirement: 3D mask shape must be (batch_size * nhead, seq_len, seq_len)
        # We repeat the mask for every item in the batch
        return combined_mask.repeat(batch_size, 1, 1)

    def forward(self, src):
        """
        src shape: (batch_size, seq_len)
        """
        batch_size, seq_len = src.shape
        
        # Embed and scale
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # Generate the ALiBi-infused causal mask dynamically based on sequence length
        mask = self._get_alibi_causal_mask(seq_len, batch_size, src.device)
        
        # Forward pass through the Transformer
        # We pass the custom mask to 'mask' (which applies to the attention weights)
        out = self.transformer(x, mask=mask, is_causal=False) # is_causal=False because our custom mask handles it
        
        # Final norm and linear projection
        out = self.final_norm(out)
        logits = self.fc_out(out)
        
        return logits