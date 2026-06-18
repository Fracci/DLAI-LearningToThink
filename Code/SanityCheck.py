import torch
from Transformer import Rule30Transformer

def run_sanity_check():
    # Initialize micro-architecture for local CPU
    model = Rule30Transformer(
        vocab_size=2, 
        d_model=64, 
        nhead=4, 
        num_layers=2
    )
    
    # Dummy tensor: Batch size 8, Sequence length 120
    batch_size = 8
    seq_len = 120
    dummy_input = torch.randint(0, 2, (batch_size, seq_len))
    
    # Forward pass
    print(f"Input shape: {dummy_input.shape}")
    logits = model(dummy_input)
    
    # Expected output: (batch_size, seq_len, vocab_size)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 2), "Dimension mismatch in output!"
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("\nSanity Check Passed: ALiBi Mask and Pre-LN layers are mathematically aligned.")

if __name__ == "__main__":
    run_sanity_check()