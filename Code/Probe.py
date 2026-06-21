import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Import your custom modules
from Transformer import Rule30Transformer
from Rule30Generator import Rule30Dataset

# ===================================================================
# 1. Define the Shallow Probe Network
# ===================================================================
class ProbeMLP(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # A tiny 2-layer network. It is mathematically too shallow 
        # to compute Rule 30 across 256 tokens on its own.
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)  # Predicts 0 or 1
        )

    def forward(self, x):
        return self.net(x)

def run_probing_experiment():
    # Hyperparameters
    D_MODEL = 128
    SEQ_LENGTH = 256
    BATCH_SIZE = 128
    PROBE_EPOCHS = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Probing Experiment on device: {device}")

    # ===================================================================
    # 2. Load and Freeze the Pre-trained Transformer
    # ===================================================================
    transformer = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=4, num_layers=4).to(device)
    
    # Load the weights you just trained
    checkpoint_path = "rule30_pretrained_new.pt"
    try:
        transformer.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Successfully loaded pre-trained Transformer weights.")
    except Exception as e:
        print(f"Error loading weights: {e}. Make sure you ran the training script first!")
        return

    # Freeze the Transformer! We do not want its weights to change.
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False

    # ===================================================================
    # 3. Setup the Forward Hook to Extract Hidden States
    # ===================================================================
    hidden_states = {}
    
    # This hook grabs the output of the final LayerNorm (the "internal brain state")
    # right before it goes into the final vocabulary prediction head.
    def get_activation(name):
        def hook(model, input, output):
            hidden_states[name] = output.detach() # Detach to prevent gradients flowing back
        return hook
        
    transformer.final_norm.register_forward_hook(get_activation('final_norm'))

    # ===================================================================
    # 4. Initialize Probe and Data
    # ===================================================================
    probe = ProbeMLP(d_model=D_MODEL).to(device)
    probe_optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Generate completely new data that the Transformer has never seen
    train_dataset = Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # ===================================================================
    # 5. Train the Probe
    # ===================================================================
    print("\nStarting Probe Training...")
    
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for state_t, state_t_plus_1 in train_loader:
            state_t = state_t.to(device, non_blocking=True)
            state_t_plus_1 = state_t_plus_1.to(device, non_blocking=True)
            
            probe_optimizer.zero_grad()
            
            # 1. Pass data through the frozen Transformer
            with torch.no_grad():
                with autocast():
                    _ = transformer(state_t) # We don't care about the logits
                    
            # 2. Extract the hidden states saved by our hook
            # Shape: (batch_size, seq_len, d_model)
            extracted_states = hidden_states['final_norm']
            
            # 3. Align the targets (same shifting logic as training)
            shifted_targets = torch.roll(state_t_plus_1, shifts=1, dims=1)
            
            # 4. Slice off the first two causally blind tokens
            states_valid = extracted_states[:, 2:, :]
            targets_valid = shifted_targets[:, 2:]
            
            # 5. Pass states through the Probe
            with autocast():
                probe_logits = probe(states_valid)
                loss = criterion(probe_logits.reshape(-1, 2), targets_valid.reshape(-1))
                
            loss.backward()
            probe_optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(probe_logits, dim=-1)
            correct_preds += (preds == targets_valid).sum().item()
            total_preds += targets_valid.numel()
            
        acc = (correct_preds / total_preds) * 100
        print(f"Probe Epoch [{epoch+1}/{PROBE_EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Probe Accuracy: {acc:.2f}%")

    print("\nProbing Test Complete.")
    print("If the Probe Accuracy is >95%, the Transformer successfully built a causal world model!")

if __name__ == "__main__":
    run_probing_experiment()