import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from Transformer import Rule30Transformer
import time

def generate_static_dataset(num_samples, seq_length):
    state_t = torch.randint(0, 2, (num_samples, seq_length), dtype=torch.long)
    padded = torch.cat([state_t[:, -1:], state_t, state_t[:, :1]], dim=1)
    left = padded[:, :-2]
    center = padded[:, 1:-1]
    right = padded[:, 2:]
    state_t_plus_1 = left ^ (center | right)
    return state_t, state_t_plus_1

def run_grokking_experiment():
    # --- GPU Optimized Hyperparameters ---
    D_MODEL = 128
    SEQ_LENGTH = 64       
    BATCH_SIZE = 128
    EPOCHS = 5000         
    WEIGHT_DECAY = 0.5    # Increased for stronger pressure
    LEARNING_RATE = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    # Generate Data
    train_x, train_y = generate_static_dataset(2000, SEQ_LENGTH)
    val_x, val_y = generate_static_dataset(1000, SEQ_LENGTH)

    # Optimized DataLoader with pin_memory
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=4, num_layers=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Gradient Scaler for AMP
    scaler = GradScaler()

    print("Initiating Grokking Training Loop (AMP Enabled)...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0; train_total = 0
        train_loss_sum = 0.0
        
        for state_t, state_t_plus_1 in train_loader:
            state_t, state_t_plus_1 = state_t.to(device), state_t_plus_1.to(device)
            optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast():
                logits = model(state_t)
                shifted_targets = torch.roll(state_t_plus_1, shifts=1, dims=1)
                logits_valid = logits[:, 2:, :]
                targets_valid = shifted_targets[:, 2:]
                loss = criterion(logits_valid.reshape(-1, 2), targets_valid.reshape(-1))
            
            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            preds = torch.argmax(logits_valid, dim=-1)
            train_correct += (preds == targets_valid).sum().item()
            train_total += targets_valid.numel()

        # Validation
        model.eval()
        val_correct = 0; val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for state_t, state_t_plus_1 in val_loader:
                state_t, state_t_plus_1 = state_t.to(device), state_t_plus_1.to(device)
                logits = model(state_t)
                shifted_targets = torch.roll(state_t_plus_1, shifts=1, dims=1)
                logits_valid = logits[:, 2:, :]
                targets_valid = shifted_targets[:, 2:]
                val_loss = criterion(logits_valid.reshape(-1, 2), targets_valid.reshape(-1))
                val_loss_sum += val_loss.item()
                preds = torch.argmax(logits_valid, dim=-1)
                val_correct += (preds == targets_valid).sum().item()
                val_total += targets_valid.numel()

        # Tracking metrics
        l2_norm = sum(p.norm(2).item() for p in model.parameters())
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:4d} | Train Acc: {(train_correct/train_total)*100:6.2f}% | Val Acc: {(val_correct/val_total)*100:6.2f}% | Val Loss: {val_loss_sum/len(val_loader):.4f} | L2 Norm: {l2_norm:.2f}")

    torch.save(model.state_dict(), "micro_rule30_pretrained.pt")
    print(f"Training Complete. Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    run_grokking_experiment()