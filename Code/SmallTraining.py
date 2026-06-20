import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time

# Import your custom modules
from Transformer import Rule30Transformer
from Rule30Generator import Rule30Dataset

def train_kaggle_gpu():
    # ---------------------------------------------------------
    # 1. Hyperparameters (Scaled UP for Kaggle GPU execution)
    # ---------------------------------------------------------
    VOCAB_SIZE = 2
    D_MODEL = 128          # Increased model width
    NHEAD = 4
    NUM_LAYERS = 4
    
    SEQ_LENGTH = 256       # Longer sequences test the memory buffer better
    BATCH_SIZE = 128       # Maximize GPU VRAM usage
    EPOCHS = 150           
    NUM_SAMPLES = 20000    # Larger virtual dataset per epoch
    
    WEIGHT_DECAY = 0.1   
    LEARNING_RATE = 1e-3
    
    # Auto-detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 3 Pre-Training on device: {device}")

    # ---------------------------------------------------------
    # 2. Initialize Architecture and Data
    # ---------------------------------------------------------
    model = Rule30Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS
    ).to(device)

    dataset = Rule30Dataset(num_samples=NUM_SAMPLES, seq_length=SEQ_LENGTH)
    
    # GPU-Optimized DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True,       # Crucial: Page-locked memory for fast GPU transfer
        num_workers=2          # Multi-threading for data generation
    )

    # ---------------------------------------------------------
    # 3. Loss, Optimizer, and AMP Configuration
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # GradScaler for Automatic Mixed Precision (FP16 speedup)
    scaler = GradScaler()

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    start_time = time.time()
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (state_t, state_t_plus_1) in enumerate(dataloader):
            # non_blocking=True prevents CPU from waiting for the GPU transfer
            state_t = state_t.to(device, non_blocking=True)
            state_t_plus_1 = state_t_plus_1.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Use AMP for forward pass to double training speed
            with autocast():
                logits = model(state_t)
                
                # Causal shifting logic
                shifted_targets = torch.roll(state_t_plus_1, shifts=1, dims=1)
                logits_valid = logits[:, 2:, :]
                targets_valid = shifted_targets[:, 2:]
                
                loss = criterion(logits_valid.reshape(-1, VOCAB_SIZE), targets_valid.reshape(-1))

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Accuracy tracking
            preds = torch.argmax(logits_valid, dim=-1)
            correct_predictions += (preds == targets_valid).sum().item()
            total_predictions += targets_valid.numel()

        # Epoch Reporting
        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_predictions) * 100
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # ---------------------------------------------------------
    # 5. Save the Checkpoint to Kaggle Output
    # ---------------------------------------------------------
    checkpoint_path = "/kaggle/working/rule30_pretrained_gpu.pt"
    torch.save(model.state_dict(), checkpoint_path)
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining complete in {elapsed_time/60:.2f} minutes.")
    print(f"Model weights saved to '{checkpoint_path}'.")

if __name__ == "__main__":
    train_kaggle_gpu()