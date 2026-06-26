import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.Transformer import GeneralTransformer
from data_generation.Rule30Generator import Rule30Dataset

def train_kaggle_gpu():
    # 1. Hyperparameters
    VOCAB_SIZE = 2
    D_MODEL = 256         
    NHEAD = 8
    NUM_LAYERS = 6
    
    SEQ_LENGTH = 256       
    BATCH_SIZE = 128      
    EPOCHS = 300         
    NUM_SAMPLES = 20000
    
    WEIGHT_DECAY = 0.2   
    LEARNING_RATE = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 3 Pre-Training on device: {device}")

    # 2. Initialize Architecture and Data
    model = GeneralTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=1024,
    ).to(device)

    dataset = Rule30Dataset(num_samples=NUM_SAMPLES, seq_length=SEQ_LENGTH)
    
    # GPU-Optimized DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True,       
        num_workers=2         
    )

    # 3. Loss, Optimizer, and AMP Configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scaler = GradScaler()

    # 4. Training Loop
    start_time = time.time()
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (state_t, state_t_plus_1) in enumerate(dataloader):
            state_t = state_t.to(device, non_blocking=True)
            state_t_plus_1 = state_t_plus_1.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with autocast():
                logits = model(state_t)
                
                shifted_targets = torch.roll(state_t_plus_1, shifts=1, dims=1)
                logits_valid = logits[:, 2:, :]
                targets_valid = shifted_targets[:, 2:]
                
                loss = criterion(logits_valid.reshape(-1, VOCAB_SIZE), targets_valid.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits_valid, dim=-1)
            correct_predictions += (preds == targets_valid).sum().item()
            total_predictions += targets_valid.numel()

        # Epoch Reporting
        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_predictions) * 100
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # 5. Save the Checkpoint
    checkpoint_path = "rule30_pretrained_new.pt"
    torch.save(model.state_dict(), checkpoint_path)
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining complete in {elapsed_time/60:.2f} minutes.")
    print(f"Model weights saved to '{checkpoint_path}'.")

if __name__ == "__main__":
    train_kaggle_gpu()