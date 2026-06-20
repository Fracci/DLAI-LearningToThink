import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Import your custom modules
from Transformer import Rule30Transformer
from Rule30Generator import Rule30Dataset

def train_local_cpu():
    # ---------------------------------------------------------
    # 1. Hyperparameters (Scaled down for local CPU execution)
    # ---------------------------------------------------------
    VOCAB_SIZE = 2
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 4
    
    SEQ_LENGTH = 128       # Manageable sequence length for CPU
    BATCH_SIZE = 32       # Smaller batch size for local memory
    EPOCHS = 300           # Number of passes over the virtual dataset
    NUM_SAMPLES = 5000    # Virtual samples per epoch generated on-the-fly
    
    WEIGHT_DECAY = 0.1   # High weight decay as specified in the plan
    LEARNING_RATE = 1e-3
    
    device = torch.device("cpu")
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

    # The dataset generates completely novel sequences every iteration 
    # to prevent memorization loss collapse.
    dataset = Rule30Dataset(num_samples=NUM_SAMPLES, seq_length=SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---------------------------------------------------------
    # 3. Loss & Optimizer Configuration
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (state_t, state_t_plus_1) in enumerate(dataloader):
            state_t = state_t.to(device)
            state_t_plus_1 = state_t_plus_1.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            logits = model(state_t)
            
            # 1. Shift dei target a destra (il token i prevede la cella i-1)
            shifted_targets = torch.roll(state_t_plus_1, shifts=1, dims=1)

            # 2. Slicing per rimuovere i primi 2 token causalmente ciechi
            logits_valid = logits[:, 2:, :]
            targets_valid = shifted_targets[:, 2:]

            # 3. Calcolo della Loss rigorosamente SUI TOKEN VALIDI
            loss = criterion(logits_valid.reshape(-1, VOCAB_SIZE), targets_valid.reshape(-1))

            loss.backward()
        
            optimizer.step()
            
            total_loss += loss.item()
            
            # 4. Calcolo dell'Accuracy SUI TOKEN VALIDI
            predictions = torch.argmax(logits_valid, dim=-1)
            
            # L'errore era probabilmente qui: assicurati di usare 'targets_valid', NON 'state_t_plus_1'
            correct_predictions += (predictions == targets_valid).sum().item()
            total_predictions += targets_valid.numel()

        # Epoch Reporting
        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # ---------------------------------------------------------
    # 5. Save the Micro Checkpoint
    # ---------------------------------------------------------
    checkpoint_path = "Code/micro_rule30_pretrained.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining complete. Model weights saved to '{checkpoint_path}'.")

if __name__ == "__main__":
    train_local_cpu()