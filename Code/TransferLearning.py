import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time

from Transformer import Rule30Transformer
from ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset

def run_phase5_ab_test():
    # ---------------------------------------------------------
    # 1. Setup & Hyperparameters
    # ---------------------------------------------------------
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 4
    BATCH_SIZE = 128
    EPOCHS = 3000        # Increased to allow the Grokking phase shift
    MAX_SEQ_LEN = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 5 A/B Test on: {device}")

    tokenizer = CharTokenizer()
    VOCAB_SIZE = tokenizer.vocab_size

    # ---------------------------------------------------------
    # 2. Data Engineering: In-Distribution & OOD
    # ---------------------------------------------------------
    train_dataset = ScratchpadAdditionDataset(
        num_samples=15000, min_digits=3, max_digits=4, 
        tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    ood_val_dataset = ScratchpadAdditionDataset(
        num_samples=1000, min_digits=5, max_digits=5, 
        tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN + 32
    )
    ood_val_loader = DataLoader(ood_val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    # ---------------------------------------------------------
    # 3. Model A: The Rule 30 Pre-Trained Engine
    # ---------------------------------------------------------
    # We use the EXACT SAME architecture, just with a wider vocabulary
    model_A = Rule30Transformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(device)
    
    # Load the perfect Rule 30 weights
    pretrained_path = "rule30_pretrained_gpu.pt"
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    
    # Filter out the old binary embeddings and output head
    filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("embedding.") and not k.startswith("fc_out.")}
    model_A.load_state_dict(filtered_dict, strict=False)
    print("Model A: Loaded Rule 30 Causal Attention Maps (Embeddings/Head re-initialized).")

    # ---------------------------------------------------------
    # 4. Model B: The Random Baseline
    # ---------------------------------------------------------
    model_B = Rule30Transformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(device)
    print("Model B: Initialized with completely random weights.")

    # ---------------------------------------------------------
    # 5. DIFFERENTIAL OPTIMIZATION (THE FIX)
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    
    # Isolate Model A's new components vs pre-trained components
    pretrained_params_A = []
    new_params_A = []
    for name, param in model_A.named_parameters():
        if "embedding" in name or "fc_out" in name:
            new_params_A.append(param)
        else:
            pretrained_params_A.append(param)
            
    # Protect the pre-trained "brain" with a tiny learning rate, but train the new head fast
    opt_A = AdamW([
        {'params': new_params_A, 'lr': 1e-3},          
        {'params': pretrained_params_A, 'lr': 2e-5}    # 50x smaller LR!
    ], weight_decay=0.1)
    
    # Model B is entirely random, so it learns globally at a standard rate
    opt_B = AdamW(model_B.parameters(), lr=5e-4, weight_decay=0.1)
    
    scaler_A = GradScaler()
    scaler_B = GradScaler()

    # ---------------------------------------------------------
    # 6. The A/B Training Loop
    # ---------------------------------------------------------
    print("\nStarting A/B Training Loop (Protected Transfer)...")
    
    for epoch in range(EPOCHS):
        model_A.train()
        model_B.train()
        
        train_correct_A = 0; train_correct_B = 0
        total_train_tokens = 0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            valid_mask = (y != tokenizer.pad_idx)
            
            # --- Train Model A ---
            opt_A.zero_grad()
            with autocast():
                logits_A = model_A(x)
                loss_A = criterion(logits_A.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            scaler_A.scale(loss_A).backward()
            scaler_A.step(opt_A)
            scaler_A.update()
            
            preds_A = torch.argmax(logits_A, dim=-1)
            train_correct_A += (preds_A[valid_mask] == y[valid_mask]).sum().item()
            
            # --- Train Model B ---
            opt_B.zero_grad()
            with autocast():
                logits_B = model_B(x)
                loss_B = criterion(logits_B.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            scaler_B.scale(loss_B).backward()
            scaler_B.step(opt_B)
            scaler_B.update()
            
            preds_B = torch.argmax(logits_B, dim=-1)
            train_correct_B += (preds_B[valid_mask] == y[valid_mask]).sum().item()
            
            total_train_tokens += valid_mask.sum().item()

        # --- Validation (OOD Length Generalization) ---
        model_A.eval()
        model_B.eval()
        val_correct_A = 0; val_correct_B = 0
        total_val_tokens = 0
        
        with torch.no_grad():
            for x_val, y_val in ood_val_loader:
                x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                valid_mask_val = (y_val != tokenizer.pad_idx)
                
                with autocast():
                    logits_val_A = model_A(x_val)
                    preds_val_A = torch.argmax(logits_val_A, dim=-1)
                    val_correct_A += (preds_val_A[valid_mask_val] == y_val[valid_mask_val]).sum().item()
                    
                    logits_val_B = model_B(x_val)
                    preds_val_B = torch.argmax(logits_val_B, dim=-1)
                    val_correct_B += (preds_val_B[valid_mask_val] == y_val[valid_mask_val]).sum().item()
                    
                total_val_tokens += valid_mask_val.sum().item()

        # --- Reporting ---
        acc_train_A = (train_correct_A / total_train_tokens) * 100
        acc_train_B = (train_correct_B / total_train_tokens) * 100
        acc_val_A = (val_correct_A / total_val_tokens) * 100
        acc_val_B = (val_correct_B / total_val_tokens) * 100
        
        # Print every 5 epochs to keep the console clean
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}]")
            print(f"  Model A (Protected Rule 30) | Train: {acc_train_A:6.2f}% | OOD (5-digit): {acc_val_A:6.2f}%")
            print(f"  Model B (Random Baseline)   | Train: {acc_train_B:6.2f}% | OOD (5-digit): {acc_val_B:6.2f}%")
            print("-" * 65)

if __name__ == "__main__":
    run_phase5_ab_test()