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
    # 1. Setup & Hyperparameters (Scaled Architecture)
    # ---------------------------------------------------------
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    
    BATCH_SIZE = 256
    EPOCHS = 3000        
    WARMUP_EPOCHS = 50   # The embedding alignment phase
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
    # 3. Model A: Pre-Trained Engine (With Freeze Protocol)
    # ---------------------------------------------------------
    model_A = Rule30Transformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, 
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)
    
    # Load the perfect Rule 30 weights
    pretrained_path = "rule30_pretrained_new.pt"
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    
    # Filter out the old binary embeddings and output head
    filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("embedding.") and not k.startswith("fc_out.")}
    model_A.load_state_dict(filtered_dict, strict=False)
    print("Model A: Loaded Pre-Trained Weights.")

    # FREEZE the core transformer layers for Warmup
    for name, param in model_A.named_parameters():
        if "transformer" in name or "final_norm" in name:
            param.requires_grad = False

    # ---------------------------------------------------------
    # 4. Model B: Random Baseline
    # ---------------------------------------------------------
    model_B = Rule30Transformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, 
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)
    print("Model B: Initialized with random weights.")

    # ---------------------------------------------------------
    # 4.5. Multi-GPU Configuration
    # ---------------------------------------------------------
    if torch.cuda.device_count() > 1:
        print(f"Optimizing for {torch.cuda.device_count()} GPUs via DataParallel...")
        model_A = nn.DataParallel(model_A)
        model_B = nn.DataParallel(model_B)

    # ---------------------------------------------------------
    # 5. Optimization 
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    
    # Model A starts by only training the embeddings/head at a high learning rate
    opt_A = AdamW(filter(lambda p: p.requires_grad, model_A.parameters()), lr=1e-3, weight_decay=0.1)
    
    # Model B trains end-to-end normally
    opt_B = AdamW(model_B.parameters(), lr=5e-4, weight_decay=0.1)
    
    scaler_A = GradScaler()
    scaler_B = GradScaler()

    # ---------------------------------------------------------
    # 6. The A/B Training Loop
    # ---------------------------------------------------------
    print("\nStarting A/B Training Loop...")
    
    for epoch in range(EPOCHS):
        
        # --- STAGE 2 TRANSITION FOR MODEL A ---
        if epoch == WARMUP_EPOCHS:
            print("\n>>> WARMUP COMPLETE: Unfreezing Model A core layers for full end-to-end tuning. <<<")
            # Unfreeze all parameters, automatically bypassing the 'module.' wrapper if using DataParallel
            for param in model_A.parameters():
                param.requires_grad = True
            opt_A = AdamW(model_A.parameters(), lr=5e-5, weight_decay=0.1)

        model_A.train()
        model_B.train()
        
        train_correct_A = 0; train_correct_B = 0
        total_train_tokens = 0
        
        # Initialize loss trackers
        total_train_loss_A = 0.0
        total_train_loss_B = 0.0
        
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
            total_train_loss_A += loss_A.item()
            
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
            total_train_loss_B += loss_B.item()
            
            total_train_tokens += valid_mask.sum().item()

        # --- Validation (OOD Length Generalization) ---
        model_A.eval()
        model_B.eval()
        val_correct_A = 0; val_correct_B = 0
        total_val_tokens = 0
        
        # Initialize validation loss trackers
        total_val_loss_A = 0.0
        total_val_loss_B = 0.0
        
        with torch.no_grad():
            for x_val, y_val in ood_val_loader:
                x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                valid_mask_val = (y_val != tokenizer.pad_idx)
                
                with autocast():
                    # Evaluate Model A
                    logits_val_A = model_A(x_val)
                    loss_val_A = criterion(logits_val_A.reshape(-1, VOCAB_SIZE), y_val.reshape(-1))
                    preds_val_A = torch.argmax(logits_val_A, dim=-1)
                    
                    val_correct_A += (preds_val_A[valid_mask_val] == y_val[valid_mask_val]).sum().item()
                    total_val_loss_A += loss_val_A.item()
                    
                    # Evaluate Model B
                    logits_val_B = model_B(x_val)
                    loss_val_B = criterion(logits_val_B.reshape(-1, VOCAB_SIZE), y_val.reshape(-1))
                    preds_val_B = torch.argmax(logits_val_B, dim=-1)
                    
                    val_correct_B += (preds_val_B[valid_mask_val] == y_val[valid_mask_val]).sum().item()
                    total_val_loss_B += loss_val_B.item()
                    
                total_val_tokens += valid_mask_val.sum().item()

        # --- Reporting ---
        acc_train_A = (train_correct_A / total_train_tokens) * 100
        acc_train_B = (train_correct_B / total_train_tokens) * 100
        acc_val_A = (val_correct_A / total_val_tokens) * 100
        acc_val_B = (val_correct_B / total_val_tokens) * 100
        
        # Calculate Average Losses
        avg_train_loss_A = total_train_loss_A / len(train_loader)
        avg_train_loss_B = total_train_loss_B / len(train_loader)
        avg_val_loss_A = total_val_loss_A / len(ood_val_loader)
        avg_val_loss_B = total_val_loss_B / len(ood_val_loader)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            stage_str = "[WARMUP]" if epoch < WARMUP_EPOCHS else "[TUNING]"
            print(f"Epoch [{epoch+1:4d}/{EPOCHS}] {stage_str}")
            print(f"  Model A | Train Loss: {avg_train_loss_A:.4f} | Val Loss: {avg_val_loss_A:.4f} | Train Acc: {acc_train_A:6.2f}% | Val Acc: {acc_val_A:6.2f}%")
            print(f"  Model B | Train Loss: {avg_train_loss_B:.4f} | Val Loss: {avg_val_loss_B:.4f} | Train Acc: {acc_train_B:6.2f}% | Val Acc: {acc_val_B:6.2f}%")
            
            print("  --- Sample Output (Model A) ---")
            target_str = tokenizer.decode(y_val[0])
            pred_str = tokenizer.decode(preds_val_A[0])
            print(f"  Target: {target_str}")
            print(f"  Pred A: {pred_str}")
            print("-" * 90)
        
        # Save intermediate checkpoints
        if epoch > WARMUP_EPOCHS and epoch % 100 == 0:
            torch.save(model_A.state_dict(), f"modelA_phase5_epoch_{epoch}.pt")
            torch.save(model_B.state_dict(), f"modelB_phase5_epoch_{epoch}.pt")

if __name__ == "__main__":
    run_phase5_ab_test()