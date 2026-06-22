import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
# Aggiornate le importazioni AMP per rimuovere i FutureWarning
from torch.amp import autocast, GradScaler
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
    START_EPOCH = 200    # Definisce l'epoca di ripartenza
    MAX_SEQ_LEN = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 5 A/B Test Recovery on: {device}")

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
    # 3. Model A: Load Checkpoint & Sblocca Pesi
    # ---------------------------------------------------------
    model_A = Rule30Transformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, 
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)
    
    checkpoint_path_A = f"modelA_phase5_epoch_{START_EPOCH}.pt"
    state_dict_A = torch.load(checkpoint_path_A, map_location=device)
    
    # Rimuove il prefisso 'module.' nel caso il checkpoint sia stato salvato sotto DataParallel
    unwrapped_dict_A = {k.replace("module.", ""): v for k, v in state_dict_A.items()}
    model_A.load_state_dict(unwrapped_dict_A)
    print(f"Model A: Loaded Checkpoint from Epoch {START_EPOCH}.")

    # La fase di Warmup è già avvenuta. Sblocca tutto immediatamente.
    for param in model_A.parameters():
        param.requires_grad = True

    # ---------------------------------------------------------
    # 4. Model B: Random Baseline
    # ---------------------------------------------------------
    model_B = Rule30Transformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, 
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)
    print("Model B: Initialized with random weights (Restarting from 0).")

    # ---------------------------------------------------------
    # 4.5. Multi-GPU Configuration
    # ---------------------------------------------------------
    if torch.cuda.device_count() > 1:
        print(f"Optimizing for {torch.cuda.device_count()} GPUs via DataParallel...")
        model_A = nn.DataParallel(model_A)
        model_B = nn.DataParallel(model_B)

    # ---------------------------------------------------------
    # 5. Optimization (Nuovi Ottimizzatori e Regolarizzazione)
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    
    # >>> MODIFICA CRITICA: Aumento della pressione di regolarizzazione <<<
    # Inizializzando l'ottimizzatore ex-novo si azzera il momento accumulato
    opt_A = AdamW(model_A.parameters(), lr=5e-5, weight_decay=0.2) 
    
    opt_B = AdamW(model_B.parameters(), lr=5e-4, weight_decay=0.1)
    
    scaler_A = GradScaler('cuda')
    scaler_B = GradScaler('cuda')

    # ---------------------------------------------------------
    # 6. The A/B Training Loop
    # ---------------------------------------------------------
    print("\nStarting A/B Training Loop...")
    
    for epoch in range(START_EPOCH, EPOCHS):
        model_A.train()
        model_B.train()
        
        train_correct_A = 0; train_correct_B = 0
        total_train_tokens = 0
        
        total_train_loss_A = 0.0
        total_train_loss_B = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            valid_mask = (y != tokenizer.pad_idx)
            
            # --- Train Model A ---
            opt_A.zero_grad()
            with autocast('cuda'):
                logits_A = model_A(x)
                loss_A = criterion(logits_A.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            
            # Implementazione corretta del Gradient Clipping con AMP
            scaler_A.scale(loss_A).backward()
            scaler_A.unscale_(opt_A)
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), max_norm=1.0)
            scaler_A.step(opt_A)
            scaler_A.update()
            
            preds_A = torch.argmax(logits_A, dim=-1)
            train_correct_A += (preds_A[valid_mask] == y[valid_mask]).sum().item()
            total_train_loss_A += loss_A.item()
            
            # --- Train Model B ---
            opt_B.zero_grad()
            with autocast('cuda'):
                logits_B = model_B(x)
                loss_B = criterion(logits_B.reshape(-1, VOCAB_SIZE), y.reshape(-1))
                
            scaler_B.scale(loss_B).backward()
            scaler_B.unscale_(opt_B)
            torch.nn.utils.clip_grad_norm_(model_B.parameters(), max_norm=1.0)
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
        
        total_val_loss_A = 0.0
        total_val_loss_B = 0.0
        
        with torch.no_grad():
            for x_val, y_val in ood_val_loader:
                x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                valid_mask_val = (y_val != tokenizer.pad_idx)
                
                with autocast('cuda'):
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
        
        avg_train_loss_A = total_train_loss_A / len(train_loader)
        avg_train_loss_B = total_train_loss_B / len(train_loader)
        avg_val_loss_A = total_val_loss_A / len(ood_val_loader)
        avg_val_loss_B = total_val_loss_B / len(ood_val_loader)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch+1:4d}/{EPOCHS}] [TUNING]")
            print(f"  Model A | Train Loss: {avg_train_loss_A:.4f} | Val Loss: {avg_val_loss_A:.4f} | Train Acc: {acc_train_A:6.2f}% | Val Acc: {acc_val_A:6.2f}%")
            print(f"  Model B | Train Loss: {avg_train_loss_B:.4f} | Val Loss: {avg_val_loss_B:.4f} | Train Acc: {acc_train_B:6.2f}% | Val Acc: {acc_val_B:6.2f}%")
            
            target_str = tokenizer.decode(y_val[0])
            pred_str_A = tokenizer.decode(preds_val_A[0])
            pred_str_B = tokenizer.decode(preds_val_B[0])
            
            print("  --- Sample Outputs ---")
            print(f"  Target: {target_str}")
            print(f"  Pred A: {pred_str_A}")
            print(f"  Pred B: {pred_str_B}")
            print("-" * 90)
        
        # Checkpoints per entrambi i modelli
        if epoch % 100 == 0:
            torch.save(model_A.state_dict(), f"modelA_phase5_epoch_{epoch}.pt")
            torch.save(model_B.state_dict(), f"modelB_phase5_epoch_{epoch}.pt")

if __name__ == "__main__":
    run_phase5_ab_test()