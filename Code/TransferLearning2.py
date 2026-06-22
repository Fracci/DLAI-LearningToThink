import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from Transformer import Rule30Transformer
from ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset

def run_phase5_ab_test():
    # 1. Setup
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD = 256, 8, 6, 1024
    BATCH_SIZE = 256
    EPOCHS = 3000        
    MAX_SEQ_LEN = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    VOCAB_SIZE = tokenizer.vocab_size
    EQUAL_ID = tokenizer.encode('=')[0] # ID del segno =

    # 2. Data
    train_dataset = ScratchpadAdditionDataset(num_samples=15000, min_digits=3, max_digits=4, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    ood_val_dataset = ScratchpadAdditionDataset(num_samples=1000, min_digits=5, max_digits=5, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN + 32)
    ood_val_loader = DataLoader(ood_val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    # 3. Modelli
    model_A = Rule30Transformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD).to(device)
    model_B = Rule30Transformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    opt_A = AdamW(model_A.parameters(), lr=5e-5, weight_decay=0.2) 
    opt_B = AdamW(model_B.parameters(), lr=5e-4, weight_decay=0.1)
    
    scaler_A, scaler_B = GradScaler('cuda'), GradScaler('cuda')

    print("\nStarting Training Loop with Exact Accuracy & Prompt Masking...")
    
    for epoch in range(EPOCHS):
        model_A.train(); model_B.train()
        
        # Monitoraggio completo
        metrics = {"A_loss": 0.0, "B_loss": 0.0, "A_acc": 0, "B_acc": 0, "A_seq_acc": 0, "B_seq_acc": 0, "total": 0}
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # MASKING: solo ciò che viene dopo '='
            prompt_mask = (x == EQUAL_ID).cumsum(dim=1) > 0
            y_masked = y.clone()
            y_masked[~prompt_mask] = tokenizer.pad_idx
            valid_mask = (y_masked != tokenizer.pad_idx)
            
            # Forward & Backward (A)
            opt_A.zero_grad()
            with autocast('cuda'):
                logits_A = model_A(x)
                loss_A = criterion(logits_A.reshape(-1, VOCAB_SIZE), y_masked.reshape(-1))
            scaler_A.scale(loss_A).backward()
            scaler_A.unscale_(opt_A)
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            scaler_A.step(opt_A); scaler_A.update()
            
            # Forward & Backward (B)
            opt_B.zero_grad()
            with autocast('cuda'):
                logits_B = model_B(x)
                loss_B = criterion(logits_B.reshape(-1, VOCAB_SIZE), y_masked.reshape(-1))
            scaler_B.scale(loss_B).backward()
            scaler_B.unscale_(opt_B)
            torch.nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
            scaler_B.step(opt_B); scaler_B.update()

        # VALIDAZIONE
        model_A.eval(); model_B.eval()
        with torch.no_grad():
            for x_val, y_val in ood_val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                prompt_mask = (x_val == EQUAL_ID).cumsum(dim=1) > 0
                y_m = y_val.clone(); y_m[~prompt_mask] = tokenizer.pad_idx
                
                l_A = model_A(x_val); p_A = torch.argmax(l_A, dim=-1)
                l_B = model_B(x_val); p_B = torch.argmax(l_B, dim=-1)
                
                # Accuracy Totale (Sequence Accuracy): confronta l'intera riga dopo '='
                # Verifichiamo dove la maschera è attiva
                seq_A = ((p_A == y_val) * prompt_mask).sum(dim=1) == prompt_mask.sum(dim=1)
                seq_B = ((p_B == y_val) * prompt_mask).sum(dim=1) == prompt_mask.sum(dim=1)
                
                metrics["A_seq_acc"] += seq_A.sum().item()
                metrics["B_seq_acc"] += seq_B.sum().item()
                metrics["total"] += x_val.size(0)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: A_SeqAcc: {metrics['A_seq_acc']/metrics['total']:.2%} | B_SeqAcc: {metrics['B_seq_acc']/metrics['total']:.2%}")

if __name__ == "__main__":
    run_phase5_ab_test()