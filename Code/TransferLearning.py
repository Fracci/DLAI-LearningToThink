import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time

from Transformer import Rule30Transformer
from ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset


# ===================================================================
# Helpers for the corrected measurement
# ===================================================================
def build_loss_targets(x, y, eq_idx, pad_idx):
    """
    Prompt masking: keep loss ONLY on the scratchpad+answer region.
    Everything in y before the first target token (i.e. the input digits
    '456+129' and the '=') is set to pad_idx so CrossEntropyLoss ignores it.

    The first target token in y sits at the same column as '=' in x
    (because y is x shifted left by one), so we mask columns < that.
    """
    L = x.size(1)
    positions = torch.arange(L, device=x.device).unsqueeze(0)          # (1, L)
    eq_col = (x == eq_idx).long().argmax(dim=1, keepdim=True)          # (B, 1)
    target_region = positions >= eq_col                                # (B, L) bool
    return torch.where(target_region, y, torch.full_like(y, pad_idx))


def answer_exact_match(preds, y, a_idx, pad_idx):
    """
    Exact-match accuracy on the final answer digits only (after 'A:').
    A row counts as correct iff EVERY answer digit is predicted correctly.

    NOTE: this is teacher-forced (we feed ground-truth x), so it is an
    optimistic upper bound vs. free-running generation. It is the standard
    cheap grokking metric and is enough to break the A==B tie.
    Returns (num_correct_rows, num_rows_with_an_answer).
    """
    L = y.size(1)
    positions = torch.arange(L, device=y.device).unsqueeze(0)
    a_col = (y == a_idx).long().argmax(dim=1, keepdim=True)            # col of 'A'
    ans_region = (positions >= (a_col + 2)) & (y != pad_idx)           # digits after 'A:'
    ok = (preds == y) | (~ans_region)                                  # correct or outside region
    row_correct = ok.all(dim=1) & ans_region.any(dim=1)
    return row_correct.sum().item(), ans_region.any(dim=1).sum().item()


def run_phase5_ab_test():
    # ---------------------------------------------------------
    # 1. Setup & Hyperparameters
    # ---------------------------------------------------------
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1024

    BATCH_SIZE = 256
    EPOCHS = 3000
    MAX_SEQ_LEN = 128

    # >>> ONE identical recipe for BOTH models. Only difference = init. <<<
    LR = 5e-4
    WEIGHT_DECAY = 0.1
    GRAD_CLIP = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 5 A/B Test (fair schedule) on: {device}")

    tokenizer = CharTokenizer()
    VOCAB_SIZE = tokenizer.vocab_size
    PAD_IDX = tokenizer.pad_idx
    EQ_IDX = tokenizer.char_to_idx["="]
    A_IDX = tokenizer.char_to_idx["A"]

    # ---------------------------------------------------------
    # 2. Data: In-Distribution (3-4 digit) train, OOD (5 digit) val
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
    # 3. Model A: pretrained transformer, fresh embedding/head, NO FREEZE
    # ---------------------------------------------------------
    model_A = Rule30Transformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)

    pretrained_dict = torch.load("rule30_pretrained_new.pt", map_location=device)
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if not k.startswith("embedding.") and not k.startswith("fc_out.")}
    model_A.load_state_dict(filtered_dict, strict=False)
    print("Model A: Loaded pretrained Rule30 weights (transformer + final_norm). Trained end-to-end, no freeze.")

    # ---------------------------------------------------------
    # 4. Model B: identical architecture, random init
    # ---------------------------------------------------------
    model_B = Rule30Transformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)
    print("Model B: Random init.")

    if torch.cuda.device_count() > 1:
        print(f"Optimizing for {torch.cuda.device_count()} GPUs via DataParallel...")
        model_A = nn.DataParallel(model_A)
        model_B = nn.DataParallel(model_B)

    # ---------------------------------------------------------
    # 5. IDENTICAL optimization for A and B
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    opt_A = AdamW(model_A.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_B = AdamW(model_B.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler_A = GradScaler("cuda")
    scaler_B = GradScaler("cuda")

    # ---------------------------------------------------------
    # 6. Training loop
    # ---------------------------------------------------------
    print("\nStarting A/B Training Loop...")
    for epoch in range(EPOCHS):
        model_A.train(); model_B.train()

        # in-distribution (train) exact-match trackers
        id_correct_A = id_correct_B = id_total = 0
        loss_sum_A = loss_sum_B = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_loss = build_loss_targets(x, y, EQ_IDX, PAD_IDX)   # prompt-masked targets

            # --- Model A ---
            opt_A.zero_grad()
            with autocast("cuda"):
                logits_A = model_A(x)
                loss_A = criterion(logits_A.reshape(-1, VOCAB_SIZE), y_loss.reshape(-1))
            scaler_A.scale(loss_A).backward()
            scaler_A.unscale_(opt_A)
            nn.utils.clip_grad_norm_(model_A.parameters(), GRAD_CLIP)
            scaler_A.step(opt_A); scaler_A.update()

            # --- Model B ---
            opt_B.zero_grad()
            with autocast("cuda"):
                logits_B = model_B(x)
                loss_B = criterion(logits_B.reshape(-1, VOCAB_SIZE), y_loss.reshape(-1))
            scaler_B.scale(loss_B).backward()
            scaler_B.unscale_(opt_B)
            nn.utils.clip_grad_norm_(model_B.parameters(), GRAD_CLIP)
            scaler_B.step(opt_B); scaler_B.update()

            # in-distribution exact match (teacher-forced)
            preds_A = torch.argmax(logits_A, dim=-1)
            preds_B = torch.argmax(logits_B, dim=-1)
            cA, n = answer_exact_match(preds_A, y, A_IDX, PAD_IDX)
            cB, _ = answer_exact_match(preds_B, y, A_IDX, PAD_IDX)
            id_correct_A += cA; id_correct_B += cB; id_total += n
            loss_sum_A += loss_A.item(); loss_sum_B += loss_B.item()

        # --- OOD validation (5-digit) ---
        model_A.eval(); model_B.eval()
        ood_correct_A = ood_correct_B = ood_total = 0
        with torch.no_grad():
            for x_val, y_val in ood_val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                with autocast("cuda"):
                    logits_val_A = model_A(x_val)
                    logits_val_B = model_B(x_val)
                preds_val_A = torch.argmax(logits_val_A, dim=-1)
                preds_val_B = torch.argmax(logits_val_B, dim=-1)
                cA, n = answer_exact_match(preds_val_A, y_val, A_IDX, PAD_IDX)
                cB, _ = answer_exact_match(preds_val_B, y_val, A_IDX, PAD_IDX)
                ood_correct_A += cA; ood_correct_B += cB; ood_total += n

        # --- Reporting (exact-match is the real signal) ---
        id_em_A = 100 * id_correct_A / id_total
        id_em_B = 100 * id_correct_B / id_total
        ood_em_A = 100 * ood_correct_A / ood_total
        ood_em_B = 100 * ood_correct_B / ood_total

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch+1:4d}/{EPOCHS}]")
            print(f"  Model A | Loss {loss_sum_A/len(train_loader):.4f} "
                  f"| InDist EM {id_em_A:6.2f}% | OOD(5dig) EM {ood_em_A:6.2f}%")
            print(f"  Model B | Loss {loss_sum_B/len(train_loader):.4f} "
                  f"| InDist EM {id_em_B:6.2f}% | OOD(5dig) EM {ood_em_B:6.2f}%")
            print(f"  Target: {tokenizer.decode(y_val[0])}")
            print(f"  Pred A: {tokenizer.decode(preds_val_A[0])}")
            print(f"  Pred B: {tokenizer.decode(preds_val_B[0])}")
            print("-" * 90)

        # Checkpoints for BOTH (needed for the attention-map comparison)
        if epoch > 0 and epoch % 100 == 0:
            torch.save(model_A.state_dict(), f"modelA_phase5_epoch_{epoch}.pt")
            torch.save(model_B.state_dict(), f"modelB_phase5_epoch_{epoch}.pt")


if __name__ == "__main__":
    run_phase5_ab_test()