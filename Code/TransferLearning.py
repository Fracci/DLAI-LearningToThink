import random
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from Transformer import Rule30Transformer
from ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset


# ===================================================================
# Measurement helpers
# ===================================================================
def build_loss_targets(x, y, eq_idx, pad_idx):
    """Prompt masking: keep loss only on the scratchpad+answer region."""
    L = x.size(1)
    positions = torch.arange(L, device=x.device).unsqueeze(0)
    eq_col = (x == eq_idx).long().argmax(dim=1, keepdim=True)
    target_region = positions >= eq_col
    return torch.where(target_region, y, torch.full_like(y, pad_idx))


def answer_exact_match(preds, y, a_idx, pad_idx):
    """Exact match on the final answer digits only (after 'A:'). Teacher-forced.
    Returns (num_correct_rows, num_rows_with_answer)."""
    L = y.size(1)
    positions = torch.arange(L, device=y.device).unsqueeze(0)
    a_col = (y == a_idx).long().argmax(dim=1, keepdim=True)
    ans_region = (positions >= (a_col + 2)) & (y != pad_idx)
    ok = (preds == y) | (~ans_region)
    row_correct = ok.all(dim=1) & ans_region.any(dim=1)
    return row_correct.sum().item(), ans_region.any(dim=1).sum().item()


def materialize_loader(tokenizer, min_d, max_d, n, max_seq_len, seed, batch_size):
    """Build a FIXED, reproducible eval set once (cached tensors)."""
    random.seed(seed)
    ds = ScratchpadAdditionDataset(
        num_samples=n, min_digits=min_d, max_digits=max_d,
        tokenizer=tokenizer, max_seq_len=max_seq_len
    )
    xs, ys = [], []
    for i in range(n):
        x, y = ds[i]
        xs.append(x); ys.append(y)
    tds = TensorDataset(torch.stack(xs), torch.stack(ys))
    return DataLoader(tds, batch_size=batch_size)


@torch.no_grad()
def evaluate_em(model, loader, a_idx, pad_idx, device):
    model.eval()
    correct = total = 0
    last = None
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast("cuda"):
            logits = model(x)
        preds = torch.argmax(logits, dim=-1)
        c, n = answer_exact_match(preds, y, a_idx, pad_idx)
        correct += c; total += n
        last = (y, preds)
    return (100.0 * correct / total), last


def run_phase5_ab_test():
    # ---------------------------------------------------------
    # 1. Hyperparameters
    # ---------------------------------------------------------
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD = 256, 8, 6, 1024
    BATCH_SIZE = 256
    EPOCHS = 3000
    MAX_SEQ_LEN = 128                 # training (3-4 digit)
    OOD_MAX_SEQ_LEN = 160             # room for up to 7-digit (~98 tokens)

    LR, WEIGHT_DECAY, GRAD_CLIP = 5e-4, 0.1, 1.0
    VAL_SEED = 20240601

    OOD_DIGITS = [5, 6, 7]            # <<< OOD lengths tracked every eval
    N_ID_VAL = 2000
    N_OOD_VAL = 2000                  # per length
    EVAL_EVERY = 5
    SAMPLE_DIGITS = OOD_DIGITS[-1]    # which length to print a sample prediction from
    LOG_PATH = "training_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 5 A/B Test on: {device}")

    tokenizer = CharTokenizer()
    VOCAB_SIZE = tokenizer.vocab_size
    PAD_IDX = tokenizer.pad_idx
    EQ_IDX = tokenizer.char_to_idx["="]
    A_IDX = tokenizer.char_to_idx["A"]

    # ---------------------------------------------------------
    # 2. Data: on-the-fly train + FIXED in-dist / multi-length OOD val
    # ---------------------------------------------------------
    train_dataset = ScratchpadAdditionDataset(
        num_samples=15000, min_digits=3, max_digits=4,
        tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    id_val_loader = materialize_loader(tokenizer, 3, 4, N_ID_VAL, MAX_SEQ_LEN,
                                       seed=VAL_SEED, batch_size=BATCH_SIZE)
    ood_loaders = {
        d: materialize_loader(tokenizer, d, d, N_OOD_VAL, OOD_MAX_SEQ_LEN,
                              seed=VAL_SEED + d, batch_size=BATCH_SIZE)
        for d in OOD_DIGITS
    }
    print(f"Fixed val sets: in-dist (3-4 dig, {N_ID_VAL}); "
          f"OOD {OOD_DIGITS} ({N_OOD_VAL} each).")

    # ---------------------------------------------------------
    # 3. Model A: pretrained core, fresh head, no freeze
    # ---------------------------------------------------------
    model_A = Rule30Transformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD).to(device)
    pre = torch.load("rule30_pretrained_new.pt", map_location=device)
    pre = {k.replace("module.", ""): v for k, v in pre.items()}
    pre = {k: v for k, v in pre.items()
           if not k.startswith("embedding.") and not k.startswith("fc_out.")}
    model_A.load_state_dict(pre, strict=False)
    print("Model A: pretrained Rule30 core loaded (end-to-end, no freeze).")

    # ---------------------------------------------------------
    # 4. Model B: random baseline
    # ---------------------------------------------------------
    model_B = Rule30Transformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD).to(device)
    print("Model B: random init.")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel...")
        model_A = nn.DataParallel(model_A)
        model_B = nn.DataParallel(model_B)

    # ---------------------------------------------------------
    # 5. Identical optimization
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    opt_A = AdamW(model_A.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_B = AdamW(model_B.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler_A = GradScaler("cuda")
    scaler_B = GradScaler("cuda")

    # CSV log header
    log_file = open(LOG_PATH, "w", newline="")
    logger = csv.writer(log_file)
    header = ["epoch", "loss_A", "loss_B", "id_A", "id_B"]
    for d in OOD_DIGITS:
        header += [f"ood{d}_A", f"ood{d}_B"]
    logger.writerow(header)
    log_file.flush()

    # ---------------------------------------------------------
    # 6. Training loop
    # ---------------------------------------------------------
    print("\nStarting A/B Training Loop...")
    for epoch in range(EPOCHS):
        model_A.train(); model_B.train()
        loss_sum_A = loss_sum_B = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_loss = build_loss_targets(x, y, EQ_IDX, PAD_IDX)

            opt_A.zero_grad()
            with autocast("cuda"):
                loss_A = criterion(model_A(x).reshape(-1, VOCAB_SIZE), y_loss.reshape(-1))
            scaler_A.scale(loss_A).backward()
            scaler_A.unscale_(opt_A)
            nn.utils.clip_grad_norm_(model_A.parameters(), GRAD_CLIP)
            scaler_A.step(opt_A); scaler_A.update()
            loss_sum_A += loss_A.item()

            opt_B.zero_grad()
            with autocast("cuda"):
                loss_B = criterion(model_B(x).reshape(-1, VOCAB_SIZE), y_loss.reshape(-1))
            scaler_B.scale(loss_B).backward()
            scaler_B.unscale_(opt_B)
            nn.utils.clip_grad_norm_(model_B.parameters(), GRAD_CLIP)
            scaler_B.step(opt_B); scaler_B.update()
            loss_sum_B += loss_B.item()

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            avg_A = loss_sum_A / len(train_loader)
            avg_B = loss_sum_B / len(train_loader)
            id_A, _ = evaluate_em(model_A, id_val_loader, A_IDX, PAD_IDX, device)
            id_B, _ = evaluate_em(model_B, id_val_loader, A_IDX, PAD_IDX, device)

            ood_A, ood_B, sample = {}, {}, {}
            for d in OOD_DIGITS:
                a, la = evaluate_em(model_A, ood_loaders[d], A_IDX, PAD_IDX, device)
                b, lb = evaluate_em(model_B, ood_loaders[d], A_IDX, PAD_IDX, device)
                ood_A[d], ood_B[d] = a, b
                if d == SAMPLE_DIGITS:
                    sample = {"target": la[0][0], "predA": la[1][0], "predB": lb[1][0]}

            # console report
            print(f"Epoch [{epoch+1:4d}/{EPOCHS}]")
            print(f"  Loss          | A {avg_A:.4f} | B {avg_B:.4f}")
            print(f"  InDist (3-4)  | A {id_A:6.2f}% | B {id_B:6.2f}%")
            for d in OOD_DIGITS:
                print(f"  OOD {d}-dig     | A {ood_A[d]:6.2f}% | B {ood_B[d]:6.2f}%")
            print(f"  Target({SAMPLE_DIGITS}): {tokenizer.decode(sample['target'])}")
            print(f"  Pred A({SAMPLE_DIGITS}):  {tokenizer.decode(sample['predA'])}")
            print(f"  Pred B({SAMPLE_DIGITS}):  {tokenizer.decode(sample['predB'])}")
            print("-" * 90)

            # csv log
            row = [epoch + 1, f"{avg_A:.4f}", f"{avg_B:.4f}", f"{id_A:.2f}", f"{id_B:.2f}"]
            for d in OOD_DIGITS:
                row += [f"{ood_A[d]:.2f}", f"{ood_B[d]:.2f}"]
            logger.writerow(row)
            log_file.flush()

        if epoch > 0 and epoch % 100 == 0:
            torch.save(model_A.state_dict(), f"modelA_phase5_epoch_{epoch}.pt")
            torch.save(model_B.state_dict(), f"modelB_phase5_epoch_{epoch}.pt")

    torch.save(model_A.state_dict(), "modelA_phase5_final.pt")
    torch.save(model_B.state_dict(), "modelB_phase5_final.pt")
    log_file.close()
    print("Saved final checkpoints and training_log.csv")


if __name__ == "__main__":
    run_phase5_ab_test()