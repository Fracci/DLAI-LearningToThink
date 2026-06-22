"""
Multi-seed A/B comparison at 6-digit addition.

For each seed: train Model A (pretrained init) and Model B (random init) on the
SAME 3-4 digit scratchpad data with the IDENTICAL schedule (only init differs),
and evaluate exact-match on a FIXED 6-digit OOD test set (same across all seeds).

Per seed, the score is the 6-digit EM averaged over a late training window (to
cut per-epoch noise). Across seeds we report mean +/- std for A and B, and the
paired gap (A - B) per seed -- the headline that turns n=1 into a real claim.
"""
import random
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from Transformer import Rule30Transformer
from ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset

# ===================== CONFIG =====================
SEEDS        = [0, 1, 2]
EPOCHS       = 300
EVAL_EVERY   = 5
LATE_FRAC    = 0.5            # average EM over the last 50% of eval points per seed
OOD_DIGITS   = [6]           # the regime where the gap lives (add 5,7 if you want context)

D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
BATCH_SIZE   = 256
MAX_SEQ_LEN  = 128
OOD_MAX_SEQ_LEN = 160
LR, WEIGHT_DECAY, GRAD_CLIP = 5e-4, 0.1, 1.0

PRETRAINED   = "rule30_pretrained_new.pt"
VAL_SEED     = 20240601       # fixed -> identical OOD test set across all seeds
N_OOD_VAL    = 3000           # per length
SAVE_CHECKPOINTS = True
# ==================================================


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_loss_targets(x, y, eq_idx, pad_idx):
    L = x.size(1)
    pos = torch.arange(L, device=x.device).unsqueeze(0)
    eq_col = (x == eq_idx).long().argmax(dim=1, keepdim=True)
    return torch.where(pos >= eq_col, y, torch.full_like(y, pad_idx))


def answer_exact_match(preds, y, a_idx, pad_idx):
    L = y.size(1)
    pos = torch.arange(L, device=y.device).unsqueeze(0)
    a_col = (y == a_idx).long().argmax(dim=1, keepdim=True)
    ans = (pos >= (a_col + 2)) & (y != pad_idx)
    ok = (preds == y) | (~ans)
    row_ok = ok.all(dim=1) & ans.any(dim=1)
    return row_ok.sum().item(), ans.any(dim=1).sum().item()


def materialize_loader(tok, d, n, max_seq_len, seed, batch):
    random.seed(seed)
    ds = ScratchpadAdditionDataset(num_samples=n, min_digits=d, max_digits=d,
                                   tokenizer=tok, max_seq_len=max_seq_len)
    xs, ys = zip(*(ds[i] for i in range(n)))
    return DataLoader(TensorDataset(torch.stack(xs), torch.stack(ys)), batch_size=batch)


@torch.no_grad()
def eval_em(model, loader, a_idx, pad_idx, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast("cuda"):
            logits = model(x)
        c, n = answer_exact_match(torch.argmax(logits, -1), y, a_idx, pad_idx)
        correct += c; total += n
    return 100.0 * correct / total


def build_A(vocab, device):
    m = Rule30Transformer(vocab, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF).to(device)
    sd = torch.load(PRETRAINED, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items()
          if not k.startswith("embedding.") and not k.startswith("fc_out.")}
    m.load_state_dict(sd, strict=False)
    return m


def build_B(vocab, device):
    return Rule30Transformer(vocab, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF).to(device)


def train_one_seed(seed, ood_loaders, tok, device):
    set_seed(seed)
    PAD, EQ, A_IDX = tok.pad_idx, tok.char_to_idx["="], tok.char_to_idx["A"]

    train_ds = ScratchpadAdditionDataset(num_samples=15000, min_digits=3, max_digits=4,
                                         tokenizer=tok, max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model_A = build_A(tok.vocab_size, device)
    model_B = build_B(tok.vocab_size, device)
    if torch.cuda.device_count() > 1:
        model_A = nn.DataParallel(model_A); model_B = nn.DataParallel(model_B)

    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    opt_A = AdamW(model_A.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_B = AdamW(model_B.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sc_A, sc_B = GradScaler("cuda"), GradScaler("cuda")

    history = {d: {"A": [], "B": []} for d in OOD_DIGITS}
    for epoch in range(EPOCHS):
        model_A.train(); model_B.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            yl = build_loss_targets(x, y, EQ, PAD)

            opt_A.zero_grad()
            with autocast("cuda"):
                lA = crit(model_A(x).reshape(-1, tok.vocab_size), yl.reshape(-1))
            sc_A.scale(lA).backward(); sc_A.unscale_(opt_A)
            nn.utils.clip_grad_norm_(model_A.parameters(), GRAD_CLIP)
            sc_A.step(opt_A); sc_A.update()

            opt_B.zero_grad()
            with autocast("cuda"):
                lB = crit(model_B(x).reshape(-1, tok.vocab_size), yl.reshape(-1))
            sc_B.scale(lB).backward(); sc_B.unscale_(opt_B)
            nn.utils.clip_grad_norm_(model_B.parameters(), GRAD_CLIP)
            sc_B.step(opt_B); sc_B.update()

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            line = f"  seed {seed} epoch {epoch+1:4d}/{EPOCHS}"
            for d in OOD_DIGITS:
                a = eval_em(model_A, ood_loaders[d], A_IDX, PAD, device)
                b = eval_em(model_B, ood_loaders[d], A_IDX, PAD, device)
                history[d]["A"].append(a); history[d]["B"].append(b)
                line += f" | {d}dig A {a:5.1f}% B {b:5.1f}%"
            print(line)

    if SAVE_CHECKPOINTS:
        torch.save(model_A.state_dict(), f"seed{seed}_modelA.pt")
        torch.save(model_B.state_dict(), f"seed{seed}_modelB.pt")

    # late-window mean per length
    scores = {}
    for d in OOD_DIGITS:
        k = max(1, int(len(history[d]["A"]) * LATE_FRAC))
        scores[d] = {
            "A": sum(history[d]["A"][-k:]) / k,
            "B": sum(history[d]["B"][-k:]) / k,
        }
    return scores, history


def mean_std(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, v ** 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CharTokenizer()
    print(f"Multi-seed 6-digit A/B sweep on {device} | seeds={SEEDS} | epochs={EPOCHS}")

    # fixed OOD test sets (identical across all seeds)
    ood_loaders = {d: materialize_loader(tok, d, N_OOD_VAL, OOD_MAX_SEQ_LEN,
                                          seed=VAL_SEED + d, batch=BATCH_SIZE)
                   for d in OOD_DIGITS}

    per_seed = {}
    for s in SEEDS:
        print(f"\n========== SEED {s} ==========")
        scores, _ = train_one_seed(s, ood_loaders, tok, device)
        per_seed[s] = scores
        for d in OOD_DIGITS:
            print(f"  seed {s} late-window {d}dig: A {scores[d]['A']:.2f}%  "
                  f"B {scores[d]['B']:.2f}%  gap {scores[d]['A']-scores[d]['B']:+.2f}")

    print("\n" + "=" * 60)
    print("AGGREGATE (mean +/- std across seeds)")
    rows = [["digits", "A_mean", "A_std", "B_mean", "B_std", "gap_mean", "gap_std"]]
    for d in OOD_DIGITS:
        A = [per_seed[s][d]["A"] for s in SEEDS]
        B = [per_seed[s][d]["B"] for s in SEEDS]
        gaps = [a - b for a, b in zip(A, B)]
        Am, As = mean_std(A); Bm, Bs = mean_std(B); Gm, Gs = mean_std(gaps)
        print(f"  {d}-digit | A {Am:6.2f} +/- {As:4.2f} | B {Bm:6.2f} +/- {Bs:4.2f} "
              f"| gap {Gm:+6.2f} +/- {Gs:4.2f}")
        if Gs > 1e-9 and len(SEEDS) > 1:
            t = Gm / (Gs / (len(SEEDS) ** 0.5))
            print(f"            paired gap t-stat ~ {t:+.2f} (n={len(SEEDS)}; small-n, interpret loosely)")
        rows.append([d, f"{Am:.2f}", f"{As:.2f}", f"{Bm:.2f}", f"{Bs:.2f}", f"{Gm:.2f}", f"{Gs:.2f}"])
    print("=" * 60)

    with open("seed_sweep_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("saved -> seed_sweep_summary.csv")


if __name__ == "__main__":
    main()