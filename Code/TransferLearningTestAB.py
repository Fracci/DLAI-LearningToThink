"""
Multi-seed A/B comparison: exact-match AND per-digit accuracy, in-distribution
(3-4 digit) AND OOD lengths.

For each seed: train Model A (pretrained init) and Model B (random init) on the
SAME 3-4 digit scratchpad data with the IDENTICAL schedule (only init differs).
Every eval reports, for each eval set:
    EM  = answer exact-match (all-or-nothing; the headline metric)
    PD  = per-digit accuracy of the answer (partial credit; the diagnostic)
Both are teacher-forced.

Per seed -> late-window mean per metric. Across seeds -> mean +/- std and the
paired gap (A - B). At the end of each seed, a right-aligned POSITIONAL answer
accuracy (least-significant digit = position 0) is saved for the final models,
to show where along the answer each model degrades.
"""
import random
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from Transformer import GeneralTransformer
from ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset

# CONFIG
SEEDS        = [3, 4]
EPOCHS       = 300
EVAL_EVERY   = 5
LATE_FRAC    = 0.5           
OOD_DIGITS   = [5, 6, 7]     
MAX_POS      = 12            

D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
BATCH_SIZE   = 256
MAX_SEQ_LEN  = 128
OOD_MAX_SEQ_LEN = 160
LR, WEIGHT_DECAY, GRAD_CLIP = 5e-4, 0.1, 1.0

PRETRAINED   = "rule30_rollout_pretrained.pt"
VAL_SEED     = 20240601       
N_ID_VAL     = 2000
N_OOD_VAL    = 3000           
SAVE_CHECKPOINTS = True


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_loss_targets(x, y, eq_idx, pad_idx):
    L = x.size(1)
    pos = torch.arange(L, device=x.device).unsqueeze(0)
    eq_col = (x == eq_idx).long().argmax(dim=1, keepdim=True)
    return torch.where(pos >= eq_col, y, torch.full_like(y, pad_idx))


def answer_region(y, a_idx, pad_idx):
    L = y.size(1)
    pos = torch.arange(L, device=y.device).unsqueeze(0)
    a_col = (y == a_idx).long().argmax(dim=1, keepdim=True)
    return (pos >= (a_col + 2)) & (y != pad_idx)


def materialize_loader(tok, min_d, max_d, n, max_seq_len, seed, batch):
    random.seed(seed)
    ds = ScratchpadAdditionDataset(num_samples=n, min_digits=min_d, max_digits=max_d,
                                   tokenizer=tok, max_seq_len=max_seq_len)
    xs, ys = zip(*(ds[i] for i in range(n)))
    return DataLoader(TensorDataset(torch.stack(xs), torch.stack(ys)), batch_size=batch)


@torch.no_grad()
def eval_metrics(model, loader, a_idx, pad_idx, device):
    model.eval()
    em_correct = em_total = 0
    dig_correct = dig_total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast("cuda"):
            preds = torch.argmax(model(x), dim=-1)
        ans = answer_region(y, a_idx, pad_idx)
        match = (preds == y)
        # exact-match per row (all answer digits correct)
        row_ok = (match | ~ans).all(dim=1) & ans.any(dim=1)
        em_correct += row_ok.sum().item(); em_total += ans.any(dim=1).sum().item()
        # per-digit (partial credit) over answer span
        dig_correct += (match & ans).sum().item(); dig_total += ans.sum().item()
    em = 100.0 * em_correct / em_total
    pd = 100.0 * dig_correct / dig_total
    return em, pd


@torch.no_grad()
def positional_accuracy(model, loader, a_idx, pad_idx, device, max_pos=MAX_POS):
    model.eval()
    correct = torch.zeros(max_pos, dtype=torch.long)
    total = torch.zeros(max_pos, dtype=torch.long)
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast("cuda"):
            preds = torch.argmax(model(x), dim=-1)
        ans = answer_region(y, a_idx, pad_idx)
        match = (preds == y)
        L = y.size(1)
        col = torch.arange(L, device=y.device).unsqueeze(0).expand_as(y)
        # last answer column per row -> distance from the right
        last_col = torch.where(ans, col, torch.full_like(col, -1)).max(dim=1, keepdim=True).values
        r = (last_col - col)                              # 0 at LSB, grows left
        sel = ans & (r >= 0) & (r < max_pos)
        rr = r[sel]; mm = match[sel]
        total += torch.bincount(rr.cpu(), minlength=max_pos)
        correct += torch.bincount(rr[mm].cpu(), minlength=max_pos)
    acc = [(100.0 * correct[i].item() / total[i].item()) if total[i] > 0 else float("nan")
           for i in range(max_pos)]
    return acc


def build_A(vocab, device):
    m = GeneralTransformer(vocab, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF).to(device)
    sd = torch.load(PRETRAINED, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items()
          if not k.startswith("embedding.") and not k.startswith("fc_out.")}
    m.load_state_dict(sd, strict=False)
    return m


def build_B(vocab, device):
    return GeneralTransformer(vocab, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF).to(device)


def train_one_seed(seed, eval_loaders, tok, device, pos_writer):
    set_seed(seed)
    PAD, EQ, A_IDX = tok.pad_idx, tok.char_to_idx["="], tok.char_to_idx["A"]
    labels = list(eval_loaders.keys())

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

    # history[label] = {A_em, A_pd, B_em, B_pd}
    history = {lab: {"A_em": [], "A_pd": [], "B_em": [], "B_pd": []} for lab in labels}

    # per-seed CSV log
    seed_log = open(f"seed{seed}_log.csv", "w", newline="")
    slog = csv.writer(seed_log)
    header = ["epoch", "loss_A", "loss_B"]
    for lab in labels:
        header += [f"{lab}_A_em", f"{lab}_B_em", f"{lab}_A_pd", f"{lab}_B_pd"]
    slog.writerow(header); seed_log.flush()

    for epoch in range(EPOCHS):
        model_A.train(); model_B.train()
        loss_sum_A = loss_sum_B = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            yl = build_loss_targets(x, y, EQ, PAD)

            opt_A.zero_grad()
            with autocast("cuda"):
                lA = crit(model_A(x).reshape(-1, tok.vocab_size), yl.reshape(-1))
            sc_A.scale(lA).backward(); sc_A.unscale_(opt_A)
            nn.utils.clip_grad_norm_(model_A.parameters(), GRAD_CLIP)
            sc_A.step(opt_A); sc_A.update()
            loss_sum_A += lA.item()

            opt_B.zero_grad()
            with autocast("cuda"):
                lB = crit(model_B(x).reshape(-1, tok.vocab_size), yl.reshape(-1))
            sc_B.scale(lB).backward(); sc_B.unscale_(opt_B)
            nn.utils.clip_grad_norm_(model_B.parameters(), GRAD_CLIP)
            sc_B.step(opt_B); sc_B.update()
            loss_sum_B += lB.item()

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            avg_A = loss_sum_A / len(train_loader)
            avg_B = loss_sum_B / len(train_loader)
            print(f"  seed {seed} ep {epoch+1:4d}/{EPOCHS} | loss A {avg_A:.4f} B {avg_B:.4f}")
            row = [epoch + 1, f"{avg_A:.4f}", f"{avg_B:.4f}"]
            for lab in labels:
                a_em, a_pd = eval_metrics(model_A, eval_loaders[lab], A_IDX, PAD, device)
                b_em, b_pd = eval_metrics(model_B, eval_loaders[lab], A_IDX, PAD, device)
                history[lab]["A_em"].append(a_em); history[lab]["A_pd"].append(a_pd)
                history[lab]["B_em"].append(b_em); history[lab]["B_pd"].append(b_pd)
                print(f"     {lab:5s} | EM  A {a_em:5.1f} B {b_em:5.1f} "
                      f"| PD  A {a_pd:5.1f} B {b_pd:5.1f}")
                row += [f"{a_em:.2f}", f"{b_em:.2f}", f"{a_pd:.2f}", f"{b_pd:.2f}"]
            slog.writerow(row); seed_log.flush()

    seed_log.close()

    # end-of-seed positional breakdown (final models)
    for lab in labels:
        pa_A = positional_accuracy(model_A, eval_loaders[lab], A_IDX, PAD, device)
        pa_B = positional_accuracy(model_B, eval_loaders[lab], A_IDX, PAD, device)
        for p in range(MAX_POS):
            pos_writer.writerow([seed, lab, "A", p, f"{pa_A[p]:.2f}"])
            pos_writer.writerow([seed, lab, "B", p, f"{pa_B[p]:.2f}"])

    if SAVE_CHECKPOINTS:
        torch.save(model_A.state_dict(), f"seed{seed}_modelA.pt")
        torch.save(model_B.state_dict(), f"seed{seed}_modelB.pt")

    # late-window means per label and metric
    scores = {}
    for lab in labels:
        k = max(1, int(len(history[lab]["A_em"]) * LATE_FRAC))
        scores[lab] = {m: sum(history[lab][m][-k:]) / k for m in
                       ("A_em", "B_em", "A_pd", "B_pd")}
    return scores


def mean_std(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, v ** 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CharTokenizer()
    print(f"Multi-seed A/B sweep (EM + per-digit, ID + OOD) on {device} "
          f"| seeds={SEEDS} | epochs={EPOCHS}")

    # fixed eval sets: in-distribution (3-4) + each OOD length
    eval_loaders = {"id": materialize_loader(tok, 3, 4, N_ID_VAL, MAX_SEQ_LEN,
                                             seed=VAL_SEED, batch=BATCH_SIZE)}
    for d in OOD_DIGITS:
        eval_loaders[f"{d}dig"] = materialize_loader(tok, d, d, N_OOD_VAL, OOD_MAX_SEQ_LEN,
                                                     seed=VAL_SEED + d, batch=BATCH_SIZE)
    labels = list(eval_loaders.keys())

    pos_file = open("positional_accuracy.csv", "w", newline="")
    pos_writer = csv.writer(pos_file)
    pos_writer.writerow(["seed", "eval_set", "model", "pos_from_LSB", "accuracy"])

    per_seed = {}
    for s in SEEDS:
        print(f"\n========== SEED {s} ==========")
        per_seed[s] = train_one_seed(s, eval_loaders, tok, device, pos_writer)
        pos_file.flush()
    pos_file.close()

    # aggregate
    print("AGGREGATE (mean +/- std across seeds)")
    rows = [["eval_set", "metric", "A_mean", "A_std", "B_mean", "B_std", "gap_mean", "gap_std"]]
    for lab in labels:
        for metric, ak, bk in (("EM", "A_em", "B_em"), ("PD", "A_pd", "B_pd")):
            A = [per_seed[s][lab][ak] for s in SEEDS]
            B = [per_seed[s][lab][bk] for s in SEEDS]
            gaps = [a - b for a, b in zip(A, B)]
            Am, As = mean_std(A); Bm, Bs = mean_std(B); Gm, Gs = mean_std(gaps)
            print(f"  {lab:5s} {metric} | A {Am:6.2f} +/- {As:4.2f} | B {Bm:6.2f} +/- {Bs:4.2f} "
                  f"| gap {Gm:+6.2f} +/- {Gs:4.2f}")
            rows.append([lab, metric, f"{Am:.2f}", f"{As:.2f}", f"{Bm:.2f}", f"{Bs:.2f}",
                         f"{Gm:.2f}", f"{Gs:.2f}"])
        print()

    with open("seed_sweep_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("saved -> seed_sweep_summary.csv, positional_accuracy.csv, seed{N}_log.csv")


if __name__ == "__main__":
    main()