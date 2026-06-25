"""
A-ONLY multi-seed sweep (reuses a previously-trained Model B).

Model B (random init) is identical across pretraining arms -- it never touches
the pretrained weights -- so it only needs to be trained once. This script trains
and evaluates ONLY Model A, while still CONSTRUCTING (but not training) Model B so
the RNG stream, A's initialization, and the training-data order are byte-identical
to the full A/B run. Verified by matching first-epoch A metrics.

B's numbers are read back from a previously saved per-seed log (seed{N}_log.csv
from the arm where B was trained) to compute the paired gap. Point B_LOG_DIR at
that arm's logs. If you only want A's curves, set USE_STORED_B = False.
"""
import random
import csv
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from Transformer import Rule30Transformer
from ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset

# ===================== CONFIG =====================
SEEDS        = [0, 1, 2, 3, 4]
EPOCHS       = 1
EVAL_EVERY   = 5
LATE_FRAC    = 0.5
OOD_DIGITS   = [5, 6, 7]
MAX_POS      = 12

D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
BATCH_SIZE   = 256
MAX_SEQ_LEN  = 128
OOD_MAX_SEQ_LEN = 160
LR, WEIGHT_DECAY, GRAD_CLIP = 5e-4, 0.1, 1.0

PRETRAINED   = "rule30_pretrained_new.pt"
VAL_SEED     = 20240601
N_ID_VAL     = 2000
N_OOD_VAL    = 3000
SAVE_CHECKPOINTS = True

# --- reusing a previously-trained B ---
USE_STORED_B = True
# Folder + filename pattern of the FULL A/B run whose B you are reusing.
# Those logs are named "seed{N}_log.csv" and DO contain {lab}_B_em / {lab}_B_pd
# columns. This must NOT point at this script's own A-only logs (which have no B).
B_LOG_DIR    = "b_logs"        # e.g. the folder with the Rule30/rollout seed{N}_log.csv
B_LOG_PATTERN= "seed{seed}_log.csv"
B_SEEDS_AVAIL= [0, 1, 2, 3, 4]       # seeds for which a trained-B log actually exists
OUT_TAG      = "Rule30"         # prefix for THIS arm's (A-only) output files
# ==================================================


def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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
    em_c = em_t = dig_c = dig_t = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast("cuda"):
            preds = torch.argmax(model(x), dim=-1)
        ans = answer_region(y, a_idx, pad_idx); match = (preds == y)
        row_ok = (match | ~ans).all(dim=1) & ans.any(dim=1)
        em_c += row_ok.sum().item(); em_t += ans.any(dim=1).sum().item()
        dig_c += (match & ans).sum().item(); dig_t += ans.sum().item()
    return 100.0 * em_c / em_t, 100.0 * dig_c / dig_t


@torch.no_grad()
def positional_accuracy(model, loader, a_idx, pad_idx, device, max_pos=MAX_POS):
    model.eval()
    correct = torch.zeros(max_pos, dtype=torch.long)
    total = torch.zeros(max_pos, dtype=torch.long)
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast("cuda"):
            preds = torch.argmax(model(x), dim=-1)
        ans = answer_region(y, a_idx, pad_idx); match = (preds == y)
        L = y.size(1)
        col = torch.arange(L, device=y.device).unsqueeze(0).expand_as(y)
        last_col = torch.where(ans, col, torch.full_like(col, -1)).max(dim=1, keepdim=True).values
        r = (last_col - col)
        sel = ans & (r >= 0) & (r < max_pos)
        rr = r[sel]; mm = match[sel]
        total += torch.bincount(rr.cpu(), minlength=max_pos)
        correct += torch.bincount(rr[mm].cpu(), minlength=max_pos)
    return [(100.0 * correct[i].item() / total[i].item()) if total[i] > 0 else float("nan")
            for i in range(max_pos)]


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


def load_stored_B(seed, labels):
    """Read B's per-seed late-window means from a FULL A/B run's seed{N}_log.csv.
    Returns None (with a warning) if no B log exists for this seed, so seeds
    without a trained B simply report A-only."""
    if seed not in B_SEEDS_AVAIL:
        print(f"    [B] no stored B for seed {seed} (available: {B_SEEDS_AVAIL}); "
              f"reporting A-only for this seed.")
        return None
    path = os.path.join(B_LOG_DIR, B_LOG_PATTERN.format(seed=seed))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"B log not found: {path}. Point B_LOG_DIR at the folder holding the "
            f"full A/B run's seed{{N}}_log.csv (NOT this arm's '{OUT_TAG}_seed*_log.csv').")
    rows = list(csv.DictReader(open(path)))
    if not rows or f"{labels[0]}_B_em" not in rows[0]:
        raise KeyError(
            f"{path} has no B columns (looked for '{labels[0]}_B_em'). This is "
            f"probably an A-only log. Use the original full A/B run's logs.")
    out = {}
    for lab in labels:
        em = [float(r[f"{lab}_B_em"]) for r in rows]
        pd = [float(r[f"{lab}_B_pd"]) for r in rows]
        k = max(1, int(len(em) * LATE_FRAC))
        out[lab] = {"B_em": sum(em[-k:]) / k, "B_pd": sum(pd[-k:]) / k}
    return out


def train_one_seed(seed, eval_loaders, tok, device, pos_writer):
    set_seed(seed)
    PAD, EQ, A_IDX = tok.pad_idx, tok.char_to_idx["="], tok.char_to_idx["A"]
    labels = list(eval_loaders.keys())

    train_ds = ScratchpadAdditionDataset(num_samples=15000, min_digits=3, max_digits=4,
                                         tokenizer=tok, max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # IMPORTANT: build A then B in the SAME order as the full run, so the RNG
    # stream and A's init are byte-identical. B is constructed but NOT trained.
    model_A = build_A(tok.vocab_size, device)
    _model_B_unused = build_B(tok.vocab_size, device)   # RNG parity only; never used
    del _model_B_unused
    if torch.cuda.device_count() > 1:
        model_A = nn.DataParallel(model_A)

    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    opt_A = AdamW(model_A.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sc_A = GradScaler("cuda")

    history = {lab: {"A_em": [], "A_pd": []} for lab in labels}

    seed_log = open(f"{OUT_TAG}_seed{seed}_log.csv", "w", newline="")
    slog = csv.writer(seed_log)
    header = ["epoch", "loss_A"] + [c for lab in labels for c in (f"{lab}_A_em", f"{lab}_A_pd")]
    slog.writerow(header); seed_log.flush()

    for epoch in range(EPOCHS):
        model_A.train()
        loss_sum_A = 0.0
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

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            avg_A = loss_sum_A / len(train_loader)
            print(f"  seed {seed} ep {epoch+1:4d}/{EPOCHS} | loss A {avg_A:.4f}")
            row = [epoch + 1, f"{avg_A:.4f}"]
            for lab in labels:
                a_em, a_pd = eval_metrics(model_A, eval_loaders[lab], A_IDX, PAD, device)
                history[lab]["A_em"].append(a_em); history[lab]["A_pd"].append(a_pd)
                print(f"     {lab:5s} | EM A {a_em:5.1f} | PD A {a_pd:5.1f}")
                row += [f"{a_em:.2f}", f"{a_pd:.2f}"]
            slog.writerow(row); seed_log.flush()

    seed_log.close()

    for lab in labels:
        pa_A = positional_accuracy(model_A, eval_loaders[lab], A_IDX, PAD, device)
        for p in range(MAX_POS):
            pos_writer.writerow([seed, lab, "A", p, f"{pa_A[p]:.2f}"])

    if SAVE_CHECKPOINTS:
        torch.save(model_A.state_dict(), f"{OUT_TAG}_seed{seed}_modelA.pt")

    scores = {}
    for lab in labels:
        k = max(1, int(len(history[lab]["A_em"]) * LATE_FRAC))
        scores[lab] = {m: sum(history[lab][m][-k:]) / k for m in ("A_em", "A_pd")}
    if USE_STORED_B:
        b = load_stored_B(seed, labels)        # None if no B for this seed
        if b is not None:
            for lab in labels:
                scores[lab].update(b[lab])
    return scores


def mean_std(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CharTokenizer()
    print(f"A-ONLY sweep on {device} | seeds={SEEDS} | epochs={EPOCHS} "
          f"| B reused={USE_STORED_B} from '{B_LOG_DIR}'")

    eval_loaders = {"id": materialize_loader(tok, 3, 4, N_ID_VAL, MAX_SEQ_LEN,
                                             seed=VAL_SEED, batch=BATCH_SIZE)}
    for d in OOD_DIGITS:
        eval_loaders[f"{d}dig"] = materialize_loader(tok, d, d, N_OOD_VAL, OOD_MAX_SEQ_LEN,
                                                     seed=VAL_SEED + d, batch=BATCH_SIZE)
    labels = list(eval_loaders.keys())

    pos_file = open(f"{OUT_TAG}_positional_accuracy.csv", "w", newline="")
    pos_writer = csv.writer(pos_file)
    pos_writer.writerow(["seed", "eval_set", "model", "pos_from_LSB", "accuracy"])

    per_seed = {}
    for s in SEEDS:
        print(f"\n========== SEED {s} ==========")
        per_seed[s] = train_one_seed(s, eval_loaders, tok, device, pos_writer)
        pos_file.flush()
    pos_file.close()

    print("\n" + "=" * 72)
    print("AGGREGATE (mean +/- std across seeds)" + ("" if USE_STORED_B else "  [A only]"))
    rows = [["eval_set", "metric", "A_mean", "A_std", "B_mean", "B_std", "gap_mean", "gap_std"]]
    paired = [s for s in SEEDS if all("B_em" in per_seed[s][lab] for lab in labels)]
    if USE_STORED_B and paired and paired != SEEDS:
        print(f"  (gap computed over seeds with a stored B: {paired}; "
              f"A stats over all {SEEDS})")
    for lab in labels:
        for metric, ak, bk in (("EM", "A_em", "B_em"), ("PD", "A_pd", "B_pd")):
            A = [per_seed[s][lab][ak] for s in SEEDS]
            Am, As = mean_std(A)
            if USE_STORED_B and paired:
                Ap = [per_seed[s][lab][ak] for s in paired]
                B  = [per_seed[s][lab][bk] for s in paired]
                gaps = [a - b for a, b in zip(Ap, B)]
                Bm, Bs = mean_std(B); Gm, Gs = mean_std(gaps)
                print(f"  {lab:5s} {metric} | A {Am:6.2f} +/- {As:4.2f} | B {Bm:6.2f} +/- {Bs:4.2f} "
                      f"| gap {Gm:+6.2f} +/- {Gs:4.2f}  (n_gap={len(paired)})")
                rows.append([lab, metric, f"{Am:.2f}", f"{As:.2f}", f"{Bm:.2f}", f"{Bs:.2f}",
                             f"{Gm:.2f}", f"{Gs:.2f}"])
            else:
                print(f"  {lab:5s} {metric} | A {Am:6.2f} +/- {As:4.2f}")
                rows.append([lab, metric, f"{Am:.2f}", f"{As:.2f}", "", "", "", ""])
        print()
    print("=" * 72)

    #with open(f"{OUT_TAG}_seed_sweep_summary.csv", "w", newline="") as f:
    #    csv.writer(f).writerows(rows)
    #print(f"saved -> {OUT_TAG}_seed_sweep_summary.csv, {OUT_TAG}_positional_accuracy.csv, "
    #      f"{OUT_TAG}_seed{{N}}_log.csv")


if __name__ == "__main__":
    main()