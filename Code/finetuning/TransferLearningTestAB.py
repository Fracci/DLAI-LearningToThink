"""
TransferSweep.py — finetunes Model A (pretrained init) and, optionally, Model B
(random-init baseline) on the scratchpad-addition target, seed-by-seed, on
identical batches/schedule. A single TRAIN_B flag selects the mode:

  TRAIN_B = True   -> paired A/B run: trains both, reports the A-B GAP per eval
                      set. This is the headline mode (6-digit EM/PD gaps,
                      positional figure).
  TRAIN_B = False  -> A-only run (e.g. the optional long-chain carry variant)..

Model B is always CONSTRUCTED regardless of TRAIN_B, so initialization RNG is
consumed identically either way — only training/eval/saving of B is skipped
in A-only mode. Also records positional accuracy (distance from LSB) per
seed/eval-set.
"""
import random
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.Transformer import GeneralTransformer
from config import FinetuneConfig, EVAL_EVERY, CARRYONLY_WEIGHTS, CARRYONLY_WEIGHTS_LONG, RULE30_WEIGHTS, ROLLOUT_WEIGHTS, OOD_DIGITS, SEEDS
from src.ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset

# CONFIG
PRETRAINED   = CARRYONLY_WEIGHTS          # Change here to change the pretrained arm/init
TRAIN_B      = False                      # True = paired A/B (gap); False = A-only sweep
OUT_TAG      = "carryonly"                # "" = legacy un-tagged names (main arms, downstream-wired);
                                          #      set e.g. "carryonly_long" to namespace a variant run
VAL_SEED     = FinetuneConfig.val_seed
N_ID_VAL     = 2000
N_OOD_VAL    = 3000
SAVE_CHECKPOINTS = True


def tagged(name):
    """Prefix an output filename with OUT_TAG when set, else return it unchanged
    (so the headline arms keep the un-tagged names downstream scripts expect)."""

    return f"{OUT_TAG}_{name}" if OUT_TAG else name


def set_seed(s):
    """Seed python/torch/cuda RNGs together so a seed sweep is actually reproducible."""

    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_loss_targets(x, y, eq_idx, pad_idx):
    """Mask the loss to positions at/after '=' — the model isn't trained to
    predict the (already-given) problem statement, only the scratchpad+answer."""

    L = x.size(1)
    pos = torch.arange(L, device=x.device).unsqueeze(0)
    eq_col = (x == eq_idx).long().argmax(dim=1, keepdim=True)
    return torch.where(pos >= eq_col, y, torch.full_like(y, pad_idx))


def answer_region(y, a_idx, pad_idx):
    """Boolean mask over the final-answer digits only (after 'A:'), used for
    EM/PD metrics — scratchpad reasoning steps are excluded from reported accuracy."""

    L = y.size(1)
    pos = torch.arange(L, device=y.device).unsqueeze(0)
    a_col = (y == a_idx).long().argmax(dim=1, keepdim=True)

    return (pos >= (a_col + 2)) & (y != pad_idx)


def materialize_loader(tok, min_d, max_d, n, max_seq_len, seed, batch):
    """Pre-generate a FIXED, seeded eval set so A, B, and every seed are scored on
    the identical id/OOD problems — otherwise per-call regeneration adds eval noise."""

    random.seed(seed)
    ds = ScratchpadAdditionDataset(num_samples=n, min_digits=min_d, max_digits=max_d,
                                   tokenizer=tok, max_seq_len=max_seq_len)
    xs, ys = zip(*(ds[i] for i in range(n)))

    return DataLoader(TensorDataset(torch.stack(xs), torch.stack(ys)), batch_size=batch)


@torch.no_grad()
def eval_metrics(model, loader, a_idx, pad_idx, device):
    """Exact-Match (whole final answer correct) and Per-Digit accuracy over the
    answer region only."""

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
def positional_accuracy(model, loader, a_idx, pad_idx, device, max_pos=FinetuneConfig.max_pos):
    """Per-digit accuracy binned by distance from the LSB (last answer column),
    counting BACKWARD so units/top digits align across different-length answers."""

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
    """Build Model A: init from PRETRAINED, dropping embedding/fc_out (vocab
    mismatch with the pretraining task) and loading the rest of the transformer
    body strict=False."""

    m = GeneralTransformer(vocab, FinetuneConfig.d_model, FinetuneConfig.n_heads, FinetuneConfig.n_layers, FinetuneConfig.dim_feedforward).to(device)
    sd = torch.load(PRETRAINED, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items()
          if not k.startswith("embedding.") and not k.startswith("fc_out.")}
    m.load_state_dict(sd, strict=False)

    return m


def build_B(vocab, device):
    """Random-init baseline, same architecture as A. Always constructed for 
    init-RNG parity; only trained when TRAIN_B is True."""

    return GeneralTransformer(vocab, FinetuneConfig.d_model, FinetuneConfig.n_heads, FinetuneConfig.n_layers, FinetuneConfig.dim_feedforward).to(device)


def train_one_seed(seed, eval_loaders, tok, device, pos_writer):
    """Finetune A (and B if TRAIN_B) in lockstep for one seed on identical
    batches, logging loss + per-eval-set EM/PD every EVAL_EVERY epochs, then
    writing positional accuracy and (optionally) checkpoints. Returns the
    late_frac-windowed EM/PD per eval set — B keys are present only in A/B mode."""

    set_seed(seed)
    PAD, EQ, A_IDX = tok.pad_idx, tok.char_to_idx["="], tok.char_to_idx["A"]
    labels = list(eval_loaders.keys())

    train_ds = ScratchpadAdditionDataset(num_samples=FinetuneConfig.num_samples, min_digits=3, max_digits=4,
                                         tokenizer=tok, max_seq_len=FinetuneConfig.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=FinetuneConfig.batch_size, shuffle=True, pin_memory=True)

    model_A = build_A(tok.vocab_size, device)

    # B is ALWAYS built so init-RNG consumption is identical whether or not it
    # ends up being trained; freed immediately in A-only mode so it isn't
    # resident on-GPU during A's training.
    model_B = build_B(tok.vocab_size, device)
    if not TRAIN_B:
        del model_B
        model_B = None

    if torch.cuda.device_count() > 1:
        model_A = nn.DataParallel(model_A)
        if TRAIN_B:
            model_B = nn.DataParallel(model_B)

    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    opt_A = AdamW(model_A.parameters(), lr=FinetuneConfig.lr, weight_decay=FinetuneConfig.weight_decay)
    sc_A = GradScaler("cuda")

    if TRAIN_B:
        opt_B = AdamW(model_B.parameters(), lr=FinetuneConfig.lr, weight_decay=FinetuneConfig.weight_decay)
        sc_B = GradScaler("cuda")

    # history keys depend on mode: A-only tracks just A_em/A_pd; A/B adds B_em/B_pd.
    metric_keys = ("A_em", "A_pd") + (("B_em", "B_pd") if TRAIN_B else ())
    history = {lab: {m: [] for m in metric_keys} for lab in labels}

    # per-seed CSV log — header/columns depend on mode.
    seed_log = open(tagged(f"seed{seed}_log.csv"), "w", newline="")
    slog = csv.writer(seed_log)

    if TRAIN_B:
        header = ["epoch", "loss_A", "loss_B"]
        for lab in labels:
            header += [f"{lab}_A_em", f"{lab}_B_em", f"{lab}_A_pd", f"{lab}_B_pd"]
    else:
        header = ["epoch", "loss_A"] + [c for lab in labels for c in (f"{lab}_A_em", f"{lab}_A_pd")]
    slog.writerow(header); seed_log.flush()

    for epoch in range(FinetuneConfig.epochs):
        model_A.train()
        if TRAIN_B:
            model_B.train()
        loss_sum_A = loss_sum_B = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            yl = build_loss_targets(x, y, EQ, PAD)

            opt_A.zero_grad()
            with autocast("cuda"):
                lA = crit(model_A(x).reshape(-1, tok.vocab_size), yl.reshape(-1))
            sc_A.scale(lA).backward(); sc_A.unscale_(opt_A)
            nn.utils.clip_grad_norm_(model_A.parameters(), FinetuneConfig.grad_clip)
            sc_A.step(opt_A); sc_A.update()
            loss_sum_A += lA.item()

            if TRAIN_B:
                opt_B.zero_grad()
                with autocast("cuda"):
                    lB = crit(model_B(x).reshape(-1, tok.vocab_size), yl.reshape(-1))
                sc_B.scale(lB).backward(); sc_B.unscale_(opt_B)
                nn.utils.clip_grad_norm_(model_B.parameters(), FinetuneConfig.grad_clip)
                sc_B.step(opt_B); sc_B.update()
                loss_sum_B += lB.item()

        if epoch % EVAL_EVERY == 0 or epoch == FinetuneConfig.epochs - 1:
            avg_A = loss_sum_A / len(train_loader)
            if TRAIN_B:
                avg_B = loss_sum_B / len(train_loader)
                print(f"  seed {seed} ep {epoch+1:4d}/{FinetuneConfig.epochs} | loss A {avg_A:.4f} B {avg_B:.4f}")
                row = [epoch + 1, f"{avg_A:.4f}", f"{avg_B:.4f}"]
            else:
                print(f"  seed {seed} ep {epoch+1:4d}/{FinetuneConfig.epochs} | loss A {avg_A:.4f}")
                row = [epoch + 1, f"{avg_A:.4f}"]

            for lab in labels:
                a_em, a_pd = eval_metrics(model_A, eval_loaders[lab], A_IDX, PAD, device)
                history[lab]["A_em"].append(a_em); history[lab]["A_pd"].append(a_pd)

                if TRAIN_B:
                    b_em, b_pd = eval_metrics(model_B, eval_loaders[lab], A_IDX, PAD, device)
                    history[lab]["B_em"].append(b_em); history[lab]["B_pd"].append(b_pd)
                    print(f"     {lab:5s} | EM  A {a_em:5.1f} B {b_em:5.1f} "
                          f"| PD  A {a_pd:5.1f} B {b_pd:5.1f}")
                    row += [f"{a_em:.2f}", f"{b_em:.2f}", f"{a_pd:.2f}", f"{b_pd:.2f}"]

                else:
                    print(f"     {lab:5s} | EM A {a_em:5.1f} | PD A {a_pd:5.1f}")
                    row += [f"{a_em:.2f}", f"{a_pd:.2f}"]
            slog.writerow(row); seed_log.flush()

    seed_log.close()

    # end-of-seed positional breakdown (final models); B rows only in A/B mode.
    for lab in labels:
        pa_A = positional_accuracy(model_A, eval_loaders[lab], A_IDX, PAD, device)
        for p in range(FinetuneConfig.max_pos):
            pos_writer.writerow([seed, lab, "A", p, f"{pa_A[p]:.2f}"])

        if TRAIN_B:
            pa_B = positional_accuracy(model_B, eval_loaders[lab], A_IDX, PAD, device)
            for p in range(FinetuneConfig.max_pos):
                pos_writer.writerow([seed, lab, "B", p, f"{pa_B[p]:.2f}"])

    # DataParallel is not unwrapped before saving; downstream loaders strip
    # "module." on load, so this is a minor inconsistency, not a correctness bug.
    if SAVE_CHECKPOINTS:
        torch.save(model_A.state_dict(), tagged(f"seed{seed}_modelA.pt"))
        if TRAIN_B:
            torch.save(model_B.state_dict(), tagged(f"seed{seed}_modelB.pt"))

    # late_frac-windowed average (not just the final epoch) — smooths over the
    # noisy end-of-training trajectory.
    scores = {}
    for lab in labels:
        k = max(1, int(len(history[lab]["A_em"]) * FinetuneConfig.late_frac))
        scores[lab] = {m: sum(history[lab][m][-k:]) / k for m in metric_keys}
    return scores


def mean_std(xs):
    """Population mean/std across the seed list."""

    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, v ** 0.5


def main():
    """Build fixed id/OOD eval sets, run train_one_seed for every seed, and write
    the seed-averaged summary CSV + per-seed positional CSV."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CharTokenizer()
    mode = "A/B (gap)" if TRAIN_B else "A-only"
    
    print(f"Transfer sweep [{mode}] on {device} | seeds={SEEDS} "
          f"| epochs={FinetuneConfig.epochs} | tag={OUT_TAG or '(none)'}")

    # fixed eval sets: in-distribution (3-4) + each OOD length
    eval_loaders = {"id": materialize_loader(tok, 3, 4, N_ID_VAL, FinetuneConfig.max_seq_len,
                                             seed=VAL_SEED, batch=FinetuneConfig.batch_size)}
    for d in OOD_DIGITS:
        eval_loaders[f"{d}dig"] = materialize_loader(tok, d, d, N_OOD_VAL, FinetuneConfig.ood_max_seq_len,
                                                     seed=VAL_SEED + d, batch=FinetuneConfig.batch_size)
    labels = list(eval_loaders.keys())

    pos_file = open(tagged("positional_accuracy.csv"), "w", newline="")
    pos_writer = csv.writer(pos_file)
    pos_writer.writerow(["seed", "eval_set", "model", "pos_from_LSB", "accuracy"])

    per_seed = {}
    for s in SEEDS:
        print(f"\n========== SEED {s} ==========")
        per_seed[s] = train_one_seed(s, eval_loaders, tok, device, pos_writer)
        pos_file.flush()
    pos_file.close()

    print("AGGREGATE (mean +/- std across seeds)")
    if TRAIN_B:
        rows = [["eval_set", "metric", "A_mean", "A_std", "B_mean", "B_std", "gap_mean", "gap_std"]]
        metric_map = (("EM", "A_em", "B_em"), ("PD", "A_pd", "B_pd"))
    else:
        rows = [["eval_set", "metric", "A_mean", "A_std"]]
        metric_map = (("EM", "A_em"), ("PD", "A_pd"))

    for lab in labels:
        for spec in metric_map:
            if TRAIN_B:
                metric, ak, bk = spec
                A = [per_seed[s][lab][ak] for s in SEEDS]
                B = [per_seed[s][lab][bk] for s in SEEDS]
                gaps = [a - b for a, b in zip(A, B)]
                Am, As = mean_std(A); Bm, Bs = mean_std(B); Gm, Gs = mean_std(gaps)
                print(f"  {lab:5s} {metric} | A {Am:6.2f} +/- {As:4.2f} | B {Bm:6.2f} +/- {Bs:4.2f} "
                      f"| gap {Gm:+6.2f} +/- {Gs:4.2f}")
                rows.append([lab, metric, f"{Am:.2f}", f"{As:.2f}", f"{Bm:.2f}", f"{Bs:.2f}",
                             f"{Gm:.2f}", f"{Gs:.2f}"])
            else:
                metric, ak = spec
                A = [per_seed[s][lab][ak] for s in SEEDS]
                Am, As = mean_std(A)
                print(f"  {lab:5s} {metric} | A {Am:6.2f} +/- {As:4.2f}")
                rows.append([lab, metric, f"{Am:.2f}", f"{As:.2f}"])
        print()

    with open(tagged("seed_sweep_summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved -> {tagged('seed_sweep_summary.csv')}, {tagged('positional_accuracy.csv')}, "
          f"{tagged('seed{N}_log.csv')}")


if __name__ == "__main__":
    main()