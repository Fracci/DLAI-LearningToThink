"""
Free-running (NON-teacher-forced) evaluation.

Teacher-forced metrics feed the ground-truth scratchpad token at every step, so
a single carry error never propagates -- they measure per-position competence,
not whether the model can actually CARRY OUT addition. This script instead lets
the model generate the entire scratchpad + answer autoregressively from just the
prompt 'n1+n2=', feeding its OWN outputs back. One early error cascades, exactly
as in real use. The metric is exact-match on the answer digits parsed from the
MODEL'S generated stream.

Expect lower numbers than teacher-forced -- that is the point. The honest
question is whether a pretrained arm's advantage SURVIVES free-running.

Greedy (argmax) decoding for determinism. Evaluated on a fixed seeded set at the
final checkpoint of each seed (free-running is ~max_new_tokens x slower than
teacher-forced, so keep the set modest).
"""
import random
import csv
import os
import torch
import torch.nn as nn
from torch.amp import autocast

from Transformer import Rule30Transformer
from ArithmeticDataset import CharTokenizer, ScratchpadAdditionDataset

# ---------------- config ----------------
D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
OOD_DIGITS   = [5, 6, 7]
N_EVAL       = 400              # per length (free-running is slow; keep modest)
MAX_NEW_TOKENS = 140            # generation budget (7-dig scratchpad ~98 tokens)
GEN_MAX_SEQ_LEN = 200           # >= prompt + MAX_NEW_TOKENS
VAL_SEED     = 20240601
VOCAB_FOR_LOAD = 16             # arithmetic vocab (CharTokenizer)

# checkpoints to evaluate: (label, pattern with {seed}); A loaded as full fine-tuned model
CHECKPOINTS = {
    "Rule30": "Weights/Rule30_seed{seed}_modelA.pt",
    "Rollout": "Weights/rollout_seed{seed}_modelA.pt",
    "Carry":   "Weights/carry_seed{seed}_modelA.pt",
    "Baseline": "Weights/seed{seed}_modelB.pt",
}
SEEDS = [0, 1, 2, 3, 4]
OUT_CSV = "freerun_results.csv"
# ----------------------------------------


def build_prompt_and_truth(tok, n1, n2, max_seq_len):
    """Return (prompt_ids, true_answer_str). Prompt is 'n1+n2=' only."""
    prompt = f"{n1}+{n2}="
    return tok.encode(prompt, max_len=None), str(n1 + n2)


@torch.no_grad()
def generate(model, prompt_ids, tok, device, max_new=MAX_NEW_TOKENS):
    """Greedy autoregressive generation from prompt_ids (1D LongTensor)."""
    ids = prompt_ids.tolist()
    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with autocast("cuda"):
            logits = model(x)
        nxt = int(torch.argmax(logits[0, -1], dim=-1).item())
        if nxt == tok.pad_idx:           # model emitted pad -> stop
            break
        ids.append(nxt)
    return ids


def parse_answer(ids, tok):
    """Extract digits after the LAST 'A:' in the generated id stream.
    Returns the answer string, or '' if not parseable."""
    text = tok.decode(torch.tensor(ids))
    if "A:" not in text:
        return ""
    tail = text.split("A:")[-1]
    # answer is the leading run of digits
    out = []
    for ch in tail:
        if ch.isdigit():
            out.append(ch)
        else:
            break
    return "".join(out)


@torch.no_grad()
def eval_freerun(model, tok, d, n_eval, device, seed):
    model.eval()
    random.seed(VAL_SEED + d)          # same fixed problems across all models/seeds
    correct = total = parseable = 0
    for _ in range(n_eval):
        n1 = random.randint(10 ** (d - 1), 10 ** d - 1)
        n2 = random.randint(10 ** (d - 1), 10 ** d - 1)
        prompt_ids, truth = build_prompt_and_truth(tok, n1, n2, GEN_MAX_SEQ_LEN)
        gen = generate(model, prompt_ids.to(device), tok, device)
        pred = parse_answer(gen, tok)
        total += 1
        if pred != "":
            parseable += 1
        if pred == truth:
            correct += 1
    return (100.0 * correct / total, 100.0 * parseable / total)


def load_model(path, device):
    m = Rule30Transformer(VOCAB_FOR_LOAD, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF).to(device)
    sd = torch.load(path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)              # full fine-tuned model -> strict load
    return m


def mean_std(xs):
    if not xs: return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) == 1: return (m, 0.0)
    return (m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CharTokenizer()
    print(f"Free-running (greedy) eval on {device} | {N_EVAL}/length | "
          f"max_new={MAX_NEW_TOKENS}\n")

    rows = [["model", "seed", "digits", "freerun_EM", "parseable_pct"]]
    agg = {}   # (model,d) -> list of EM over seeds

    for label, pat in CHECKPOINTS.items():
        for s in SEEDS:
            path = pat.format(seed=s)
            if not os.path.exists(path):
                print(f"  [skip] {label} seed {s}: {path} not found")
                continue
            model = load_model(path, device)
            for d in OOD_DIGITS:
                em, parse = eval_freerun(model, tok, d, N_EVAL, device, s)
                rows.append([label, s, d, f"{em:.2f}", f"{parse:.1f}"])
                agg.setdefault((label, d), []).append(em)
                print(f"  {label:9s} seed {s} {d}dig | free-run EM {em:5.1f}% "
                      f"| parseable {parse:5.1f}%")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("FREE-RUNNING EM — mean +/- std over seeds")
    print(f"{'model':<10}" + "".join(f"{d}dig".rjust(12) for d in OOD_DIGITS))
    summary = [["model"] + [f"{d}dig_mean" for d in OOD_DIGITS]
               + [f"{d}dig_std" for d in OOD_DIGITS]]
    for label in CHECKPOINTS:
        line = f"{label:<10}"
        means = []; stds = []
        for d in OOD_DIGITS:
            m, sd = mean_std(agg.get((label, d), []))
            means.append(m); stds.append(sd)
            line += (f"{m:5.1f}±{sd:<4.1f}").rjust(12)
        print(line)
        summary.append([label] + [f"{x:.2f}" for x in means] + [f"{x:.2f}" for x in stds])
    print("=" * 60)

    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(OUT_CSV.replace(".csv", "_summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(summary)
    print(f"\nsaved -> {OUT_CSV}, {OUT_CSV.replace('.csv','_summary.csv')}")


if __name__ == "__main__":
    main()