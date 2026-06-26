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

For a clean comparison this ALSO measures TEACHER-FORCED EM on the SAME final
checkpoint and the SAME fixed problems, so the teacher-forced - free-running DROP
is a pure error-accumulation signature (no late-window vs snapshot confound).
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
WEIGHTS_DIR  = "Weights"        # all checkpoints live here
# (vocab size is read from each checkpoint / the tokenizer, not hardcoded)

# checkpoints to evaluate: (label, pattern with {seed}); A loaded as full fine-tuned model
CHECKPOINTS = {
    "Rule30": "Rule30_seed{seed}_modelA.pt",
    "Rollout": "rollout_seed{seed}_modelA.pt",
    "Carry":   "carry_seed{seed}_modelA.pt",
    "Baseline": "seed{seed}_modelB.pt",
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
def eval_teacher_forced_matched(model, tok, d, n_eval, device):
    """Teacher-forced answer exact-match on the SAME fixed problems used by
    eval_freerun, by regenerating the dataset samples with the same RNG seed and
    scoring the answer region with ground-truth context (batched)."""
    model.eval()
    PAD = tok.pad_idx; A_IDX = tok.char_to_idx["A"]
    # rebuild the identical problem set: ScratchpadAdditionDataset seeded the same way
    random.seed(VAL_SEED + d)
    ds = ScratchpadAdditionDataset(num_samples=n_eval, min_digits=d, max_digits=d,
                                   tokenizer=tok, max_seq_len=GEN_MAX_SEQ_LEN)
    xs, ys = zip(*(ds[i] for i in range(n_eval)))
    X = torch.stack(xs).to(device); Y = torch.stack(ys).to(device)
    em_c = em_t = 0
    B = 256
    for i in range(0, n_eval, B):
        xb = X[i:i+B]; yb = Y[i:i+B]
        with autocast("cuda"):
            preds = torch.argmax(model(xb), dim=-1)
        Lc = yb.size(1)
        pos = torch.arange(Lc, device=device).unsqueeze(0)
        a_col = (yb == A_IDX).long().argmax(dim=1, keepdim=True)
        ans = (pos >= (a_col + 2)) & (yb != PAD)
        match = (preds == yb)
        row_ok = (match | ~ans).all(dim=1) & ans.any(dim=1)
        em_c += row_ok.sum().item(); em_t += ans.any(dim=1).sum().item()
    return 100.0 * em_c / em_t if em_t else float("nan")


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


def load_model(path, vocab, device):
    sd = torch.load(path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    # vocab from the checkpoint itself (robust to tokenizer changes)
    ckpt_vocab = sd["embedding.weight"].shape[0] if "embedding.weight" in sd else vocab
    if ckpt_vocab != vocab:
        print(f"    [note] checkpoint vocab={ckpt_vocab} != tokenizer vocab={vocab}; "
              f"building model with {ckpt_vocab} to match the checkpoint.")
    m = Rule30Transformer(ckpt_vocab, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF).to(device)
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

    rows = [["model", "seed", "digits", "freerun_EM", "parseable_pct",
             "teacherforced_EM_sameckpt", "drop_TF_minus_free"]]
    agg = {}      # (model,d) -> list of free-run EM over seeds
    agg_tf = {}   # (model,d) -> list of teacher-forced EM (same ckpt) over seeds

    for label, pat in CHECKPOINTS.items():
        for s in SEEDS:
            path = os.path.join(WEIGHTS_DIR, pat.format(seed=s))
            if not os.path.exists(path):
                print(f"  [skip] {label} seed {s}: {path} not found")
                continue
            model = load_model(path, tok.vocab_size, device)
            for d in OOD_DIGITS:
                em, parse = eval_freerun(model, tok, d, N_EVAL, device, s)
                tf_em = eval_teacher_forced_matched(model, tok, d, N_EVAL, device)
                drop = (tf_em - em) if tf_em == tf_em else float("nan")  # nan-safe
                rows.append([label, s, d, f"{em:.2f}", f"{parse:.1f}",
                             f"{tf_em:.2f}", f"{drop:.2f}"])
                agg.setdefault((label, d), []).append(em)
                agg_tf.setdefault((label, d), []).append(tf_em)
                print(f"  {label:9s} seed {s} {d}dig | free-run EM {em:5.1f}% "
                      f"| TF EM {tf_em:5.1f}% | drop {drop:5.1f} | parseable {parse:5.1f}%")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("FREE-RUN EM | TEACHER-FORCED EM (same ckpt) | DROP — mean over seeds")
    print(f"{'model':<10}" + "".join(f"{d}dig".rjust(22) for d in OOD_DIGITS))
    summary = [["model"]
               + [f"{d}dig_freerun_mean" for d in OOD_DIGITS]
               + [f"{d}dig_freerun_std" for d in OOD_DIGITS]
               + [f"{d}dig_TF_mean" for d in OOD_DIGITS]
               + [f"{d}dig_drop_mean" for d in OOD_DIGITS]]
    for label in CHECKPOINTS:
        line = f"{label:<10}"
        fmeans=[]; fstds=[]; tfmeans=[]; dropmeans=[]
        for d in OOD_DIGITS:
            fm, fsd = mean_std(agg.get((label, d), []))
            tfm, _  = mean_std(agg_tf.get((label, d), []))
            dm = (tfm - fm) if (fm==fm and tfm==tfm) else float("nan")
            fmeans.append(fm); fstds.append(fsd); tfmeans.append(tfm); dropmeans.append(dm)
            line += (f"{fm:4.1f}/{tfm:4.1f}/{dm:+4.1f}").rjust(22)
        print(line)
        summary.append([label]
                       + [f"{x:.2f}" for x in fmeans] + [f"{x:.2f}" for x in fstds]
                       + [f"{x:.2f}" for x in tfmeans] + [f"{x:.2f}" for x in dropmeans])
    print("=" * 78)
    print("cells: free-run / teacher-forced / drop   (all EM %, same final ckpt)")

    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(OUT_CSV.replace(".csv", "_summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(summary)
    print(f"\nsaved -> {OUT_CSV}, {OUT_CSV.replace('.csv','_summary.csv')}")


if __name__ == "__main__":
    main()