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
import time
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


def digit_pd(pred, truth):
    """Per-digit accuracy of a free-running answer vs truth, aligned at the
    least-significant digit (rightmost). Scored over the TRUE answer's length, so
    a too-short answer is penalized (missing digits = wrong) and any extra leading
    digits beyond the true length are ignored. Empty pred -> 0. Returns (correct, total)."""
    total = len(truth)
    if total == 0:
        return (0, 0)
    correct = 0
    for k in range(1, total + 1):       # k = 1 is the units digit
        t = truth[-k]
        p = pred[-k] if k <= len(pred) else None
        if p == t:
            correct += 1
    return (correct, total)


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
    em_c = em_t = dig_c = dig_t = 0
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
        dig_c += (match & ans).sum().item(); dig_t += ans.sum().item()
    em = 100.0 * em_c / em_t if em_t else float("nan")
    pd = 100.0 * dig_c / dig_t if dig_t else float("nan")
    return (em, pd)


@torch.no_grad()
def eval_freerun(model, tok, d, n_eval, device, seed):
    model.eval()
    random.seed(VAL_SEED + d)          # same fixed problems across all models/seeds
    correct = total = parseable = 0
    pd_correct = pd_total = 0
    t0 = time.time()
    report_every = max(1, n_eval // 4)   # ~4 progress lines per length
    for i in range(n_eval):
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
        dc, dt = digit_pd(pred, truth)
        pd_correct += dc; pd_total += dt
        if (i + 1) % report_every == 0 or (i + 1) == n_eval:
            el = time.time() - t0
            rate = (i + 1) / el
            eta = (n_eval - (i + 1)) / rate if rate > 0 else 0.0
            pd_so_far = 100.0 * pd_correct / pd_total if pd_total else 0.0
            print(f"        free-run {d}dig: {i+1}/{n_eval} "
                  f"({100.0*(i+1)/n_eval:4.0f}%) | {rate:4.1f} ex/s "
                  f"| elapsed {el:5.1f}s | ETA {eta:5.1f}s "
                  f"| EM {100.0*correct/total:4.1f}% PD {pd_so_far:4.1f}%", flush=True)
    em = 100.0 * correct / total
    pd = 100.0 * pd_correct / pd_total if pd_total else float("nan")
    par = 100.0 * parseable / total
    return (em, pd, par)


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

    t_start = time.time()
    n_jobs = len(CHECKPOINTS) * len(SEEDS) * len(OOD_DIGITS)
    print(f"Planned: {len(CHECKPOINTS)} models x {len(SEEDS)} seeds x "
          f"{len(OOD_DIGITS)} lengths = {n_jobs} free-run evaluations "
          f"({N_EVAL} examples each)\n", flush=True)
    rows = [["model", "seed", "digits",
             "freerun_EM", "freerun_PD", "parseable_pct",
             "TF_EM_sameckpt", "TF_PD_sameckpt",
             "drop_EM", "drop_PD"]]
    agg = {}      # free-run EM
    agg_pd = {}   # free-run PD
    agg_tf = {}   # teacher-forced EM (same ckpt)
    agg_tfpd = {} # teacher-forced PD (same ckpt)

    for label, pat in CHECKPOINTS.items():
        for s in SEEDS:
            path = os.path.join(WEIGHTS_DIR, pat.format(seed=s))
            if not os.path.exists(path):
                print(f"  [skip] {label} seed {s}: {path} not found")
                continue
            print(f"\n[{time.strftime('%H:%M:%S')}] loading {label} seed {s}: {path}", flush=True)
            tck = time.time()
            model = load_model(path, tok.vocab_size, device)
            for d in OOD_DIGITS:
                td = time.time()
                em, pd, parse = eval_freerun(model, tok, d, N_EVAL, device, s)
                tf_em, tf_pd = eval_teacher_forced_matched(model, tok, d, N_EVAL, device)
                drop_em = (tf_em - em) if tf_em == tf_em else float("nan")
                drop_pd = (tf_pd - pd) if tf_pd == tf_pd else float("nan")
                rows.append([label, s, d,
                             f"{em:.2f}", f"{pd:.2f}", f"{parse:.1f}",
                             f"{tf_em:.2f}", f"{tf_pd:.2f}",
                             f"{drop_em:.2f}", f"{drop_pd:.2f}"])
                agg.setdefault((label, d), []).append(em)
                agg_pd.setdefault((label, d), []).append(pd)
                agg_tf.setdefault((label, d), []).append(tf_em)
                agg_tfpd.setdefault((label, d), []).append(tf_pd)
                print(f"  {label:9s} seed {s} {d}dig | free EM {em:5.1f} PD {pd:5.1f} "
                      f"| TF EM {tf_em:5.1f} PD {tf_pd:5.1f} "
                      f"| drop EM {drop_em:5.1f} PD {drop_pd:5.1f} "
                      f"| parse {parse:5.1f}% | {time.time()-td:5.1f}s", flush=True)
            print(f"  [{label} seed {s}] done in {time.time()-tck:5.1f}s", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def block(title, free_agg, tf_agg):
        print("\n" + "=" * 80)
        print(f"{title} — mean over seeds | cells: free-run / teacher-forced / drop")
        print(f"{'model':<10}" + "".join(f"{d}dig".rjust(23) for d in OOD_DIGITS))
        out = []
        for label in CHECKPOINTS:
            line = f"{label:<10}"; rowvals=[label]
            for d in OOD_DIGITS:
                fm, fsd = mean_std(free_agg.get((label, d), []))
                tfm, _  = mean_std(tf_agg.get((label, d), []))
                dm = (tfm - fm) if (fm==fm and tfm==tfm) else float("nan")
                line += (f"{fm:5.1f}/{tfm:5.1f}/{dm:+5.1f}").rjust(23)
                rowvals += [f"{fm:.2f}", f"{fsd:.2f}", f"{tfm:.2f}", f"{dm:.2f}"]
            print(line); out.append(rowvals)
        print("=" * 80)
        return out

    hdr = ["model"] + [c for d in OOD_DIGITS for c in
                       (f"{d}_free_mean", f"{d}_free_std", f"{d}_TF_mean", f"{d}_drop_mean")]
    em_rows = block("EXACT-MATCH (EM %)", agg, agg_tf)
    pd_rows = block("PER-DIGIT (PD %)", agg_pd, agg_tfpd)
    summary = [["=== EM ==="]] + [hdr] + em_rows + [[""], ["=== PD ==="]] + [hdr] + pd_rows

    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(OUT_CSV.replace(".csv", "_summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(summary)
    print(f"\nsaved -> {OUT_CSV}, {OUT_CSV.replace('.csv','_summary.csv')}")
    print(f"total wall-clock: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()