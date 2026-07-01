# Learning to Think: Structural Compatibility Governs Transfer to Length-Generalized Addition

Deep Learning & Applied AI (DLAI) course project — Sapienza, profs. Rodolà / Solombrino.

This repository studies **when pretraining a small causal Transformer on a synthetic
task helps it generalize to *longer* numbers** when transferred to scratchpad
multi-digit addition. The central claim is that transfer is governed by **structural
compatibility with the target's core operation (carry propagation)** — not by
locality, task sophistication, world-model strength, or how many pretrained weights
survive fine-tuning. A verified world model turns out to be **necessary but not
sufficient**.

We pretrain three arms spanning a spectrum, then fine-tune each on 3–4 digit
addition (mixed operand lengths) and test out-of-distribution (OOD) on 5/6/7-digit:

| Arm | Structure | Relation to carry propagation |
|-----|-----------|-------------------------------|
| **Rule30**  | local | compatible (local) |
| **Rollout** | long-range but **fixed-period** | mismatched |
| **Carry**   | long-range, **variable-distance** | matched |

Every arm's pretrained model (**Model A**) is compared against a shared **random-init
baseline (Model B)** fine-tuned on the *identical* schedule — only the initialization
differs. This "identical schedule, only init differs" design is the methodological
backbone of every comparison in the repo.

> The full write-up (thesis, related work, method, experiments, appendix experiment
> log) is in the accompanying 2-page report + appendix. This README covers **how to
> reproduce the numbers and figures**.

---

## 1. Requirements & environment

```bash
# with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# or with uv
uv venv && uv pip install -r requirements.txt
```

- Python 3.10+
- A CUDA GPU is assumed by the training/eval scripts (they use `torch.amp` autocast +
  `GradScaler("cuda")`). They fall back to CPU for model construction, but full
  reproduction of the 5-seed sweeps is only practical on GPU. Multi-GPU is handled
  transparently via `DataParallel`.
- Exact pinned versions are in `requirements.txt`. They reflect the development
  environment; a byte-identical lock can be regenerated with `pip freeze` / `uv pip
  compile` if needed.

---

## 2. Repository layout

```
DLAI-LearningToThink/
├── config.py                     # single source of truth: ModelConfig / FinetuneConfig /
│                                 #   ProbeConfig, SEEDS, OOD_DIGITS, checkpoint paths
├── src/
│   ├── Transformer.py            # GeneralTransformer (ALiBi, Pre-LN, 6L, d=256, 8 heads)
│   └── ArithmeticDataset.py      # CharTokenizer (vocab=17) + ScratchpadAdditionDataset
├── data_generation/
│   ├── Rule30Generator.py        # local arm — Rule 30 CA rows
│   ├── RolloutGenerator.py       # mismatched arm — fixed-period rollout
│   └── CarryOnlyGenerator.py     # matched arm — planted carry chains
├── pretraining/
│   ├── Rule30PreTraining.py
│   ├── RolloutPretraining.py
│   └── CarryOnlyPretraining.py
├── transfer/
│   └── TransferSweep.py          # unified fine-tuning sweep (TRAIN_B flag: A/B or A-only)
├── probes/
│   ├── Rule30Probe.py            # layer-sweep linear probe (local features)
│   ├── RolloutProbe.py           # layer-sweep linear probe (row-above / neighborhood)
│   └── CarryOnlyProbe.py         # layer-sweep linear probe (carry_in / gen_dist)
├── analysis/
│   ├── WeightDistance.py         # retention margin cos(A,pre) − cos(B,pre), 5-seed
│   ├── FreeRunEval.py            # non-teacher-forced (autoregressive) eval + TF drop
│   └── AttentionVisualize.py     # (illustrative only, not part of the 5-seed results)
├── plotting/
│   ├── Probes.py                 # → fig_gap_by_layer, trained_vs_floor, peak_gap_bars, carry_focus
│   ├── WeightDistance.py         # → wd_fig_global_retention / layer_margin / rell2
│   ├── TrainingMetrics.py        # → tr_fig_trajectories / 6dig_focus / indist_mastery / loss / gap_spectrum
│   ├── PositionalAccuracy.py     # → pos_fig_per_arm / overlay_6dig / overlay_7dig / heatmap
│   └── FreeRun.py                # → fr_fig_tf_vs_free / drop / free_pd_parseable
├── Weights/                      # checkpoints (NOT committed — see §5)
├── Results/                      # CSV/XLSX outputs the plotting scripts read
├── Plots/                        # generated figures
├── requirements.txt
└── README.md
```

> **Note on filenames/paths.** A few scripts hard-code checkpoint locations near the
> top (`FreeRunEval.py`'s `CHECKPOINTS` dict, `WeightDistance.py`'s `A_PATTERN` /
> `B_PATTERN`, `TransferSweep.py`'s `PRETRAINED` / `OUT_TAG`). If your `Weights/` tree
> is laid out differently from §5, edit those constants rather than moving files.
> The two `WeightDistance.py` files (one under `analysis/` that *computes* distances,
> one under `plotting/` that *draws* them) are intentionally distinct — keep them in
> separate directories to avoid a name clash.

---

## 3. Reproduction pipeline

Run in order. Each stage writes artifacts the next stage (or the plotting scripts)
consume.

### 3.1 Pretraining (Model A per arm)

```bash
python pretraining/Rule30PreTraining.py
python pretraining/RolloutPretraining.py
python pretraining/CarryOnlyPretraining.py
```

Saves to the `*_WEIGHTS` paths in `config.py` (`Weights/pretraining/…`). All three
arms share `ModelConfig` verbatim (same architecture, epochs, optimizer, AMP +
grad-clip), so any downstream difference traces to the pretraining *task*, not
hyperparameters.

### 3.2 Transfer / fine-tuning (Model A vs baseline B)

`TransferSweep.py` is the unified sweep. Set the arm + mode at the top:

```python
PRETRAINED = CARRYONLY_WEIGHTS   # or RULE30_WEIGHTS / ROLLOUT_WEIGHTS
TRAIN_B    = True                # True = paired A/B (headline gap); False = A-only
OUT_TAG    = ""                  # "" = un-tagged names (main arms); set to namespace a variant
```

Run **once per arm** with `TRAIN_B=True`:

```bash
python transfer/TransferSweep.py     # writes seed{N}_log.csv, seed{N}_modelA/B.pt,
                                     #        seed_sweep_summary.csv, positional_accuracy.csv
```

> **Overwrite caveat.** With `OUT_TAG=""` the output filenames are shared across arms.
> Move each arm's outputs into a per-arm folder (e.g. `Results/Training/rule30/`)
> **between** runs, or give each arm a distinct `OUT_TAG`. The shared random-init
> baseline **B** is regenerated on every run but is reproducible across arms (same
> seeds, random init), so any arm's B serves as the baseline.

The optional **longer-chain Carry** variant uses `TRAIN_B=False`,
`PRETRAINED=CARRYONLY_WEIGHTS_LONG`, `OUT_TAG="carryonly_long"`.

### 3.3 Probes (world-model decodability, layer sweep)

Each probe sweeps all 6 layers and writes a per-layer trained/random/gap CSV. Set
`TARGET` at the top of each file to select which latent is decoded:

```bash
python probes/Rule30Probe.py       # neighborhood (local)
python probes/RolloutProbe.py      # TARGET = cell_above | neighborhood
python probes/CarryOnlyProbe.py    # TARGET = carry_in | gen_dist
```

Report the **gap over the random-init floor**, not raw accuracy (floors are high:
`carry_in ~75%`, `gen_dist ~49%`). Read long-range latents at their **peak/deep**
layer. Aggregate the CSVs into `Results/probe_results.xlsx` (`All_Probes` sheet) for
the plotting script.

### 3.4 Weight distance (retention margin)

`analysis/WeightDistance.py` computes cosine similarity of each fine-tuned body to its
pretrained init, vs. the random-init baseline. Point `PRETRAINED` / `A_PATTERN` /
`OUT_CSV` at each arm and run:

```bash
python analysis/WeightDistance.py   # → weight_distance_<arm>.csv (+ _summary.csv)
```

Cross-arm comparisons use the **retention margin** `cos(A,pre) − cos(B,pre)`; raw
cosines are not comparable across arms (different inits/vocab).

### 3.5 Free-running evaluation (honest scope-limiter)

`analysis/FreeRunEval.py` runs autoregressive (non-teacher-forced) generation and a
matched teacher-forced eval **on the same operand set**, so the TF−free drop is a pure
error-accumulation signal.

```bash
python analysis/FreeRunEval.py      # → freerun_results.csv (+ _summary.csv)
```

Expect free-run EM ≈ 0 OOD across all arms: the teacher-forced transfer advantage does
**not** survive error accumulation. This is reported honestly as a scope limit.

### 3.6 Figures

The plotting scripts are **read-only** consumers of `Results/*` → `Plots/*` (they run
no training/eval). Adjust the `INDIR` / `OUTDIR` constants at the top to match your
tree:

```bash
python plotting/Probes.py
python plotting/WeightDistance.py
python plotting/TrainingMetrics.py     # NOTE: only fig_loss is un-commented in main()
python plotting/PositionalAccuracy.py
python plotting/FreeRun.py
```

Shared palette across all figure sets: **Rule30** `#d1495b`, **Carry** `#1b9e77`,
**Rollout** `#6a3d9a`, **Baseline** `#8a8a8a`. Switch `savefig` to `.pdf` for vector
output in LaTeX.

---

## 4. Script → result mapping

| Script | Role | Reads | Writes |
|--------|------|-------|--------|
| `pretraining/*PreTraining.py` | pretrain Model A | generators | `Weights/pretraining/*.pt` |
| `transfer/TransferSweep.py` | fine-tune A (+B) | pretrained `*.pt` | `seed{N}_log.csv`, `seed{N}_modelA/B.pt`, `seed_sweep_summary.csv`, `positional_accuracy.csv` |
| `probes/*Probe.py` | layer-sweep linear probe | pretrained `*.pt` | `*_probe_layers_*.csv` |
| `analysis/WeightDistance.py` | retention margin | `*_modelA/B.pt`, pretrained | `weight_distance_<arm>(_summary).csv` |
| `analysis/FreeRunEval.py` | autoregressive eval + TF drop | fine-tuned `*.pt` | `freerun_results(_summary).csv` |
| `plotting/Probes.py` | figures | `probe_results.xlsx` | `fig_gap_by_layer` + 3 others |
| `plotting/WeightDistance.py` | figures | `weight_distance_*_summary.csv` | `wd_fig_*` |
| `plotting/TrainingMetrics.py` | figures | `full_results.xlsx`, `results_summary.xlsx`, seed logs | `tr_fig_*` |
| `plotting/PositionalAccuracy.py` | figures | `*_positional_accuracy.csv` | `pos_fig_*` |
| `plotting/FreeRun.py` | figures | `freerun_results.csv` | `fr_fig_*` |

**Aggregated result workbooks** (`Results/`): `full_results.xlsx` (raw per-seed/epoch
EM & PD), `results_summary.xlsx` (windowed gap-over-baseline), `probe_results.xlsx`
(layer-sweep probe gaps).

---

## 5. Checkpoints & data

Model checkpoints (`.pt`) and generated datasets are **not committed** (size, and the
grading criterion to ship code that regenerates results). Recreate them via §3.1–3.2.
Recommended `Weights/` layout matching the scripts' expected patterns:

```
Weights/
├── pretraining/
│   ├── rule30_pretrained_new.pt
│   ├── rule30_rollout_pretrained.pt
│   ├── carryonly_pretrained.pt
│   └── carryonly_pretrained_long.pt        # optional long-chain variant
├── rule30/      Rule30_seed{0..4}_modelA.pt
├── rollout/     Rollout_seed{0..4}_modelA.pt
├── carryonly/   carryonly_seed{0..4}_modelA.pt
└── baseline/    seed{0..4}_modelB.pt
```

All checkpoints are vocab-17 (tokenizer: `<PAD>` + `0123456789+=C:,A`).

---

## 6. Reproducibility & honest caveats

- **5 seeds** (`SEEDS = [0,1,2,3,4]`), all headline numbers reported as mean ± std.
- Headline metric is **Exact-Match (EM)**; **Per-Digit (PD)** and positional accuracy
  are diagnostics. All accuracy numbers are **teacher-forced** unless labelled free-run.
- EM error bars **overlap** between Carry and Rule30 at 6-digit — we do **not** claim
  EM statistical separation there; PD carries the "best *and* most reliable" claim.
- In-distribution accuracy saturates (~100%) for all arms, so the OOD gap is pure
  length generalization (control: `tr_fig_indist_mastery`).
- Rule30's AMP+grad-clipping was harmonized with the other arms after its original
  checkpoint was trained; it converges identically but is not byte-for-byte identical
  to the pre-clipping run (documented in the appendix).
- One stylistic inconsistency is left as an acknowledged choice: `Rule30Probe.py`
  hard-codes its class count, while the other two probes derive it from config
  (see appendix).

The **appendix** contains the full experiment log — probe-alignment bug, fp16 collapse,
label imbalance, the gen_dist clamp fix, the two dissociations, seed instabilities, and
the free-running collapse — as a record of what was tried and how validity was checked.
