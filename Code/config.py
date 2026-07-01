"""
config.py — central configuration for the project: shared model architecture
(ModelConfig, used identically across all three pretraining arms so "identical
schedule, only init differs" holds), finetuning hyperparameters (FinetuneConfig),
probing hyperparameters (ProbeConfig), and the pretrained-checkpoint paths /
seed list / OOD digit lengths referenced throughout the eval and plotting
scripts. Kept as one file so every script imports the same numbers rather than
risking silent drift between e.g. pretraining and finetuning d_model.
"""
from dataclasses import dataclass, field
from typing import List


EVAL_EVERY = 5                  # how often (in epochs) training loops log/eval
OOD_DIGITS = [5, 6, 7]          # out-of-distribution operand lengths tested at transfer time
SEEDS = [0, 1, 2, 3, 4]         # the 5 seeds all headline results are averaged over

# Pretrained Model-A checkpoints, one per arm of the spectrum
RULE30_WEIGHTS = "Weights/pretraining/rule30_pretrained_new.pt"
ROLLOUT_WEIGHTS = "Weights/pretraining/rule30_rollout_pretrained.pt"
CARRYONLY_WEIGHTS = "Weights/pretraining/carryonly_pretrained.pt"
CARRYONLY_WEIGHTS_LONG = "Weights/pretraining/carryonly_pretrained_long.pt"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture + pretraining hyperparameters, shared verbatim by all three
    pretraining scripts (Rule30/Rollout/CarryOnly) so any transfer difference
    traces to the pretraining TASK, not a hyperparameter mismatch."""
    
    d_model: int = 256
    n_heads: int = 8              # ALiBi get_slopes is power-of-2 only — correct at 8, see Transformer.py note
    n_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    epochs: int = 300
    batch_size: int = 256
    num_samples: int = 20000
    lr: float = 1e-3
    weight_decay: float = 0.2
    grad_clip: float = 1.0


@dataclass(frozen=True)
class FinetuneConfig:
    """Scratchpad-addition finetuning hyperparameters, shared by Model A and
    Model B (only the init differs — this is the methodological backbone of
    the A/B comparison)."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 1024
    epochs: int = 300
    batch_size: int = 256
    num_samples: int = 20000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    lr: float = 5e-4
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    val_seed: int = 20240601      # fixed, distinct from training seeds so val problems don't leak into training
    max_pos : int = 12
    max_seq_len : int = 128       # must fit the longest in-distribution (3-4 digit) scratchpad string
    ood_max_seq_len : int = 160   # longer budget for OOD eval (5-7 digit scratchpads run longer)
    late_frac : float = 0.5       # fraction of training used as the "late" window for windowed-average reporting


@dataclass(frozen=True)
class ProbeConfig:
    """Linear-probe training hyperparameters, shared by Rule30Probe/RolloutProbe/
    CarryOnlyProbe. Architecture fields (d_model/n_heads/n_layers/dim_feedforward)
    must match ModelConfig so the frozen pretrained checkpoints load correctly."""

    iters_per_epoch: int = 80
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 1024
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 20