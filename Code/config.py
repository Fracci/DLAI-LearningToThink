from dataclasses import dataclass, field
from typing import List


EVAL_EVERY = 5
OOD_DIGITS = [5, 6, 7]
SEEDS = [0, 1, 2, 3, 4]

RULE30_WEIGHTS = "rule30_pretrained_new.pt"
ROLLOUT_WEIGHTS = "rule30_rollout_pretrained.pt"
CARRYONLY_WEIGHTS = "carryonly_pretrained.pt"


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256##
    n_heads: int = 8##
    n_layers: int = 6##
    dim_feedforward: int = 1024##
    block_size: int = 128         # Maximum sequence context window
    dropout: float = 0.1##
    epochs: int = 1##
    batch_size: int = 256##
    num_samples: int = 20000##
    lr: float = 1e-3##
    weight_decay: float = 0.2##
    grad_clip: float = 1.0##


@dataclass(frozen=True)
class FinetuneConfig:
    d_model: int = 256##
    n_heads: int = 8##
    n_layers: int = 6##
    dim_feedforward: int = 1024##
    dropout: float = 0.1
    epochs: int = 1##
    batch_size: int = 256##
    num_samples: int = 15000##
    weight_decay: float = 0.1##
    grad_clip: float = 1.0##
    lr: float = 5e-4##
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])##
    val_seed: int = 20240601##
    max_pos : int = 12##
    max_seq_len : int = 128##
    ood_max_seq_len : int = 160##
    late_frac : float = 0.5##


@dataclass(frozen=True)
class ProbeConfig:
    iters_per_epoch: int = 80##
    d_model: int = 256##
    n_heads: int = 8##
    n_layers: int = 6##
    dim_feedforward: int = 1024##
    block_size: int = 128         # Maximum sequence context window
    dropout: float = 0.1
    epochs: int = 1
    batch_size: int = 128##
    num_samples: int = 20000
    lr: float = 3e-4##
    weight_decay: float = 0.2
    grad_clip: float = 1.0

    layer_target: int = 3         # Target layer for internal state extraction
    probe_lr: float = 1e-3
    epochs: int = 20##