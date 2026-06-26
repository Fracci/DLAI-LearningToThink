from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 13          # e.g., digits 0-9, +, =, padding, etc.
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    block_size: int = 128         # Maximum sequence context window
    dropout: float = 0.1

@dataclass(frozen=True)
class FinetuneConfig:
    epochs: int = 300
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.01
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    val_seed: int = 20240601

@dataclass(frozen=True)
class ProbeConfig:
    layer_target: int = 3         # Target layer for internal state extraction
    probe_lr: float = 1e-3
    probe_epochs: int = 50