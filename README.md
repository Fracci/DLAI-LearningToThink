# Sequence Pre-Training on 1D Cellular Automata for Out-of-Distribution Length Generalization

This repository contains the codebase for a Deep Learning research project exploring **decoupled Transformer training**. 

When neural sequence models learn from raw data or algorithmic tasks, they must simultaneously map structural logic (causality, spatial invariance) and domain-specific representations (vocabulary, operators). We hypothesize that these two phases can be decoupled. This project pre-trains a lightweight Transformer (1–5M parameters) exclusively on a deterministic mathematical environment—the **Rule 30 1D Cellular Automaton**—to build a "cognitive foundation" of local-to-global causal propagation that is inherently scale-invariant. 

The mathematically pre-trained model is then fine-tuned on synthetic algorithmic reasoning tasks (such as multi-digit arithmetic) to evaluate whether learning deterministic execution laws allows the attention mechanism to achieve superior **Out-of-Distribution (OOD) length generalization** compared to a randomly initialized baseline.

## Key Features
* **Rule 30 Synthetic Data:** Native 1D sequence generation for causal pre-training, bypassing heavy 2D tokenization overhead while providing complex, chaotic spatiotemporal dependencies.
* **Hybrid Transformer Architecture:** Built using native PyTorch primitives (`nn.TransformerEncoderLayer`) configured for strict causal attention, with specific focus on length-invariant relative positional encodings and rapid single-GPU iteration.
* **Transfer Learning & Robustness Testing:** Fine-tuning the pre-trained causal attention layers on mathematical string evaluation restricted to a specific sequence length (e.g., 3-digit and 4-digit arithmetic).
* **OOD Generalization Benchmarking:** Direct A/B performance testing against an identical architecture trained from scratch, explicitly evaluating accuracy and attention map degradation when scaled to unseen sequence lengths (e.g., 5-to-7-digit arithmetic).
