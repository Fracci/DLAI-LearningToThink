# Sequence Pre-Training on 1D Cellular Automata for Financial Time-Series Forecasting

This repository contains the codebase for a Deep Learning research project exploring **decoupled Transformer training**. 

When language models learn from empirical data, they must simultaneously map structural logic (causality, temporal dependencies) and domain-specific content. We hypothesize that these two phases can be separated. This project pre-trains a lightweight Transformer (1–5M parameters) exclusively on a deterministic mathematical environment—the **Rule 30 1D Cellular Automaton**—to build a "cognitive foundation" of local-to-global causal propagation. 

The mathematically pre-trained model is then fine-tuned on a noisy, real-world sequential dataset to predict **financial market shocks**, comparing its convergence speed and predictive accuracy against a randomly initialized baseline.

## Key Features
* **Rule 30 Synthetic Data:** Native 1D sequence generation for causal pre-training without heavy 2D tokenization overhead.
* **Hybrid Transformer Architecture:** Built using native PyTorch primitives (`nn.TransformerEncoderLayer`) configured for strict causal attention and rapid single-GPU iteration.
* **Transfer Learning Pipeline:** Fine-tuning the causal attention maps on financial time-series data to detect structural breaks and cascading volatility.
* **A/B Performance Testing:** Direct benchmark against an identical architecture trained from scratch on the target financial data.
