# Project Execution Plan: Sequence Pre-Training on 1D Cellular Automata

## Phase 1: Environment Setup and Micro-Architecture
Your first goal is to establish the pipeline and build a miniature version of the model that runs on your local CPU.

1. **Initialize the Repository:** Set up your project structure locally in VS Code (creating the `core/` folder for `.py` files and the root folder for `.ipynb` notebooks).
2. **Configure the Cloud Sync:** Choose your preferred Colab/VS Code connection method (e.g., Google Drive sync) and verify that changes saved in VS Code update instantly in your Colab environment.
3. **Build the Transformer Class:** Code your hybrid architecture in PyTorch using `nn.TransformerEncoderLayer`. 
4. **Implement Relative Positions:** Integrate ALiBi or RoPE positional encodings. Ensure you are not using standard sinusoidal embeddings.
5. **Configure Pre-LN:** Double-check that your Layer Normalization is applied *before* the attention mechanism to ensure stable training gradients.
6. **Local CPU Sanity Check:** Initialize a micro-model (e.g., 2 layers, 64-dimensional embeddings) and pass a dummy tensor of shape `(batch_size, sequence_length)` through it to confirm the causal mask and matrix dimensions are strictly correct.

---

## Phase 2: Data Engineering
Next, build the data generators. Since everything is algorithmic, you do not need to download or clean external datasets.

1. **Code the Rule 30 Generator:** Write a Python function to generate 1D Rule 30 arrays. Implement periodic boundary conditions so the edges wrap around, preventing artifacts.
2. **Implement On-the-Fly CA Batches:** Wrap the Rule 30 generator in a standard PyTorch `Dataset` and `DataLoader` so it generates new, randomized sequences dynamically during training, preventing the model from memorizing a finite dataset.
3. **Code the Arithmetic Scratchpad:** Write a generator for your fine-tuning phase. It must take two numbers (e.g., 3-digit and 4-digit integers) and output the step-by-step target string, explicitly writing out the carried bits from right to left.
4. **Build the Tokenizer:** Create a simple character-level tokenizer. Your vocabulary only needs the digits 0-9, arithmetic operators, formatting tokens, and the binary 0 and 1 for Rule 30.

---

## Phase 3: The Cognitive Gym (Causal Pre-Training)
Move to the cloud GPU to build your model's fundamental causal routing.

1. **Scale the Architecture:** Initialize the full model in your Colab notebook with 1 to 5 million parameters.
2. **Execute Pre-Training:** Train the model to predict the next token in the Rule 30 sequences using the AdamW optimizer and a high weight decay. 
3. **Validate World Modeling:** Freeze the Transformer weights. Pass in unseen Rule 30 sequences and train a simple 2-layer MLP probe on the hidden states to predict cell transitions.
4. **Checkpointing:** Once the probe achieves near-perfect accuracy, save the pre-trained Transformer weights (e.g., `rule30_pretrained.pt`).

---

## Phase 4: Transfer Learning and Grokking
You will now execute the A/B test by training two models simultaneously on your arithmetic scratchpad data.

1. **Initialize Model A (Pre-Trained):** Load `rule30_pretrained.pt` and replace the binary classification head with a new head scaled to your arithmetic tokenizer vocabulary.
2. **Initialize Model B (Baseline):** Initialize an identical, entirely random Transformer.
3. **Set Up the Fine-Tuning Loop:** Feed both models the 3-digit and 4-digit scratchpad addition data.
4. **Train Through the Grokking Delay:** Monitor the validation accuracy. Both models will likely hit 100% training accuracy while validation accuracy stays near 0%. Do not stop training. Wait for the validation accuracy to spike.
5. **Record Convergence Metrics:** Document exactly how many optimization steps it took for Model A to grok the arithmetic compared to Model B.

---

## Phase 5: OOD Evaluation and Analysis
The final step is proving the Out-of-Distribution length generalization hypothesis.

1. **Generate the Test Set:** Create a strict testing dataset of 5-digit, 6-digit, and 7-digit addition problems, formatted with the same scratchpad rules.
2. **Execute the OOD Test:** Run both Model A and Model B on this unseen test set. 
3. **Plot the Degradation:** Create a line chart where the X-axis is the sequence length and the Y-axis is accuracy. 
4. **Extract Attention Maps:** Extract the attention weights from the first layer of both models. Visualize them using heatmaps to show whether the Rule 30 model maintained strict, localized shift-invariance compared to the baseline.
5. **Draft the Report:** Synthesize the training curves, grokking speed, OOD degradation plots, and attention heatmaps into your final academic paper.
