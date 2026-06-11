## Engineering and Execution Checklist

Here is a comprehensive checklist of critical engineering considerations, algorithmic mechanics, and potential pitfalls to keep in mind as you develop this project solo.

---

### Phase 1: Hybrid Transformer Architecture Design

#### Positional Encoding Constraints
* **The Pitfall:** Standard absolute sinusoidal positional encodings *cannot* extrapolate to lengths unseen during training. If you use them, Phase 5 will fail by default regardless of pre-training.
* **The Action:** Implement **RoPE (Rotary Position Embedding)** or **ALiBi (Attention with Linear Biases)**. These relative embedding techniques are explicitly designed to allow attention maps to compute distance-based scaling invariant to absolute sequence length.

#### Optimization & Precision
* **The Action:** Stick to PyTorch’s standard `nn.TransformerEncoderLayer` with a decoder-style causal mask using `nn.Transformer.generate_square_subsequent_mask`. 
* **The Concern:** Ensure you place `LayerNorm` *before* the attention blocks (Pre-LN) rather than after (Post-LN). Pre-LN is significantly more stable for training deep transformers from scratch and helps avoid early gradient explosion.

---

### Phase 2: The Mathematical Environment (Rule 30 CA)

#### Edge Conditions
* **The Concern:** When generating the 1D lattice, what happens to the cells at the absolute boundaries (index 0 and index $N-1$)? 
* **The Action:** Use **periodic boundary conditions** (wrapping around like a torus) or pad with fixed zeros. If you don't keep this consistent, the model will struggle to generalize because the rules will suddenly break at the edges of the sequence.

#### Pattern Chaotic Depth
* **The Concern:** If you initialize the Rule 30 array with a single active pixel in the center, it creates a clean, predictable triangle before the chaotic structures emerge. The model might overfit to the early structured region.
* **The Action:** Generate sequences using a mix of single-seed initializations and fully randomized binary vectors to force the model to learn the generalized transition logic, not just a specific spatial pattern layout.

---

### Phase 3: Direct Sequence Tokenization

#### Vocabulary Sparsity
* **The Action:** Start with the simplest vocabulary: **bit-level tokenization** (Vocabulary Size = 2: `0` or `1`).
* **The Concern:** If you switch to k-mer tokenization (e.g., chunking 4 bits into a single token to shrink sequence length), you create a vocabulary size of 16. While this shortens the sequence, it forces the transformer to learn the permutations inside each token rather than directly observing the $3 \to 1$ cell mapping of Rule 30. Keep it at the bit level first to mirror pure causal transitions.

---

### Phase 4: Causal Pre-Training (The Cognitive Gym)

#### Tracking Internal World Models
* **The Concern:** How do you know the model actually learned the rules of the CA instead of just memorizing training sequences?
* **The Action:** Borrow the probing methodology from *Othello-GPT*. Freeze your pre-trained Transformer weights, feed it a novel Rule 30 row sequence, and train a shallow 2-layer MLP classifier to see if it can successfully predict cell transitions on unseen rows. If probe accuracy is near 100%, your cognitive gym is a success.

#### Preventing Memorization Loss Collapse
* **The Concern:** Because the dataset is synthetic, a 5M parameter model can easily overfit and completely memorize a finite set of trajectories.
* **The Action:** Ensure your training data generator produces sequences *on the fly* inside the PyTorch data loader loop. The model should never see the exact same initial state twice during pre-training.

---

### Phase 5: Transfer Learning & Length Generalization

#### The "Grokking" Delay
* **The Concern:** Algorithmic tasks like multi-digit addition are notorious for a phenomenon called **grokking**. The model will quickly hit 100% training accuracy via memorization while its validation accuracy stays at 0% for thousands of steps.
* **The Action:** Do not stop training early. You must continue optimizing the fine-tuning loop long after the training loss hits a floor. Algorithmic generalization often happens abruptly, long after severe overfitting has occurred.

#### Intermediate "Scratchpads"
* **The Concern:** Expecting a Transformer to read `123+456=` and instantly output `579` at long sequence lengths fails because it tries to compute the entire operation in a single forward pass.
* **The Action:** Format your fine-tuning dataset to output intermediate calculations (a scratchpad). For example, train it to output the addition step-by-step from right to left, tracking the "carry" bit explicitly. This maps perfectly onto the step-by-step causal logic the model learned from tracking Rule 30.

#### The True Baseline Check
* **The Concern:** During the OOD length generalization test (testing 6-digit math when trained only on 3-digit math), both your models might drop in accuracy.
* **The Action:** Success is measured by the *rate of degradation*. Plot a curve where the x-axis is the token length ($N, N+1, N+2\dots$) and the y-axis is the accuracy. Your project succeeds if the Rule 30 pre-trained model decays significantly slower than the randomly initialized baseline.
