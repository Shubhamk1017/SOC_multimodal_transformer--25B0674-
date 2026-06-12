# Task 2: Conceptual Writeup

## 24. What is the role of the MLP in each block?
While the Multi-Head Attention layer is responsible for **communication** (mixing information *across* different sequence positions), the Feed-Forward MLP is responsible for **computation** (processing information *within* each individual position independently). Attention determines *where* to look and aggregates that information, but it is purely a weighted sum of linear projections. The MLP applies non-linear transformations (using ReLU) to the aggregated feature vectors, giving the model the capacity to "think" about what that aggregated information actually means and learn complex, non-linear representations before passing it to the next layer.

## 25. Pre-norm versus post-norm
I used **Pre-norm**, where LayerNorm is applied *before* the sub-layers (Attention and MLP), and the residual connection directly adds the sub-layer output to the original input (`x = x + sublayer(ln(x))`). 
Pre-norm is easier to train for deep networks because it keeps the main residual pathway completely clean and identity-mapped from the bottom of the network all the way to the top. During backpropagation, gradients can flow directly from the final loss to the earliest layers without passing through any LayerNorm operations, completely bypassing the vanishing gradient problem. In post-norm (`x = ln(x + sublayer(x))`), the gradient must pass through a LayerNorm operation at every single block, which can cause the gradient to degrade over many layers.

## 26. Generated Text Comparison
**Transformer Model Samples (Baseline):**
```text
(Waiting for generation to finish...)
```

**Qualitative Comparison to Task 1:**
(Waiting to review the generated text before commenting...)

## 27. Ablation Study: Residuals vs LayerNorm
(Waiting for ablation runs to complete...)

## 28. What was hardest, and what clicked?
**Hardest:** Deriving the softmax Jacobian and the full chain rule back to $Q$ by hand was definitely the most mathematically rigorous part. Keeping track of the indices for the Kronecker delta and ensuring the matrix dimensions lined up during the product rule took careful attention to detail.
**What clicked unexpectedly:** Writing the `Block` class and seeing how simple a full Transformer actually is. Once you strip away the massive scale and infrastructure, realizing that GPT-3 is essentially just a `for` loop over a self-attention module and a 2-layer MLP was a massive "aha" moment.
