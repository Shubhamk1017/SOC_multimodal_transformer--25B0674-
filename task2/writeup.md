# Task 2: Conceptual Writeup

## 24. What is the role of the MLP in each block?
While the Multi-Head Attention layer is responsible for **communication** (mixing information *across* different sequence positions), the Feed-Forward MLP is responsible for **computation** (processing information *within* each individual position independently). Attention determines *where* to look and aggregates that information, but it is purely a weighted sum of linear projections. The MLP applies non-linear transformations (using ReLU) to the aggregated feature vectors, giving the model the capacity to "think" about what that aggregated information actually means and learn complex, non-linear representations before passing it to the next layer.

## 25. Pre-norm versus post-norm
I used **Pre-norm**, where LayerNorm is applied *before* the sub-layers (Attention and MLP), and the residual connection directly adds the sub-layer output to the original input (`x = x + sublayer(ln(x))`). 
Pre-norm is easier to train for deep networks because it keeps the main residual pathway completely clean and identity-mapped from the bottom of the network all the way to the top. During backpropagation, gradients can flow directly from the final loss to the earliest layers without passing through any LayerNorm operations, completely bypassing the vanishing gradient problem. In post-norm (`x = ln(x + sublayer(x))`), the gradient must pass through a LayerNorm operation at every single block, which can cause the gradient to degrade over many layers.

## 26. Generated Text Comparison
**Transformer Model Samples (Baseline):**
```text
BOWARWCOMILLO:
What I have ease not dlam your have a brother
busin prasurer repurs; sign up thy corturn, our not looks sween,
We doled 'ISABELO: for liper in the burgt;
Let his duke I ded and: no wight betake that leave his house
And I life in Cite,
What Ranceid it.
Thousand restrouse abon,
God it 
```

**Qualitative Comparison to Task 1:**
Compared to the single-head attention model in Task 1, this text is vastly superior. It has started generating plausible English words ("brother", "house", "life", "Thousand") and the capitalization pattern clearly mimics a script (Character names followed by a colon). It still produces gibberish sequences ("BOWARWCOMILLO", "prasurer"), but the overarching structure, word spacing, and rhythm are unmistakably starting to look like Shakespeare.

## 27. Ablation Study: Residuals vs LayerNorm
**Removing Residuals:** Catastrophic failure. The validation loss remained stuck around 3.24. Without residual connections, the deep network suffers from the vanishing gradient problem. The gradient has to multiply through 4 layers of Attention and MLPs, diminishing it before it can effectively train the lower layers. Residual connections provide a "gradient highway" that allows gradients to flow directly through the addition operation unchanged.
**Removing LayerNorm:** At this very small scale (4 layers, tiny dataset), removing LayerNorm only marginally impacted the final loss (1.68 vs 1.71). However, in larger models, removing LayerNorm would be catastrophic. LayerNorm stabilizes the forward pass by keeping activations bounded and standardizing their scale, which prevents gradients from exploding or vanishing and ensures the model can train deeply without diverging.

## 28. What was hardest, and what clicked?
**Hardest:** Deriving the softmax Jacobian and the full chain rule back to $Q$ by hand was definitely the most mathematically rigorous part. Keeping track of the indices for the Kronecker delta and ensuring the matrix dimensions lined up during the product rule took careful attention to detail.
**What clicked unexpectedly:** Writing the `Block` class and seeing how simple a full Transformer actually is. Once you strip away the massive scale and infrastructure, realizing that GPT-3 is essentially just a `for` loop over a self-attention module and a 2-layer MLP was a massive "aha" moment.
