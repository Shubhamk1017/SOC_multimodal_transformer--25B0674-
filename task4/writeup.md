# Task 4 Writeup — Cross-Attention & Putting ViT + Transformer Together

---

### What is Cross-Attention and How Does It Work?
Cross-attention is the bridge between our Vision Transformer (image encoder) and text decoder. In self-attention, a text token only looks at other text tokens. But in cross-attention, we let the text decoder look at the visual patches from the ViT.

Specifically:
- **Queries ($Q$):** Generated from the text decoder representations.
- **Keys ($K$) and Values ($V$):** Generated from the ViT patch embeddings of the image.

The attention formula remains the standard scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

This allows every text token to attend over all visual patches, picking up spatial information about where objects are in the image.

---

### Decoder Architecture & Causal Masking
In the multimodal decoder block, I stacked:
1. **Causal Masked Self-Attention:** Ensures that while generating text, tokens can only look at previous tokens and not future words.
2. **Visual Cross-Attention:** Connects the text sequence with the ViT patch features.
3. **Feed-Forward MLP:** A standard 2-layer MLP with GELU activations to process the combined embeddings.

---

### Verification and Gradient Tests
I wrote `task4/test_vlm.py` to check if everything runs smoothly. The model takes images of shape `(B, 3, 64, 64)` and text token IDs, passes them through the ViT encoder and decoder blocks, and outputs logits over the vocabulary.

I also verified that calling `.backward()` properly computes gradients for both the ViT patch embedding layer and the language model head, proving that gradients flow back into the vision encoder through cross-attention.
