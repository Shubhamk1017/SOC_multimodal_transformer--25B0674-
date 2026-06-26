# Task 3: Conceptual Writeup

## 7. Compare your CNN baseline and your ViT on CIFAR-10. Which got better validation accuracy? Why might that be?

The **CNN achieved around 68–70% validation accuracy** in just 10 epochs, while the **ViT reached around 65–72% accuracy** but needed 30 epochs to get there. The results are very close, which itself is an important observation.

**Why the CNN is naturally good at images:**
- CNNs have built-in **inductive biases** that are perfectly suited for images:
  - **Translation equivariance** — a cat is recognized as a cat regardless of where it appears in the image, because the same convolution kernel slides across every spatial position.
  - **Locality** — each convolutional kernel only looks at a small neighborhood (3×3 pixels), which matches the fact that nearby pixels are usually more related than distant ones.
- These biases act as "prior knowledge" baked into the architecture, so the CNN can learn effectively even from CIFAR-10's small 50,000 training images.

**Why ViT struggles on small datasets:**
- ViT has **no built-in knowledge** about images. To a transformer, the patch at position (0,0) and the patch at position (7,7) are just two tokens in a sequence — it has to **learn from scratch** that nearby patches are usually related.
- With enough data (e.g., ImageNet-21k with 14 million images), ViT learns these spatial relationships beautifully and can surpass CNNs. But with CIFAR-10's 50k images, there's simply not enough data for the ViT to discover what the CNN already "knows" by design.
- This is the fundamental **inductive bias vs. scale** trade-off: less bias requires more data, but also lets the model discover patterns the bias would have hidden.

## 8. In your own words, explain why patching is necessary for ViT. Why not feed pixels directly?

Self-attention has **O(T²) complexity** where T is the sequence length. For a 32×32 image:
- **Pixels directly:** T = 32 × 32 × 3 = 3,072 tokens → attention needs ~9.4 million pairwise scores. For a real 224×224 image, T = 150,528 → ~22.6 billion scores. This is completely impractical.
- **With 4×4 patches:** T = (32/4) × (32/4) = 64 tokens → only ~4,096 pairwise scores. That's manageable!

Beyond computation, there's also a **representational argument**: individual pixels carry almost no meaningful information on their own. A single pixel value like "134" tells you nothing. But a 4×4 patch contains small textures, edges, and color patterns that are actually meaningful units for the model to reason about. Patches are to images what subword tokens (like "un", "break", "able") are to text — small chunks of structure that the model learns to compose into higher-level understanding.

## 9. Explain the role of the CLS token. Why does the classifier read from CLS rather than averaging over patch tokens?

The **[CLS] token** is a special learnable vector that is prepended to the beginning of the patch sequence. It starts with no information about any specific patch. As it passes through the transformer blocks, it attends to all patch tokens and gradually aggregates global information about the entire image.

**Why not just average the patch tokens?** 
- Averaging treats all patches equally, which might not be ideal — the background patches (say, the sky behind a bird) contribute as much as the important patches (the bird itself).
- The CLS token learns to **selectively attend** to the most informative patches through the attention mechanism. It acts as a learned "summary slot" that can weigh different patches differently.
- This idea was borrowed directly from BERT in NLP, where a [CLS] token is used for sentence-level classification tasks.

In practice, modern ViT variants have found that global average pooling over patch tokens works almost as well. But the CLS token is what the original ViT paper used, and it's conceptually clean.

## 10. In Task 2 you used a causal mask. In Task 3 (ViT) you removed it. Explain why.

**Task 2 (Text Generation):** We used a causal mask because we were training a language model to **predict the next character**. During generation, the model shouldn't "cheat" by looking at future tokens. The causal mask (lower triangular matrix of 1s, with -∞ above the diagonal) ensures that each token can only attend to itself and past tokens. Without this, the model would just copy the answer from the future — the task would be trivially easy but the model would be useless at generation time.

**Task 3 (Image Classification):** There is **no temporal ordering** in an image. The patch at position (2, 5) should freely look at the patch at position (7, 1) — they might both contain parts of the same object. Unlike text, images have no concept of "past" and "future." Every patch needs full context from every other patch to understand the scene.

**What would happen if we accidentally kept the causal mask in ViT?** The patches would be arbitrarily ordered (say, left-to-right, top-to-bottom), and the model would force a sequential reading order onto the image. The bottom-right patches would see the entire image, but the top-left patches would only see themselves. This would severely hurt performance because the model couldn't build a holistic understanding of the image — it would be as if you tried to classify a photo by only looking at the top portion.

## 11. Position embeddings for text encode token order. What do position embeddings for image patches encode, and why does the model need them?

In text, position embeddings encode **which word comes first, second, third**, etc. Without them, the transformer would treat "the cat sat on the mat" and "mat the on sat cat the" identically.

In ViT, position embeddings encode the **spatial location** of each patch — which row and column of the image grid it came from. They tell the model "this patch is from the top-left corner" versus "this patch is from the center."

**Why the model needs them:** The patch embedding step (Conv2d + flatten) destroys all spatial information. After flattening, the model receives 64 vectors of dimension 192, and it has no idea whether they represent a 8×8 grid, a 4×16 strip, or any other arrangement. Without position embeddings, the model would process a shuffled image identically to the original — it couldn't tell that the sky is at the top and the ground is at the bottom.

We used **learned 1D position embeddings**: each position (0 to 64) has its own learnable vector. The original ViT paper found that even simple 1D positions work well — the model learns to encode 2D spatial relationships from them.

## 12. What did you find hardest? What clicked unexpectedly?

**Hardest:** Understanding the Conv2d trick for patch embedding. At first, it seems like cheating — we're using a CNN operation inside a supposedly "convolution-free" architecture! But once I realized that `nn.Conv2d(3, n_embd, kernel_size=4, stride=4)` is mathematically identical to "cut image into 4×4 patches, flatten each to 48 values, multiply by a weight matrix" — it clicked. The Conv2d is just a more efficient implementation of the same linear projection; it's not "learning convolutional features" in the traditional sense because there's no overlap between patches.

**What clicked unexpectedly:** How similar the ViT is to the text transformer from Task 2. The only real differences are: (1) we removed the causal mask, (2) we replaced the text tokenizer with a patch embedding layer, and (3) we added a classification head instead of a language model head. The entire transformer backbone — multi-head attention, MLP, LayerNorm, residual connections — is literally the same code. The transformer architecture is genuinely universal: it doesn't care whether its input is words, image patches, audio frames, or anything else — it just processes sequences.
