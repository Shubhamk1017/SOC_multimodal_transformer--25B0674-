# Task 5 Writeup — Contrastive Learning and InfoNCE Loss

---

### 17. InfoNCE Loss Explained
Up until now, every classification model I built was predicting a fixed label out of a set of categories (like classifying CIFAR-10 images into 10 classes). Contrastive learning is completely different because there are no fixed target labels. 

Instead, InfoNCE asks the model: *"Given an image and a batch of captions, can you pick out the one caption that actually belongs to this image?"*

For image $i$, its matching caption $i$ is the positive pair, and all other $N-1$ captions in the batch act as negatives. We compute cosine similarities between image and text embeddings, treat these as logits, and use cross-entropy where the target label is the batch index itself (the diagonal). We do this in both directions (image-to-text and text-to-image) and average them.

---

### 18. Why Temperature $\tau$ Matters
Temperature scales the similarity dot products before we take the softmax ($\text{logits} = \frac{1}{\tau} \cdot \text{similarity}$). It basically controls how sharp or flat our probability distribution is:

- **If temperature is too high ($\tau \approx 1.0$):** The logits become flat and uniform. The model assigns roughly equal probability to all candidates, so the loss stays high, gradients become weak, and learning is very slow.
- **If temperature is too low ($\tau \le 0.01$):** The softmax becomes overly sharp. The model gets obsessed with hard negatives, leading to huge gradient spikes, numerical instability, or training collapse.
- **Why CLIP learns it:** Parameterizing temperature  allows the network to automatically tune logit sharpness via gradient descent as embeddings get better during training.

---

### 19. Why We Must L2-Normalize Embeddings
Before taking the dot product, we normalize both image embeddings and text embeddings to lie on the unit sphere,

If we don't normalize:
- The dot product measures vector length as well as direction.
- The model could cheat the loss simply by making matching vectors super long instead of actually aligning their directions in feature space.
- Normalization restricts similarity to pure cosine similarity between $-1.0$ and $1.0$, forcing the model to focus entirely on semantic alignment.

---

### 20. Toy Alignment Experiment Results
In `task5/toy_alignment.py`, I created 32 random 192-dim image vectors and 32 random 192-dim text vectors, and trained two linear projection heads ($192 \to 128$) using InfoNCE loss for 500 steps.

- **Loss Curve:** The loss started at 4.0879 (very close to the theoretical baseline log(32) approx 3.4657) and dropped smoothly step-by-step down to 0.000049 at Step 500.
- **Final Similarity Matrix:**
  - **Average Diagonal Value (matching pairs):** `0.8621`
  - **Average Off-Diagonal Value (unmatched pairs):** `-0.0274`

This confirmed that the InfoNCE loss works as intended before trying it on real Flickr8k data.

---

### 21. How Batch Size Affects Contrastive Learning
In InfoNCE, the batch size **N** directly dictates how many negative examples **N-1** the model sees for every positive pair.

- **Why larger batch size helps:** Having more negatives per batch makes the task harder and forces the model to learn finer distinctions between similar images/captions. It also gives a better approximation of the overall dataset distribution.
- **The Catch:** Memory usage grows quadratically ($N \times N$ similarity matrix). Large batch sizes can quickly hit GPU Out-Of-Memory (OOM) errors unless you use techniques like mixed precision or gradient accumulation.
