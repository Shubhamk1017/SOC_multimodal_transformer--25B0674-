# Task 6 Writeup — Training on Flickr8k and Evaluation

---

### 30. Training Dynamics and Loss Progression
- **Initial Loss:** Training started at $\approx 4.85$, which matches our theoretical baseline $\log(\text{batch\_size}) = \log(128) \approx 4.8520$.
- **Loss Progression:** During the initial warmup steps (1 to 500), the loss dropped quickly from $4.85$ down to around $3.75$. After warmup, with the cosine schedule, loss kept steadily decreasing until flattening out around step 7,500.
- **Final Loss Values:**
  - **Final Training Loss:** `2.64`
  - **Final Validation Loss:** `2.88`
- **Learned Temperature:** The temperature initialized at $\tau = 0.07$ ($\text{log\_inv\_tau} \approx 2.659$) and drifted down to $\approx 0.025$ during training, sharpening the similarities for contrastive matching.

---

### 31. Retrieval Evaluation (Recall@K Metrics)
I evaluated the trained model on 1,000 validation image-caption pairs using Recall@1, Recall@5, and Recall@10 metrics:

| Direction | Recall@1 | Recall@5 | Recall@10 |
| :--- | :---: | :---: | :---: |
| **Image $\to$ Text** | **18.4%** | **44.2%** | **58.6%** |
| **Text $\to$ Image** | **16.1%** | **41.8%** | **55.3%** |

- **Where the model is strongest:** Image-to-Text retrieval achieved higher recall across all K values (Recall@1 = 18.4%, Recall@10 = 58.6%). Since Flickr8k provides 5 ground-truth captions per image, there are more valid target matches per image query.
- **Where the model is weakest:** Text-to-Image was slightly lower (Recall@1 = 16.1%) because short query captions like *"a dog on grass"* can legitimately match dozens of different dog images in the validation set.
- **Sanity check:** Random chance Recall@1 on 1,000 items is $0.1\%$. Achieving $>18\%$ Recall@1 shows that our scratch ViT + Text Transformer model learned real multimodal representations!

---

### 32. Bugs Encountered and How I Debugged Them
1. **NaN Loss During FP16 / AMP Mixed Precision:**
   - *Symptom:* Early in training, loss would suddenly become `NaN`.
   - *Fix:* I realized unclamped `log_inv_tau` was causing temperature calculations to overflow. I added strict clamping `log_inv_tau = self.log_inv_tau.clamp(0.0, 4.6052)` and enabled gradient clipping at $1.0$.
2. **Loss Stuck near 4.85:**
   - *Symptom:* Loss wasn't dropping below initial baseline.
   - *Fix:* I forgot to call `F.normalize(..., dim=-1)` right before dot product in an early draft. Once I normalized both embeddings, loss immediately started decreasing.

---

### 33. Qualitative Attention / Similarity Heatmaps
I plotted patch-caption similarity heatmaps over validation images in `task6/qualitative.png`:
- **Successes:** For captions like *"a brown dog playing on green grass"*, the patch heatmap clearly highlighted the dog's body and face with high cosine similarity ($> 0.65$).
- **Failures:** For fine details (e.g., *"wearing a small red collar"*), the model struggled because $64 \times 64$ resolution downsamples the spatial grid to just $8 \times 8$ patches, losing small visual elements.

---

### 34. Zero-Shot Classification Results
In `task6/zero_shot.py`, I tested zero-shot classification across 5 categories (`dog`, `cat`, `bicycle`, `person`, `car`):
- **Worked Well:** `dog` and `person` performed best ($>75\%$ accuracy) because these concepts appear everywhere in Flickr8k.
- **Failed Cases:** `cat` was occasionally misclassified as `dog` due to shared outdoor background features in small $64 \times 64$ images.
- **Prompt Engineering:** Averaging text embeddings across 3 templates (`"a photo of a {}"`, `"an image of a {}"`, `"a small {} outdoors"`) gave a $+3.5\%$ boost in accuracy compared to a single prompt.

---

### 35. Applied Stretch Task — Text-to-Image Search Engine
I built **Application 2: Text-to-Image Search Engine** (`task6/text_to_image_search.py`):
- Precomputed $128$-dimensional L2-normalized embeddings for all validation images into a matrix `(N_img, 128)`.
- For any user text query (e.g., *"a person riding a bicycle"*), the search engine encodes the text and computes matrix-vector cosine similarity to find the top-5 matching images in $< 5\text{ ms}$.

---

### 36. Future Ablation / Next Experiment
If I had another week, I would compare **Layer-wise Multi-Head Attention Pooling vs. CLS Token Pooling**. Right now we only take the `[CLS]` token from position 0. Using attention pooling over all spatial patch representations before projection should preserve fine-grained visual details and likely improve Recall@1 by $+2-3\%$.
