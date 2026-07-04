# Task 2: Conceptual Writeup

## 24. What does the MLP do in each block?

Attention handles the "communication" part — it lets tokens share information with each other by computing weighted sums. But a weighted sum is still a linear operation. The MLP comes after attention and applies non-linear transformations (ReLU) to the aggregated features at each position independently. So attention decides what information to gather, and the MLP decides what to do with it.

I like to think of it as: attention is "listening to everyone", MLP is "thinking about what you heard".

## 25. Pre-norm vs post-norm

I used pre-norm (LayerNorm before the sublayer, not after). The idea is that with pre-norm, the residual path stays completely clean — gradients can flow all the way from the loss back to the first layer without passing through any normalisation. With post-norm the gradient has to go through a LayerNorm at every block, which can cause issues in deep networks.

From what I've read, the original "Attention Is All You Need" paper used post-norm, but most modern implementations (like GPT-2, nanoGPT) switched to pre-norm because it's more stable to train.

## 26. Generated text comparison

**Transformer samples:**
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

Compared to the bigram model from Task 1, this is a clear improvement — the text has some semblance of sentence structure and even character names that look Shakespeare-ish. The bigram output was basically random character soup. Still not great by any means, but you can see the model is picking up on patterns like dialogue formatting and line breaks.
