# Task 2: Conceptual Writeup

## 24. What does the MLP do in each block?

Attention is mostly about communication — it lets different tokens talk to each other and mix their information. But attention by itself is just a weighted sum, which is a linear operation. The MLP (Feed Forward network) comes after attention and applies non-linear stuff (like ReLU activation) to the features at each position independently. So attention gathers the information, and the MLP processes it.

i like to think of it as attention is "listening to other words" and MLP is "thinking about what it heard".

## 25. Pre-norm vs post-norm

i went with pre-norm, meaning LayerNorm is applied before the self-attention and MLP layers, not after. Pre-norm is much better for training deep models because it keeps the residual path clean. Gradients can flow directly from the end of the network back to the beginning without getting messed up by normalisation layers. With post-norm, the gradients have to pass through LayerNorm at every block, which can make training unstable.

Apparently the original paper used post-norm, but almost every modern model (like GPT/nanoGPT) uses pre-norm now because it just trains much easier.

## 26. Generated text comparison

The transformer output looks like this:
```text
BOWARWCOMILLO:
What I have ease not dlam your have a brother
busin prasurer the with it musdry our tracerve,
And looks sween an ag; raif to know friend-make,
Is of hither friends you, Edie;
Thrus with in take that leave his house me greaths in Cite,
What Ranceid it.
Thousand rest out saburn's delay
```

This is way better than the bigram model from task1. The bigram output was just random character soup, but here it actually looks like Shakespeare dialogues with character names and proper line breaks. It still doesnt make much sense logically, but it's a huge step up.
