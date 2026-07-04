# Task 0: Paper Response — Attention Is All You Need

## 1. What problem were the authors solving?

The standard approach before Transformers was using RNNs/LSTMs for sequence tasks like translation. The problem is these models process tokens one by one — you can't parallelise that, and they tend to lose track of things from early in the sequence (vanishing gradient issue). The Transformer ditches recurrence completely and uses attention to let every position look at every other position at once. This makes training much faster since you can actually use your GPU properly.

I think the key insight was realising that the sequential processing wasn't actually necessary — you could get the same (or better) contextual understanding through attention alone.

## 2. What is self-attention computing?

For each token, you compute three things: a Query vector (roughly "what am I looking for"), a Key vector ("what do I contain"), and a Value vector ("what information do I carry"). You take dot products between one token's Query and all other tokens' Keys to get relevance scores, normalise with softmax, and use those as weights to combine the Values.

The scaling by 1/sqrt(d_k) is important — without it the dot products grow too large and softmax saturates (I saw this happen when I removed the scaling in my code, the attention weights became basically one-hot).

## 3. What does the decoder change?

Two main things:
- A causal mask so position i can only attend to positions ≤ i (otherwise the model would "cheat" by looking at future tokens during training)
- A cross-attention layer where the decoder queries attend to the encoder's keys and values — this is how the output stays grounded in the input

The masking has to happen before softmax (set future entries to -inf), not after. If you zero things out after softmax, the probabilities don't sum to 1 anymore and you get weird gradient issues. I got confused about this initially and my mentor had to explain it.
