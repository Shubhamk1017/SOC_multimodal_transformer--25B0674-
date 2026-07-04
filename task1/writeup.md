# Task 1: Conceptual Writeup

## 12. Why divide by sqrt(d_k)?

When you take the dot product of Q and K vectors, the variance grows with the dimension d_k. So if d_k is large (like 64 or 128), the scores can get really huge. When you pass these large numbers to softmax, it acts like a hard max — one value gets a probability of almost 1.0 and everything else becomes 0. The gradients for softmax in this region are basically zero, which means the model stops learning (vanishing gradients).

Dividing by sqrt(d_k) scales the variance back down to 1. This keeps the softmax inputs in a normal range where gradients can actually flow. i verified this by printing the attention weights during training — without scaling, they just became binary one-hot vectors after a few steps.

## 13. Causal mask: before or after softmax?

It definitely has to be before softmax. If you apply it after (like just setting future values to 0), the weights won't sum to 1 anymore because softmax already distributed probability to the future tokens. Even if you try to normalise them again, you've already let the future tokens affect the denominator of the softmax, which leaks info.

By setting future positions to -inf before softmax, exp(-inf) becomes exactly 0. This means future tokens contribute nothing to the numerator or denominator. Softmax then calculates a proper probability distribution that sums to 1 over only the past tokens.

## 14. Q, K, V — what are they?

i think of it like searching through my lecture notes. My query (Q) is the specific topic i want to find, like "gradient descent formula". The headings of different pages are the Keys (K). The actual text on those pages is the Value (V). i match my query against all the keys to see which page is relevant, and then i read the values (the text) from the most relevant pages.

In the code, Q, K, and V are just linear projections of the input embeddings. Since we have multiple heads, each head learns different weights so they can focus on different types of relationships (like some heads looking at grammar and others at meaning).
