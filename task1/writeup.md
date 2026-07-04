# Task 1: Conceptual Writeup

## 12. Why divide by sqrt(d_k)?

The dot product between Q and K vectors has variance proportional to d_k. So if d_k = 64, the raw scores can get pretty large in magnitude. When you feed large numbers into softmax, it basically becomes a hard argmax — one element gets ~1.0 and everything else gets ~0.0. Gradients in that regime are basically zero, so the model stops learning.

Dividing by sqrt(d_k) scales the variance back to ~1, keeping softmax in the "useful" range where it actually produces meaningful gradients. I checked this by printing attention weights with and without scaling — without it they looked almost binary after a few hundred steps.

## 13. Causal mask: before or after softmax?

Has to be before. If you apply it after (just zeroing out entries), the remaining weights don't add up to 1 because softmax already distributed probability to those future tokens. Even renormalising doesn't fix it because the future token logits already affected the denominator.

Setting them to -inf before softmax works cleanly: exp(-inf) = 0, so future tokens contribute zero to both numerator and denominator. The distribution sums to 1 over only the valid (past) positions.

## 14. Q, K, V — what are they?

My way of thinking about it: you're searching your notes before a viva. Your question ("how does backprop work?") is the Query. Each page of your notes has a heading — those are the Keys. The actual content on each page is the Value. You match your question against each heading, figure out which pages are most relevant, then read the content from those pages weighted by relevance.

In the model, Q/K/V are linear projections of the input embeddings. Each head learns different projection matrices, so different heads can focus on different kinds of relationships (one might track position, another might track meaning).
