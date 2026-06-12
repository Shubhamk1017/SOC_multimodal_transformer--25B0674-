# Task 1: Conceptual Writeup

## 12. Why do we divide attention scores by sqrt(d_k)?
When we compute the dot product between a Query and a Key vector, both of dimension $d_k$, the terms we are summing up add variance to the result. Specifically, if the elements of Q and K have a mean of 0 and variance of 1, their dot product will have a mean of 0 and a variance of $d_k$. As $d_k$ grows (e.g. 64 or 128), the variance of the dot product becomes very large, leading to attention scores with huge magnitudes (both positive and negative). 
When these large scores are fed into the softmax function, the exponentiation pushes the output distribution to become extremely peaked: one value gets almost all the probability mass (close to 1), and the others become effectively 0. The gradients of the softmax function for such extreme inputs approach zero, which effectively halts training (the "vanishing gradient" problem). Dividing the dot products by $\sqrt{d_k}$ scales the variance back down to 1, ensuring the inputs to softmax are within a moderate range where gradients can flow properly.

## 13. Applying the causal mask: Before vs After Softmax
If we applied the mask *after* softmax by simply zeroing out future entries, the resulting attention weights for a given row would no longer sum to 1. The softmax function would have already distributed probability mass to those future tokens, which is then destroyed. If we tried to re-normalize them to sum to 1, we would still be incorporating the existence of those future tokens into the denominator of the softmax for the past tokens, thus leaking information about how many future tokens exist or how large their logits were.
By applying the mask *before* softmax (setting future scores to $-\infty$), $e^{-\infty}$ evaluates precisely to $0$. This means future tokens contribute $0$ to the numerator and $0$ to the denominator. The softmax correctly calculates a probability distribution that sums to 1 *strictly* over the valid past tokens, completely blinding the model to the future.

## 14. Describe Q, K, and V
**Analogy:** Imagine you are in a library searching for a book. You have a search query in mind, like "Python programming" (this is the **Query**). The books on the shelves all have titles and index keywords describing what they are about (these are the **Keys**). When your Query matches a book's Key, you pull that book off the shelf and read its contents (this is the **Value**).
**Linear-algebra view:** For each token in a sequence, we apply three separate linear transformations (matrix multiplications) to its embedding vector.
- **Query ($Q$)**: A projection of the token's embedding that encodes "what I am looking for" from the context.
- **Key ($K$)**: A projection that encodes "what I contain" to be matched against queries from other tokens.
- **Value ($V$)**: A projection that encodes "my actual content/meaning" that will be aggregated into the final representation if my Key matches a Query.

## 15. Why does the single-head model only marginally outperform bigram?
The primary bottleneck is **capacity and depth**. A single attention head with only 32 dimensions and a tiny context length of 8 tokens simply cannot capture deep structural or syntactic patterns of Shakespearean English. It only has one single layer of representation to work with, meaning it can only perform one "hop" of reasoning (e.g. "I am the letter 'e', I should probably follow a 'th'"). Furthermore, there is no feed-forward network (MLP) to process the aggregated information, meaning the model lacks the non-linear capacity to compute complex features after gathering the context.

## 16. Generated Text Comparison
**Bigram Model Samples:**
```text
T:
FIIn:
I:
Tous stht y afofo.
Shirsk$CHaghid.an w wars ng me prst mise wisthayonis:

ORKI t tourek-ppo shiswhun wes bethin anda medis.
P baneinge ors VabyForve, por fave
Thetas t t:
Anousthisolin, a
```

**Single-Head Attention Model Samples:**
```text
Thol anbretherathe ht yo fo omy hi ske srod dianicaky she I:

ING se, sen?

I sonid:
W:
KI ther hakeapis dis.


Tous bet inon:
a mod:
NGiseaneinge ors Vaby whove por fak.

LALas tht:
Yalisttisol ne ar
```

**Qualitative Difference:**
Both models still output largely nonsensical words because they are trained for very few steps on a very small context window. However, the Single-Head Attention model shows slightly better structure. The Bigram model has more disjointed random characters and symbols like `$`, while the Attention model has more consistent spacing, word-like structures (e.g., "Thol", "anbretherathe", "sonid"), and better capitalization patterns following line breaks, indicating it's starting to capture some longer-range context than just the immediately preceding character.
