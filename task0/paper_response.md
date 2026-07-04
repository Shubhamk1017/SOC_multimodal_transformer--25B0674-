# Task 0: Paper Response — Attention Is All You Need

## 1. What problem were the authors solving?

So basically before this paper, everyone was using RNNs or LSTMs for seq2seq tasks like translation. But the major issue with those is that they process words one by one. You cant really parallelise that, and they struggle with long sentences because they tend to forget stuff from the start (vanishing gradient problem). The authors solved this by introducing the Transformer, which completely gets rid of recurrence. It uses self attention so everything can look at everything else in parallel. This makes training way faster and you can actually use the GPUs fully.

i think the main insight was realising that you dont even need recurrence to get context, attention alone does a better job.

## 2. What is self-attention computing?

For every word, you calculate three vectors: Query (what the word is looking for), Key (what the word has), and Value (the actual content). To see how much one word should focus on another, you take the dot product of the Query and Key. This gives a similarity score, which you normalise using softmax. Finally, you multiply these weights with the Value vectors to get the output.

The scaling factor 1/sqrt(d_k) is really important here. without it, dot products grow too big and softmax saturates (the gradients become zero and training stops). i actually tried running it without this scaling in task1 and the attention weights just became one-hot vectors, so it definitely matters.

## 3. What does the decoder change?

The decoder has a couple of changes:
- Causal masking: this ensures that when predicting the next word, the model can only look at past words. otherwise it would just cheat by looking ahead.
- Cross-attention: here, the decoder Queries attend to the encoder Keys and Values so the output actually translates the input sentence.

Also, the masking has to be done before softmax (by setting future scores to -inf). If you try to do it after softmax by just setting future values to zero, the weights dont sum to 1 and you leak future info. i got stuck on this for a bit untill my mentor explained why doing it after softmax is wrong.
