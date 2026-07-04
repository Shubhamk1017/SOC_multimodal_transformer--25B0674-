# Task 3: Conceptual Writeup

## 7. CNN vs ViT on CIFAR-10

My CNN got around 68-70% accuracy in 10 epochs, and the ViT got to roughly the same range but needed 30 epochs. Pretty close overall.

The CNN has a natural advantage here because convolutions encode two assumptions that happen to be true for images: nearby pixels matter more than distant ones (locality), and a feature should be recognised regardless of where it appears (translation equivariance). These biases mean the CNN doesn't need to "discover" basic spatial relationships from scratch.

ViT doesn't have any of that. It treats patches as a flat sequence — patch (0,0) and patch (7,7) are just two tokens with no built-in notion of "nearness". Given enough data (millions of images) ViT can learn spatial structure and even beat CNNs, but CIFAR-10 only has 50k samples. Not enough for the ViT to figure out what the CNN already "knows" architecturally.

This is basically the inductive bias tradeoff — more built-in assumptions = less data needed, but also less flexibility.

## 8. Why patching? Why not feed individual pixels?

Computation. Self-attention is O(T²) in sequence length. For a 32×32 image, feeding pixels directly gives T = 3072, meaning ~9.4M pairwise scores. For a 224×224 image that balloons to billions — totally impractical.

With 4×4 patches on CIFAR-10, T = 64 → only 4096 pairwise scores. Much more reasonable.

There's also a representation argument: a single pixel value is basically meaningless on its own. A 4×4 patch at least captures a tiny texture or edge, which is a more useful unit for the model to work with. It's similar to how NLP tokenisers use subwords ("un", "break", "able") instead of individual characters.

## 9. What's the CLS token for?

It's a learnable vector prepended to the patch sequence. It starts with zero information about the image and gradually absorbs global context as it attends to all the patch tokens through the transformer blocks. The final classifier reads from CLS because it ends up being a summary of the whole image.

Why not just average all patch tokens instead? Averaging weights every patch equally, but some patches (like background sky) are less informative than others (like the actual object). The CLS token can learn to attend more to the important patches through attention. It's basically a learned pooling mechanism.

That said, I've seen some papers that just use mean pooling over patches and get similar results, so it's not clear-cut. For this task I followed the standard ViT approach with CLS.
