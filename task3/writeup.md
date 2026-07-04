# Task 3: Conceptual Writeup

## 7. CNN vs ViT on CIFAR-10

So my CNN got around 68-70% validation accuracy in just 10 epochs, but the ViT needed 30 epochs to reach about the same (65-72%). 

The reason CNN does so well quickly is because it has "inductive bias" built for images. Basically, it assumes that nearby pixels are related (locality) and that a feature is the same no matter where it is in the image (translation equivariance). So the CNN doesn't have to learn these basic rules from scratch.

But ViT doesn't know anything about images. To the transformer, the patches are just a list of tokens. It doesn't know that patch 1 is next to patch 2. It has to learn all of this from the data. On huge datasets (like ImageNet with millions of images), ViT works great, but CIFAR-10 is too small (only 50k images) for it to learn everything from scratch effectively.

## 8. Why patching? Why not feed individual pixels?

Mostly because of computational limits. Self-attention complexity is O(T²) where T is sequence length. If we feed pixels directly for a 32x32 image, T is 3072, which means we need around 9.4 million operations for attention. For a larger image (like 224x224), it would be billions of operations, which is just impossible.
By using 4x4 patches, T becomes 64, so we only need 4096 operations. That is way more manageable.

Also, single pixels don't really mean anything on their own. A patch has textures and edges, which makes more sense as a basic unit, kind of like subwords in NLP tokenizers.

## 9. What's the CLS token for?

The CLS token is a dummy token prepended to the start of the sequence. It doesn't represent any patch initially, but as it passes through the layers, it attends to all other patches and aggregates global info about the image. The classifier then reads from this token.

if you just average all the patches, you treat background noise (like sky or grass) the same as the main object. The CLS token learns to focus only on the important parts through attention. i think some papers use average pooling too and get decent results, but CLS is the standard way ViT does it.
