import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Loss (CLIP Loss).
    
    Computes cross-entropy loss in both directions:
    1. Image -> Text (loss_i2t)
    2. Text -> Image (loss_t2i)
    
    Uses a learnable log-temperature parameter to ensure stable optimization.
    """
    def __init__(self, init_temperature=0.07):
        super().__init__()
        # Parameterize as log(1/tau) for numerical stability during gradient descent.
        # Initial value log(1 / 0.07) ~ 2.6592
        self.log_inv_tau = nn.Parameter(
            torch.tensor([1.0 / init_temperature]).log()
        )

    def forward(self, image_embeds, text_embeds):
        # 1. L2 normalize embeddings onto the unit hypersphere
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # 2. Clamp log-temperature to prevent numerical instability or overflow (log(100) ~ 4.6052)
        log_inv_tau = self.log_inv_tau.clamp(0.0, 4.6052)
        inv_tau = log_inv_tau.exp()

        # 3. Compute cosine similarity matrix scaled by inverse temperature (N, N)
        logits = inv_tau * (image_embeds @ text_embeds.T)

        # 4. Diagonal entries represent matching image-caption pairs
        N = image_embeds.size(0)
        labels = torch.arange(N, device=logits.device)

        # 5. Calculate bidirectional loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2.0
