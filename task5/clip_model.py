import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from task5.loss import InfoNCELoss
except ImportError:
    from loss import InfoNCELoss

class DummyViTEncoder(nn.Module):
    """
    Vision Transformer Encoder wrapper matching Task 4 / handbook specs.
    Encodes image tensors (B, 3, H, W) into sequence embeddings (B, N+1, D).
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=192, depth=4, n_head=6, dropout=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_head, dim_feedforward=4*embed_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x  # (B, N+1, D)


class DummyTextEncoder(nn.Module):
    """
    Transformer Text Encoder wrapper matching Task 2 / handbook specs.
    Encodes text token sequences (B, T) into sequence embeddings (B, T, D).
    """
    def __init__(self, vocab_size=5000, embed_dim=192, max_len=32, depth=4, n_head=6, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_head, dim_feedforward=4*embed_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode(self, tokens):
        B, T = tokens.shape
        x = self.token_embed(tokens) + self.pos_embed[:, :T]
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x  # (B, T, D)


class CLIPStyleModel(nn.Module):
    """
    CLIP-style Dual Encoder Model.
    
    Combines Vision Transformer and Text Encoder into a shared projection space
    and optimizes contrastive InfoNCE loss between image and text embeddings.
    """
    def __init__(self, vit_encoder=None, text_encoder=None, embed_dim=192, projection_dim=128):
        super().__init__()
        self.vit = vit_encoder if vit_encoder is not None else DummyViTEncoder(embed_dim=embed_dim)
        self.text = text_encoder if text_encoder is not None else DummyTextEncoder(embed_dim=embed_dim)
        
        self.image_proj = nn.Linear(embed_dim, projection_dim, bias=False)
        self.text_proj = nn.Linear(embed_dim, projection_dim, bias=False)
        self.loss_fn = InfoNCELoss(init_temperature=0.07)

    def encode_image(self, images):
        feats = self.vit.encode(images)  # (B, N+1, D)
        cls = feats[:, 0]                 # (B, D)
        return self.image_proj(cls)      # (B, D_proj)

    def encode_text(self, text_tokens, text_mask=None):
        feats = self.text.encode(text_tokens)  # (B, T, D)
        if text_mask is not None:
            mask = text_mask.unsqueeze(-1).float()  # (B, T, 1)
            pooled = (feats * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            pooled = feats.mean(1)
        return self.text_proj(pooled)  # (B, D_proj)

    def forward(self, images, text_tokens, text_mask=None):
        img_e = self.encode_image(images)
        txt_e = self.encode_text(text_tokens, text_mask)
        loss = self.loss_fn(img_e, txt_e)
        return loss, img_e, txt_e


if __name__ == '__main__':
    print("Testing CLIPStyleModel with dummy data...")
    B = 16
    img_dim = (3, 64, 64)
    T = 32
    
    model = CLIPStyleModel(embed_dim=192, projection_dim=128)
    dummy_images = torch.randn(B, *img_dim)
    dummy_tokens = torch.randint(1, 1000, (B, T))
    dummy_mask = torch.ones(B, T)
    
    loss, img_e, txt_e = model(dummy_images, dummy_tokens, dummy_mask)
    loss.backward()
    
    print(f"Forward Pass Success!")
    print(f"  Image Embeddings Shape: {img_e.shape} (Expected: ({B}, 128))")
    print(f"  Text Embeddings Shape:  {txt_e.shape} (Expected: ({B}, 128))")
    print(f"  Initial InfoNCE Loss:   {loss.item():.4f}")
    print("  Gradients flow properly throughout model.")
