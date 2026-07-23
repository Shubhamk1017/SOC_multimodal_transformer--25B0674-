import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from task4.cross_attention import CrossAttention
    from task5.clip_model import DummyViTEncoder
except ImportError:
    import sys
    sys.path.append('.')
    from task4.cross_attention import CrossAttention
    from task5.clip_model import DummyViTEncoder


class VLMDecoderBlock(nn.Module):
    """
    Decoder block with Causal Self-Attention + Cross-Attention + FeedForward MLP.
    """
    def __init__(self, n_embd=192, n_head=6, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.self_attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=dropout, batch_first=True)
        
        self.ln_2 = nn.LayerNorm(n_embd)
        self.cross_attn = CrossAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
        
        self.ln_3 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x_text, visual_feats, causal_mask=None):
        # 1. Causal Self-Attention
        norm_x = self.ln_1(x_text)
        sa_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=causal_mask, is_causal=(causal_mask is not None))
        x_text = x_text + sa_out
        
        # 2. Cross-Attention over Visual Feats
        x_text = x_text + self.cross_attn(self.ln_2(x_text), visual_feats)
        
        # 3. FeedForward MLP
        x_text = x_text + self.mlp(self.ln_3(x_text))
        
        return x_text


class MultimodalVLM(nn.Module):
    """
    Full Vision-Language Model (Task 4 Architecture).
    Combines Vision Transformer Encoder with Cross-Attention Text Decoder for multimodal generation/VQA.
    """
    def __init__(self, vocab_size=5000, img_size=64, patch_size=8, embed_dim=192, depth=4, n_head=6, dropout=0.1):
        super().__init__()
        self.vit = DummyViTEncoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, n_head=n_head, dropout=dropout)
        
        self.tok_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, 128, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.decoder_blocks = nn.ModuleList([
            VLMDecoderBlock(n_embd=embed_dim, n_head=n_head, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images, text_tokens, targets=None):
        B, T = text_tokens.shape
        
        # Encode image into patch features (B, N+1, D)
        visual_feats = self.vit.encode(images)
        
        # Text embedding + position embedding
        x_text = self.tok_embed(text_tokens) + self.pos_embed[:, :T]
        x_text = self.dropout(x_text)
        
        # Generate causal mask for text decoder
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=text_tokens.device), diagonal=1)
        
        # Pass through decoder blocks with cross-attention
        for block in self.decoder_blocks:
            x_text = block(x_text, visual_feats, causal_mask=causal_mask)
            
        x_text = self.ln_final(x_text)
        logits = self.lm_head(x_text)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            
        return logits, loss


if __name__ == '__main__':
    print("Testing MultimodalVLM implementation...")
    B, T = 4, 16
    images = torch.randn(B, 3, 64, 64)
    tokens = torch.randint(1, 1000, (B, T))
    targets = torch.randint(1, 1000, (B, T))
    
    vlm = MultimodalVLM(embed_dim=192)
    logits, loss = vlm(images, tokens, targets)
    loss.backward()
    
    print(f"Logits shape: {logits.shape} (Expected: ({B}, {T}, 5000))")
    print(f"Loss value:   {loss.item():.4f}")
    print("MultimodalVLM forward & backward pass verified!")
