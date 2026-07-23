import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention Layer.
    
    Allows text decoder tokens (Queries) to attend over visual patch embeddings (Keys & Values).
    - Query shape: (B, T_text, D)
    - Key / Value shape: (B, N_patches, D)
    - Output shape: (B, T_text, D)
    """
    def __init__(self, n_embd=192, n_head=6, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_text, x_visual, visual_mask=None):
        B, T, C = x_text.shape
        _, N, _ = x_visual.shape
        
        # Project Queries from text and Keys/Values from visual patches
        q = self.q_proj(x_text).view(B, T, self.n_head, self.head_dim).transpose(1, 2)     # (B, n_head, T, head_dim)
        k = self.k_proj(x_visual).view(B, N, self.n_head, self.head_dim).transpose(1, 2)   # (B, n_head, N, head_dim)
        v = self.v_proj(x_visual).view(B, N, self.n_head, self.head_dim).transpose(1, 2)   # (B, n_head, N, head_dim)
        
        # Attention scores: (B, n_head, T, N)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        
        if visual_mask is not None:
            attn_scores = attn_scores.masked_fill(visual_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of visual values: (B, n_head, T, head_dim)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)


if __name__ == '__main__':
    print("Testing CrossAttention implementation...")
    B, T_text, N_patches, D = 4, 16, 65, 192
    
    cross_attn = CrossAttention(n_embd=D, n_head=6)
    dummy_text = torch.randn(B, T_text, D)
    dummy_visual = torch.randn(B, N_patches, D)
    
    out = cross_attn(dummy_text, dummy_visual)
    print(f"Output shape: {out.shape} (Expected: ({B}, {T_text}, {D}))")
    assert out.shape == (B, T_text, D)
    print("CrossAttention verification passed!")
