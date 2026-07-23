import torch
try:
    from task4.vlm import MultimodalVLM
except ImportError:
    from vlm import MultimodalVLM

def test_vlm_pipeline():
    print("=" * 60)
    print("Running Task 4 Multimodal VLM Pipeline Verification")
    print("=" * 60)
    
    B = 8
    img_shape = (3, 64, 64)
    seq_len = 24
    vocab_size = 5000
    
    model = MultimodalVLM(vocab_size=vocab_size, embed_dim=192, depth=4, n_head=6)
    
    dummy_images = torch.randn(B, *img_shape)
    dummy_tokens = torch.randint(1, vocab_size, (B, seq_len))
    dummy_targets = torch.randint(1, vocab_size, (B, seq_len))
    
    logits, loss = model(dummy_images, dummy_tokens, dummy_targets)
    loss.backward()
    
    print(f"[Check 1] Logits Shape: {logits.shape} (Expected ({B}, {seq_len}, {vocab_size}))")
    assert logits.shape == (B, seq_len, vocab_size), "Logits shape mismatch"
    print("          STATUS: PASSED ✓\n")
    
    print(f"[Check 2] Cross-Entropy Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss must be positive"
    print("          STATUS: PASSED ✓\n")
    
    print("[Check 3] Gradient Verification:")
    print(f"          ViT Patch Embed Grad: {model.vit.patch_embed.weight.grad is not None}")
    print(f"          LM Head Grad:        {model.lm_head.weight.grad is not None}")
    assert model.vit.patch_embed.weight.grad is not None, "ViT patch embedding grad missing"
    assert model.lm_head.weight.grad is not None, "LM head grad missing"
    print("          STATUS: PASSED ✓\n")
    
    print("=" * 60)
    print("All Task 4 VLM Verification Tests Passed!")
    print("=" * 60)

if __name__ == '__main__':
    test_vlm_pipeline()
