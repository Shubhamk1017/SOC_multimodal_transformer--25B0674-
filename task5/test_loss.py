import math
import torch
from loss import InfoNCELoss

def run_sanity_tests():
    print("=" * 60)
    print("Running InfoNCELoss Sanity Tests (Task 5)")
    print("=" * 60)
    
    loss_fn = InfoNCELoss(init_temperature=0.07)
    N = 128
    D = 192

    # -------------------------------------------------------------
    # Test 1: Identical Embeddings (Perfectly Aligned)
    # Expected: Very low loss because correct pairs match perfectly.
    # -------------------------------------------------------------
    same_embeds = torch.randn(N, D)
    loss_test1 = loss_fn(same_embeds, same_embeds).item()
    print(f"[Test 1] Perfectly Aligned Inputs (I == T):")
    print(f"         Loss = {loss_test1:.4f} (Expected: Very close to 0.0)")
    assert loss_test1 < 0.05, f"Test 1 failed: loss {loss_test1} is not small enough."
    print("         STATUS: PASSED ✓\n")

    # -------------------------------------------------------------
    # Test 2: Independent Random Embeddings (Uncorrelated)
    # Expected: Loss approximately log(N) = log(128) ~ 4.8520.
    # Note: Sharper initial temperature (1/0.07 ~ 14.3) adds variance to logits,
    # giving loss in range ~4.85 - 5.40, which converges to log(N) as tau ~ 1.
    # -------------------------------------------------------------
    img_rand = torch.randn(N, D)
    txt_rand = torch.randn(N, D)
    loss_test2 = loss_fn(img_rand, txt_rand).item()
    expected_random_loss = math.log(N)
    
    # Also verify unscaled temperature loss is exact log(N)
    loss_fn_unit_tau = InfoNCELoss(init_temperature=1.0)
    loss_test2_unit = loss_fn_unit_tau(img_rand, txt_rand).item()
    
    print(f"[Test 2] Independent Random Inputs (Uncorrelated):")
    print(f"         Loss (tau=0.07) = {loss_test2:.4f} | Loss (tau=1.0) = {loss_test2_unit:.4f}")
    print(f"         Theoretical log(N) baseline = {expected_random_loss:.4f}")
    assert abs(loss_test2_unit - expected_random_loss) < 0.15, f"Test 2 failed: unscaled loss {loss_test2_unit} deviates from log(N)."
    assert 4.5 < loss_test2 < 5.6, f"Test 2 failed: loss {loss_test2} outside expected range for random inputs."
    print("         STATUS: PASSED ✓\n")

    # -------------------------------------------------------------
    # Test 3: Controlled Mixture (Half Aligned, Half Random)
    # Expected: Loss between perfectly aligned (~0) and random (~4.85)
    # -------------------------------------------------------------
    half_N = N // 2
    shared_vectors = torch.randn(half_N, D)
    
    img_mix = torch.cat([shared_vectors, torch.randn(half_N, D)], dim=0)
    txt_mix = torch.cat([shared_vectors, torch.randn(half_N, D)], dim=0)
    
    loss_test3 = loss_fn(img_mix, txt_mix).item()
    print(f"[Test 3] Controlled Mixture (50% Aligned, 50% Random):")
    print(f"         Loss = {loss_test3:.4f} (Expected: Between 0 and {loss_test2:.4f})")
    assert 0.1 < loss_test3 < loss_test2 - 0.1, f"Test 3 failed: loss {loss_test3} outside expected bounds."
    print("         STATUS: PASSED ✓\n")

    # -------------------------------------------------------------
    # Test 4: Gradient Flow Check
    # Expected: Backward pass populates .grad on log_inv_tau and inputs
    # -------------------------------------------------------------
    img_grad = torch.randn(N, D, requires_grad=True)
    txt_grad = torch.randn(N, D, requires_grad=True)
    
    loss_fn_grad = InfoNCELoss(init_temperature=0.07)
    loss_val = loss_fn_grad(img_grad, txt_grad)
    loss_val.backward()
    
    print("[Test 4] Backward Pass Gradient Check:")
    print(f"         log_inv_tau.grad exists: {loss_fn_grad.log_inv_tau.grad is not None}")
    print(f"         image_embeds.grad exists: {img_grad.grad is not None}")
    print(f"         text_embeds.grad exists:  {txt_grad.grad is not None}")
    
    assert loss_fn_grad.log_inv_tau.grad is not None, "log_inv_tau gradient missing!"
    assert img_grad.grad is not None, "image_embeds gradient missing!"
    assert txt_grad.grad is not None, "text_embeds gradient missing!"
    print("         STATUS: PASSED ✓\n")

    print("=" * 60)
    print("All 4 InfoNCELoss Sanity Tests Passed Successfully!")
    print("=" * 60)

if __name__ == '__main__':
    run_sanity_tests()
