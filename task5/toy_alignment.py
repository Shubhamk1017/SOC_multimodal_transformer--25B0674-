import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from task5.loss import InfoNCELoss
except ImportError:
    from loss import InfoNCELoss

def run_toy_alignment():
    print("=" * 60)
    print("Task 5: Toy Alignment Experiment")
    print("=" * 60)
    
    torch.manual_seed(42)
    N = 32
    D_in = 192
    D_proj = 128
    
    # 1. Create fixed random image and text features
    I = torch.randn(N, D_in)
    T = torch.randn(N, D_in)
    
    # 2. Learnable linear projections
    P_img = nn.Linear(D_in, D_proj, bias=False)
    P_text = nn.Linear(D_in, D_proj, bias=False)
    
    loss_fn = InfoNCELoss(init_temperature=0.07)
    optimizer = torch.optim.Adam(
        list(P_img.parameters()) + list(P_text.parameters()) + list(loss_fn.parameters()),
        lr=1e-2
    )
    
    initial_expected_loss = math.log(N)
    print(f"Number of pairs N: {N}")
    print(f"Initial expected loss baseline log(32): {initial_expected_loss:.4f}\n")
    
    losses = []
    
    # 3. Train for 500 steps
    for step in range(1, 501):
        optimizer.zero_grad()
        
        img_proj = P_img(I)
        text_proj = P_text(T)
        
        loss = loss_fn(img_proj, text_proj)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step == 1 or step % 100 == 0:
            inv_tau = loss_fn.log_inv_tau.clamp(0.0, 4.6052).exp().item()
            print(f"Step {step:3d}/500 | InfoNCE Loss: {loss.item():.6f} | Temperature: {1.0/inv_tau:.4f}")
    
    # 4. Compute final similarity matrix
    with torch.no_grad():
        norm_img = F.normalize(P_img(I), dim=-1)
        norm_txt = F.normalize(P_text(T), dim=-1)
        sim_matrix = norm_img @ norm_txt.T  # (32, 32)
        
        diag_mask = torch.eye(N, dtype=torch.bool)
        avg_diag = sim_matrix[diag_mask].mean().item()
        avg_off_diag = sim_matrix[~diag_mask].mean().item()
    
    print("\n" + "-" * 60)
    print("Convergence Results:")
    print(f"  Final Loss at Step 500:       {losses[-1]:.6f}")
    print(f"  Average Diagonal Cosine Sim:  {avg_diag:.4f} (High alignment for matching pairs)")
    print(f"  Average Off-Diag Cosine Sim:  {avg_off_diag:.4f} (Low alignment for random pairs)")
    print("-" * 60)
    
    assert losses[-1] < 0.05, f"Toy alignment failed: Final loss {losses[-1]} is higher than 0.05"
    assert avg_diag > 0.80, f"Toy alignment failed: Avg diagonal similarity {avg_diag} is too low"
    print("SUCCESS: Toy alignment verification completed!\n")
    
    return losses, sim_matrix.numpy()

if __name__ == '__main__':
    run_toy_alignment()
