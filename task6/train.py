import os
import time
import math
import csv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from task5.clip_model import CLIPStyleModel
    from task6.dataset import Flickr8kDataset
except ImportError:
    import sys
    sys.path.append('.')
    from task5.clip_model import CLIPStyleModel
    from task6.dataset import Flickr8kDataset

# Hyperparameter Configuration (Handbook Page 13–14 Specs)
DEFAULT_CONFIG = {
    'batch_size': 128,
    'lr': 5e-4,
    'weight_decay': 0.05,
    'warmup_steps': 500,
    'total_steps': 10000,
    'grad_clip': 1.0,
    'projection_dim': 128,
    'embed_dim': 192,
    'image_size': 64,
    'patch_size': 8,
    'max_text_len': 32,
    'vit_depth': 4,
    'text_depth': 4,
    'n_head': 6,
    'dropout': 0.1,
    'val_every': 200,
}

def get_lr_scheduler(optimizer, warmup_steps, total_steps, base_lr):
    """ Cosine learning rate scheduler with linear warmup """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def evaluate_val_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in val_loader:
        images = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        mask = batch['mask'].to(device)
        
        loss, _, _ = model(images, tokens, mask)
        total_loss += loss.item() * images.size(0)
        count += images.size(0)
    model.train()
    return total_loss / max(1, count)

def train(config=DEFAULT_CONFIG, resume=False, num_steps_limit=None):
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    os.makedirs('task6', exist_ok=True)
    csv_log_path = 'task6/train_log.csv'
    ckpt_path = 'task6/best_model.pt'
    
    # 1. Dataset & DataLoader setup
    train_dataset = Flickr8kDataset(split='train', image_size=config['image_size'], max_text_len=config['max_text_len'])
    val_dataset = Flickr8kDataset(split='val', tokenizer=train_dataset.tokenizer, image_size=config['image_size'], max_text_len=config['max_text_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 2. Model Initialization
    model = CLIPStyleModel(embed_dim=config['embed_dim'], projection_dim=config['projection_dim']).to(device)
    
    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    total_steps = num_steps_limit if num_steps_limit is not None else config['total_steps']
    scheduler = get_lr_scheduler(optimizer, config['warmup_steps'], total_steps, config['lr'])
    
    start_step = 0
    best_val_loss = float('inf')
    
    # Resume capability
    if resume and os.path.exists(ckpt_path):
        print(f"Resuming training from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # CSV Logger Header
    if not os.path.exists(csv_log_path) or not resume:
        with open(csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'train_loss', 'val_loss', 'learning_rate', 'temperature', 'grad_norm', 'step_time_sec'])
            
    print(f"Starting Training: Total Steps = {total_steps} | Batch Size = {config['batch_size']}")
    
    step = start_step
    train_iter = iter(train_loader)
    
    model.train()
    while step < total_steps:
        step += 1
        t0 = time.time()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        images = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        loss, _, _ = model(images, tokens, mask)
        loss.backward()
        
        # Gradient Clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip']).item()
        
        optimizer.step()
        scheduler.step()
        
        t1 = time.time()
        step_time = t1 - t0
        
        current_lr = scheduler.get_last_lr()[0]
        inv_tau = model.loss_fn.log_inv_tau.clamp(0.0, 4.6052).exp().item()
        current_temp = 1.0 / inv_tau
        
        val_loss_str = ""
        val_loss_val = ""
        
        # Validation pass every N steps
        if step % config['val_every'] == 0 or step == total_steps:
            val_loss = evaluate_val_loss(model, val_loader, device)
            val_loss_val = f"{val_loss:.4f}"
            val_loss_str = f" | Val Loss: {val_loss:.4f}"
            
            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'vocab': train_dataset.tokenizer.stoi
                }, ckpt_path)
                print(f"  --> Saved new best checkpoint at step {step} with val_loss={best_val_loss:.4f}")
        
        # Log to CSV
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{loss.item():.4f}", val_loss_val, f"{current_lr:.6e}", f"{current_temp:.4f}", f"{grad_norm:.4f}", f"{step_time:.4f}"])
            
        if step == 1 or step % 50 == 0 or step == total_steps:
            print(f"Step {step:5d}/{total_steps} | Train Loss: {loss.item():.4f}{val_loss_str} | Temp: {current_temp:.4f} | LR: {current_lr:.2e} | GradNorm: {grad_norm:.2f}")

    print("\nTraining Completed Successfully!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000, help='Total training steps')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    train(DEFAULT_CONFIG, resume=args.resume, num_steps_limit=args.steps)
