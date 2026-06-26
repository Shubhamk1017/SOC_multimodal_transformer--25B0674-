"""
Run this script to train both CNN and ViT on CIFAR-10 and generate all plots.
Run from the SOC_multimodal_transformer directory:
    python3 task3/train_all.py
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import os
import time

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(1337)

# ─── Data ────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

print("Loading CIFAR-10...")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)


# ─── HELPERS ─────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100 * correct / total


# ═══════════════════════════════════════════════════════════════
# CNN
# ═══════════════════════════════════════════════════════════════
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ═══════════════════════════════════════════════════════════════
# ViT
# ═══════════════════════════════════════════════════════════════
class HeadBi(nn.Module):
    def __init__(self, n_embd, head_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class MHABi(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([HeadBi(n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FFN(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class BlockBi(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        hs = n_embd // n_head
        self.attn = MHABi(n_embd, n_head, hs, dropout)
        self.ffn = FFN(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 n_embd=192, n_head=6, n_layer=6, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, n_embd, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, n_embd))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([BlockBi(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.dropout(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════
os.makedirs('task3', exist_ok=True)
criterion = nn.CrossEntropyLoss()

# --- CNN Training ---
print("\n" + "="*55)
print("TRAINING CNN (10 epochs)")
print("="*55)
cnn = TinyCNN().to(device)
cnn_opt = torch.optim.AdamW(cnn.parameters(), lr=1e-3)
cnn_loader_tr = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
cnn_loader_va = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

cnn_train_accs, cnn_val_accs = [], []
t0 = time.time()
for ep in range(1, 11):
    tl, ta = train_one_epoch(cnn, cnn_loader_tr, cnn_opt, criterion)
    vl, va = evaluate(cnn, cnn_loader_va, criterion)
    cnn_train_accs.append(ta)
    cnn_val_accs.append(va)
    print(f"  Epoch {ep:2d} | train {ta:.1f}% | val {va:.1f}% | loss {vl:.4f}")
print(f"CNN done in {time.time()-t0:.0f}s, final val acc: {cnn_val_accs[-1]:.1f}%")

# save CNN curves
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), cnn_train_accs, 'o-', label='Train')
ax.plot(range(1, 11), cnn_val_accs, 's-', label='Val')
ax.set(xlabel='Epoch', ylabel='Accuracy (%)', title='CNN: Training & Validation Accuracy')
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('task3/cnn_curves.png', dpi=150); plt.close()

# --- ViT Training ---
print("\n" + "="*55)
print("TRAINING ViT (30 epochs)")
print("="*55)
vit = ViT(n_embd=192, n_head=6, n_layer=6, dropout=0.1).to(device)
n_params = sum(p.numel() for p in vit.parameters())
print(f"  ViT params: {n_params:,}")
vit_opt = torch.optim.AdamW(vit.parameters(), lr=3e-4)
vit_loader_tr = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
vit_loader_va = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

vit_train_accs, vit_val_accs = [], []
t0 = time.time()
for ep in range(1, 31):
    tl, ta = train_one_epoch(vit, vit_loader_tr, vit_opt, criterion)
    vl, va = evaluate(vit, vit_loader_va, criterion)
    vit_train_accs.append(ta)
    vit_val_accs.append(va)
    print(f"  Epoch {ep:2d} | train {ta:.1f}% | val {va:.1f}% | loss {vl:.4f}")
print(f"ViT done in {time.time()-t0:.0f}s, final val acc: {vit_val_accs[-1]:.1f}%")

# save ViT curves
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 31), vit_train_accs, 'o-', label='Train', markersize=3)
ax.plot(range(1, 31), vit_val_accs, 's-', label='Val', markersize=3)
ax.set(xlabel='Epoch', ylabel='Accuracy (%)', title='ViT: Training & Validation Accuracy')
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('task3/vit_curves.png', dpi=150); plt.close()

# --- Comparison Plot ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cnn_val_accs, 'o-', label=f'CNN (final: {cnn_val_accs[-1]:.1f}%)', linewidth=2, markersize=6)
plt.plot(range(1, 31), vit_val_accs, 's-', label=f'ViT (final: {vit_val_accs[-1]:.1f}%)', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.title('CIFAR-10: CNN vs ViT Validation Accuracy', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task3/comparison_plot.png', dpi=150)
plt.close()

print("\n✅ All done! Saved:")
print("   task3/cnn_curves.png")
print("   task3/vit_curves.png")
print("   task3/comparison_plot.png")
