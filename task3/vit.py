"""
Task 3, Part E: Vision Transformer (ViT) on CIFAR-10
=====================================================
A Vision Transformer built from scratch that classifies CIFAR-10 images.

Key idea: cut an image into patches, flatten each patch into a vector,
add position embeddings, and feed the sequence to a standard Transformer.

Architecture:
  Image → Patch Embedding (Conv2d trick) → Add [CLS] token → Add Position Embeddings
  → N Transformer Blocks (bidirectional self-attention + MLP) → LayerNorm
  → Read [CLS] token → Linear classifier → 10 classes
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# ─── Hyperparameters ─────────────────────────────────────────
# These match what the handbook recommends
img_size = 32         # CIFAR-10 images are 32x32
patch_size = 4        # each patch is 4x4 pixels → 8x8 = 64 patches total
in_chans = 3          # RGB images
num_classes = 10      # CIFAR-10 has 10 classes
n_embd = 192          # embedding dimension (each patch becomes a 192-dim vector)
n_head = 6            # number of attention heads
n_layer = 6           # number of transformer blocks
dropout_rate = 0.1
batch_size = 128
learning_rate = 3e-4
num_epochs = 30

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(1337)

# ─── Data Loading ────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),       # simple augmentation
    transforms.RandomCrop(32, padding=4),    # randomly shift the image a bit
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
val_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_val
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# ─── Transformer Building Blocks (Bidirectional) ─────────────
# These are the same blocks from Task 2, but WITHOUT the causal mask.
# In ViT, every patch can attend to every other patch — there's no
# "past" and "future" like in text generation.

class HeadBidirectional(nn.Module):
    """
    One head of bidirectional self-attention.
    
    Same as Task 2's Head, but WITHOUT the causal (triangular) mask.
    Every patch can freely look at every other patch.
    """
    def __init__(self, n_embd, head_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1)         # (B, T, T)
        wei = wei * (k.size(-1) ** -0.5)      # scale by 1/sqrt(d_k)
        # NO CAUSAL MASK here — this is the key difference from Task 2!
        wei = F.softmax(wei, dim=-1)           # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttentionBidirectional(nn.Module):
    """Multiple heads of bidirectional self-attention in parallel."""
    def __init__(self, n_embd, n_head, head_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            HeadBidirectional(n_embd, head_size, dropout)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation.
    
    Same idea as Task 2, but we use GELU instead of ReLU
    (GELU is the standard activation in ViT and modern transformers).
    """
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),     # expand to 4x
            nn.GELU(),                           # smooth activation
            nn.Linear(4 * n_embd, n_embd),      # project back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BlockBidirectional(nn.Module):
    """
    Transformer block with BIDIRECTIONAL attention.
    
    Structure (pre-norm, same as Task 2):
      x = x + Attention(LayerNorm(x))
      x = x + MLP(LayerNorm(x))
    """
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttentionBidirectional(n_embd, n_head, head_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention with residual
        x = x + self.ffn(self.ln2(x))    # MLP with residual
        return x


# ─── Vision Transformer ─────────────────────────────────────
class ViT(nn.Module):
    """
    Vision Transformer for image classification.
    
    Step by step:
    1. Split image into patches using Conv2d (a clever trick!)
    2. Prepend a special [CLS] token to the sequence
    3. Add learnable position embeddings
    4. Pass through N transformer blocks
    5. Read the [CLS] token's final representation
    6. Classify with a linear layer
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 num_classes=10, n_embd=192, n_head=6,
                 n_layer=6, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2  # 8*8 = 64 patches

        # Patch embedding: Conv2d with kernel_size=patch_size, stride=patch_size
        # This is equivalent to: flatten each patch → linear projection
        # But Conv2d does both in one efficient operation!
        self.patch_embed = nn.Conv2d(
            in_chans, n_embd,
            kernel_size=patch_size, stride=patch_size
        )

        # [CLS] token: a learnable vector prepended to the sequence
        # After all transformer blocks, the classifier reads from this token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))

        # Position embeddings: tell the model where each patch came from
        # +1 for the CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, n_embd))

        self.dropout = nn.Dropout(dropout)

        # Stack of transformer blocks (bidirectional — no causal mask!)
        self.blocks = nn.ModuleList([
            BlockBidirectional(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(n_embd)

        # Classification head: maps CLS token output → class scores
        self.head = nn.Linear(n_embd, num_classes)

        # Initialize special parameters
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)

        # Step 1: Patchify the image
        # Conv2d: (B, 3, 32, 32) → (B, n_embd, 8, 8)
        x = self.patch_embed(x)
        # Flatten spatial dims and transpose: (B, n_embd, 8, 8) → (B, 64, n_embd)
        x = x.flatten(2).transpose(1, 2)

        # Step 2: Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (1, 1, n_embd) → (B, 1, n_embd)
        x = torch.cat([cls, x], dim=1)          # (B, 65, n_embd)

        # Step 3: Add position embeddings
        x = x + self.pos_embed                  # (B, 65, n_embd)
        x = self.dropout(x)

        # Step 4: Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Step 5: Read from [CLS] token (first position)
        cls_final = x[:, 0]  # (B, n_embd)

        # Step 6: Classify
        return self.head(cls_final)  # (B, num_classes)


# ─── Training and Evaluation ─────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100 * correct / total


# ─── Main Training Loop ─────────────────────────────────────
if __name__ == '__main__':
    model = ViT(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        num_classes=num_classes, n_embd=n_embd, n_head=n_head,
        n_layer=n_layer, dropout=dropout_rate
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    print(f"\nTraining ViT on CIFAR-10 for {num_epochs} epochs...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>6.2f}%")

    print(f"\nFinal ViT Validation Accuracy: {val_accs[-1]:.2f}%")

    # ─── Save Training Curves ────────────────────────────────
    os.makedirs('task3', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, num_epochs + 1)

    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('ViT: Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label='Train Accuracy')
    ax2.plot(epochs, val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('ViT: Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task3/vit_curves.png', dpi=150)
    print("Saved training curves to task3/vit_curves.png")

    # Save results for comparison plot
    torch.save({'val_accs': val_accs, 'train_accs': train_accs}, 'task3/vit_results.pt')

    # ─── Generate Comparison Plot ────────────────────────────
    # Try to load CNN results and make comparison plot
    try:
        cnn_data = torch.load('task3/cnn_results.pt', weights_only=True)
        cnn_val_accs = cnn_data['val_accs']

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cnn_val_accs) + 1), cnn_val_accs,
                 'o-', label=f'CNN (final: {cnn_val_accs[-1]:.1f}%)', linewidth=2)
        plt.plot(range(1, len(val_accs) + 1), val_accs,
                 's-', label=f'ViT (final: {val_accs[-1]:.1f}%)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('CIFAR-10: CNN vs ViT Validation Accuracy')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('task3/comparison_plot.png', dpi=150)
        print("Saved comparison plot to task3/comparison_plot.png")
    except FileNotFoundError:
        print("CNN results not found. Run cnn_baseline.py first to generate comparison_plot.png")
