"""
Task 3, Part B: Tiny CNN Baseline on CIFAR-10
==============================================
A simple 3-layer CNN to classify CIFAR-10 images.
This serves as a baseline to compare against the Vision Transformer (ViT).

Architecture:
  Conv2d(3→32) → ReLU → MaxPool  (32x32 → 16x16)
  Conv2d(32→64) → ReLU → MaxPool (16x16 → 8x8)
  Conv2d(64→128) → ReLU → MaxPool (8x8 → 4x4)
  Flatten → Linear(128*4*4 → 256) → ReLU → Dropout
  Linear(256 → 10)
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
batch_size = 64
learning_rate = 1e-3
num_epochs = 10
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── Data Loading ────────────────────────────────────────────
# Simple transforms: convert to tensor and normalize using CIFAR-10 mean/std
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 channel means
                         (0.2470, 0.2435, 0.2616)),   # CIFAR-10 channel stds
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
val_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ─── Model ───────────────────────────────────────────────────
class TinyCNN(nn.Module):
    """
    A small CNN for CIFAR-10 classification.
    
    How it works:
    - Each Conv2d layer looks at small 3x3 patches of the image
      and learns to detect features (edges, textures, shapes).
    - MaxPool2d shrinks the image by half each time, so the
      network sees bigger and bigger patterns as we go deeper.
    - Finally, we flatten everything and use Linear layers
      to map the features to 10 class scores.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # 3 convolutional layers that gradually increase channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dimensions

        # Fully connected layers
        # After 3 rounds of pooling: 32→16→8→4, so feature map is 128 × 4 × 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))   # → (B, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))   # → (B, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))   # → (B, 128, 4, 4)
        x = x.flatten(1)                       # → (B, 128*4*4) = (B, 2048)
        x = self.dropout(F.relu(self.fc1(x)))  # → (B, 256)
        return self.fc2(x)                     # → (B, 10)


# ─── Training and Evaluation Functions ───────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    """Train the model for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate the model on a dataset and return loss and accuracy."""
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
    model = TinyCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Track metrics for plotting
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    print(f"\nTraining TinyCNN on CIFAR-10 for {num_epochs} epochs...")
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

    print(f"\nFinal CNN Validation Accuracy: {val_accs[-1]:.2f}%")

    # ─── Save Training Curves ────────────────────────────────
    os.makedirs('task3', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, num_epochs + 1)

    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('CNN: Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label='Train Accuracy')
    ax2.plot(epochs, val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('CNN: Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task3/cnn_curves.png', dpi=150)
    print("Saved training curves to task3/cnn_curves.png")

    # Save val_accs for comparison plot
    torch.save({'val_accs': val_accs, 'train_accs': train_accs}, 'task3/cnn_results.pt')
