import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# Hyperparameters
batch_size = 64
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

# set seed for reproducibility
torch.manual_seed(1337)

# Read the dataset
with open('task1/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenizer dictionaries
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split (90/10)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Batch generation
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # attention scores
        wei = q @ k.transpose(-2, -1) # (B, T, T)
        wei = wei * (k.size(-1) ** -0.5) # scale by 1/sqrt(d_k)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # weighted sum of values
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, n_embd, n_head, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_embd, n_head, head_size, block_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Apply ablation flags if present in the model
        ablation_no_res = getattr(self, 'ablation_no_res', False)
        ablation_no_ln = getattr(self, 'ablation_no_ln', False)
        
        # Pre-norm formulation
        if ablation_no_ln:
            attn_out = self.attn(x)
        else:
            attn_out = self.attn(self.ln1(x))
            
        if ablation_no_res:
            x = attn_out
        else:
            x = x + attn_out
            
        if ablation_no_ln:
            ffn_out = self.ffn(x)
        else:
            ffn_out = self.ffn(self.ln2(x))
            
        if ablation_no_res:
            x = ffn_out
        else:
            x = x + ffn_out
            
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        
        # apply blocks
        x = self.blocks(x) # (B,T,n_embd)
        
        # Ablation check
        ablation_no_ln = getattr(self, 'ablation_no_ln', False)
        if not ablation_no_ln:
            x = self.ln_f(x) # (B,T,n_embd)
            
        # final linear layer
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def run_experiment(variant_name="Baseline"):
    model = TransformerLanguageModel(vocab_size)
    
    # Configure ablations
    if variant_name == "No Residuals":
        for block in model.blocks:
            block.ablation_no_res = True
    elif variant_name == "No LayerNorm":
        for block in model.blocks:
            block.ablation_no_ln = True
        model.ablation_no_ln = True
            
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"[{variant_name}] step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            loss_history.append((iter, losses['train'].item(), losses['val'].item()))

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    return model, loss_history

if __name__ == '__main__':
    print("Running Baseline...")
    model_baseline, history_baseline = run_experiment("Baseline")
    
    # Generate samples for Baseline
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = model_baseline.generate(context, max_new_tokens=300)[0].tolist()
    generated_text = decode(generated_indices)
    
    os.makedirs('task2', exist_ok=True)
    with open('task2/samples.txt', 'w', encoding='utf-8') as f:
        f.write("transformer baseline output:\n")
        f.write(generated_text)
        f.write("\n\n")
    
    print("Running No Residuals Ablation...")
    _, history_no_res = run_experiment("No Residuals")
    
    print("Running No LayerNorm Ablation...")
    _, history_no_ln = run_experiment("No LayerNorm")
    
    # Plotting ablations
    try:
        import matplotlib.pyplot as plt
        iters = [x[0] for x in history_baseline]
        val_baseline = [x[2] for x in history_baseline]
        val_no_res = [x[2] for x in history_no_res]
        val_no_ln = [x[2] for x in history_no_ln]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iters, val_baseline, label='Baseline (Full Transformer)')
        plt.plot(iters, val_no_res, label='No Residuals')
        plt.plot(iters, val_no_ln, label='No LayerNorm')
        plt.xlabel('Iterations')
        plt.ylabel('Validation Loss')
        plt.title('Ablation Study: Residual Connections and LayerNorm')
        plt.legend()
        plt.grid(True)
        plt.savefig('task2/ablation_plot.png')
        print("Saved ablation_plot.png")
    except ImportError:
        print("matplotlib not installed, skipping plot generation.")
