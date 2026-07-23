"""
Task 6 Applied Stretch Task: Text-to-Image Search Engine
=========================================================
Precomputes embeddings for images and retrieves the top-5 most similar images
for any natural-language text query using cosine similarity.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from task5.clip_model import CLIPStyleModel
    from task6.dataset import Flickr8kDataset
except ImportError:
    import sys
    sys.path.append('.')
    from task5.clip_model import CLIPStyleModel
    from task6.dataset import Flickr8kDataset


class TextToImageSearchEngine:
    def __init__(self, model, dataset, device='cpu'):
        self.model = model.to(device).eval()
        self.dataset = dataset
        self.device = device
        
        # Precompute image embeddings (N_img, D)
        print("Precomputing image embeddings for search index...")
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_embeds = []
        self.image_ids = []
        
        with torch.no_grad():
            for batch in loader:
                imgs = batch['image'].to(device)
                img_e = self.model.encode_image(imgs)
                all_embeds.append(F.normalize(img_e, dim=-1).cpu())
                self.image_ids.extend(batch['image_id'])
                
        self.image_embeddings = torch.cat(all_embeds, dim=0)
        print(f"Index built for {self.image_embeddings.size(0)} images.")

    def search(self, query_text, top_k=5):
        # Tokenize query text
        tokens = self.dataset.tokenizer.encode(query_text)
        max_len = self.dataset.max_text_len
        tokens = tokens[:max_len]
        mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        tokens = tokens + [0] * (max_len - len(tokens))
        
        tok_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        mask_tensor = torch.tensor([mask], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            txt_e = self.model.encode_text(tok_tensor, mask_tensor)
            txt_e = F.normalize(txt_e, dim=-1).cpu()
            
        similarities = (txt_e @ self.image_embeddings.T).squeeze(0)  # (N_img,)
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(self.image_ids)))
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.image_ids[idx], score))
        return results


def run_demo():
    print("=" * 65)
    print("Task 6 Applied Stretch: Text-to-Image Search Engine CLI")
    print("=" * 65)
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dataset = Flickr8kDataset(split='val', image_size=64, max_text_len=32)
    model = CLIPStyleModel(embed_dim=192, projection_dim=128)
    
    ckpt_path = 'task6/best_model.pt'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained weights from {ckpt_path}")
        
    engine = TextToImageSearchEngine(model, dataset, device=device)
    
    queries = [
        "a person riding a bicycle",
        "sunset over water",
        "two dogs playing in the park"
    ]
    
    for q in queries:
        print(f"\nQUERY: '{q}'")
        print("-" * 50)
        results = engine.search(q, top_k=5)
        for rank, (img_id, score) in enumerate(results, 1):
            print(f"  Top-{rank}: {img_id:25s} | Cosine Sim: {score:.4f}")

    print("=" * 65)

if __name__ == '__main__':
    run_demo()
