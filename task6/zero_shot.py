"""
Task 6 Applied Stretch Task: Zero-Shot Image Classification
============================================================
Performs zero-shot classification using CLIP text prompt embeddings.
Compares standard single-prompt classification vs. prompt engineering (ensembling).
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


CLASSES = ['dog', 'cat', 'bicycle', 'person', 'car']
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "an image of a {}",
    "a small {} playing outdoors"
]


def encode_class_prompts(model, tokenizer, classes, templates, max_len=32, device='cpu'):
    class_embeds = []
    
    for cls in classes:
        template_embeds = []
        for tpl in templates:
            text = tpl.format(cls)
            tokens = tokenizer.encode(text)[:max_len]
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            tokens = tokens + [0] * (max_len - len(tokens))
            
            tok_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            mask_tensor = torch.tensor([mask], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                txt_e = model.encode_text(tok_tensor, mask_tensor)
                txt_e = F.normalize(txt_e, dim=-1)
                template_embeds.append(txt_e)
                
        # Prompt ensemble average
        avg_embed = torch.stack(template_embeds, dim=0).mean(dim=0)
        avg_embed = F.normalize(avg_embed, dim=-1)
        class_embeds.append(avg_embed)
        
    return torch.cat(class_embeds, dim=0)  # (Num_classes, D)


def classify_image(model, image_tensor, class_embeddings, device='cpu'):
    with torch.no_grad():
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        img_e = model.encode_image(image_tensor.to(device))
        img_e = F.normalize(img_e, dim=-1)
        
        similarities = (img_e @ class_embeddings.T).squeeze(0)
        pred_idx = torch.argmax(similarities).item()
        
    return pred_idx, similarities.tolist()


def run_zero_shot_demo():
    print("=" * 65)
    print("Task 6 Applied Stretch: Zero-Shot Image Classification Demo")
    print("=" * 65)
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dataset = Flickr8kDataset(split='val', image_size=64, max_text_len=32)
    model = CLIPStyleModel(embed_dim=192, projection_dim=128).to(device)
    
    ckpt_path = 'task6/best_model.pt'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained checkpoint from {ckpt_path}")
        
    class_embeds = encode_class_prompts(model, dataset.tokenizer, CLASSES, PROMPT_TEMPLATES, device=device)
    print(f"Encoded zero-shot class embeddings for {len(CLASSES)} target classes using {len(PROMPT_TEMPLATES)} prompt templates.\n")
    
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    batch = next(iter(loader))
    images = batch['image']
    captions = batch['caption']
    
    for i in range(min(5, len(images))):
        pred_idx, scores = classify_image(model, images[i], class_embeds, device=device)
        print(f"Sample [{i+1}] True Caption: '{captions[i]}'")
        print(f"            Predicted Class:  '{CLASSES[pred_idx]}'")
        print(f"            Class Scores:     {dict(zip(CLASSES, [round(s, 4) for s in scores]))}\n")

    print("=" * 65)

if __name__ == '__main__':
    run_zero_shot_demo()
