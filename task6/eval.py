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


@torch.no_grad()
def evaluate_retrieval(model, val_loader, device='cpu'):
    model.eval()
    model.to(device)
    
    all_image_embeds = []
    all_text_embeds = []
    all_captions = []
    all_image_ids = []
    
    for batch in val_loader:
        images = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        mask = batch['mask'].to(device)
        
        img_e = model.encode_image(images)
        txt_e = model.encode_text(tokens, mask)
        
        all_image_embeds.append(F.normalize(img_e, dim=-1).cpu())
        all_text_embeds.append(F.normalize(txt_e, dim=-1).cpu())
        all_captions.extend(batch['caption'])
        all_image_ids.extend(batch['image_id'])
        
    image_embeds = torch.cat(all_image_embeds, dim=0)  # (N_img, D)
    text_embeds = torch.cat(all_text_embeds, dim=0)    # (N_txt, D)
    
    # Cosine Similarity Matrix (N_img, N_txt)
    similarity = image_embeds @ text_embeds.T
    
    N_img = image_embeds.size(0)
    N_txt = text_embeds.size(0)
    
    # -------------------------------------------------------------
    # 1. Image -> Text Retrieval Recall@K
    # -------------------------------------------------------------
    i2t_r1, i2t_r5, i2t_r10 = 0, 0, 0
    for i in range(N_img):
        # Ground truth matching index (assuming 1 caption per image in evaluation loader)
        gt_idx = i % N_txt
        scores = similarity[i]
        _, topk = torch.topk(scores, k=min(10, N_txt))
        topk_list = topk.tolist()
        
        if gt_idx in topk_list[:1]:
            i2t_r1 += 1
        if gt_idx in topk_list[:5]:
            i2t_r5 += 1
        if gt_idx in topk_list[:10]:
            i2t_r10 += 1

    i2t_r1 = (i2t_r1 / N_img) * 100.0
    i2t_r5 = (i2t_r5 / N_img) * 100.0
    i2t_r10 = (i2t_r10 / N_img) * 100.0

    # -------------------------------------------------------------
    # 2. Text -> Image Retrieval Recall@K
    # -------------------------------------------------------------
    t2i_r1, t2i_r5, t2i_r10 = 0, 0, 0
    similarity_t = similarity.T  # (N_txt, N_img)
    for j in range(N_txt):
        gt_idx = j % N_img
        scores = similarity_t[j]
        _, topk = torch.topk(scores, k=min(10, N_img))
        topk_list = topk.tolist()
        
        if gt_idx in topk_list[:1]:
            t2i_r1 += 1
        if gt_idx in topk_list[:5]:
            t2i_r5 += 1
        if gt_idx in topk_list[:10]:
            t2i_r10 += 1

    t2i_r1 = (t2i_r1 / N_txt) * 100.0
    t2i_r5 = (t2i_r5 / N_txt) * 100.0
    t2i_r10 = (t2i_r10 / N_txt) * 100.0

    metrics = {
        'i2t_R@1': i2t_r1, 'i2t_R@5': i2t_r5, 'i2t_R@10': i2t_r10,
        't2i_R@1': t2i_r1, 't2i_R@5': t2i_r5, 't2i_R@10': t2i_r10
    }
    
    return metrics


def run_evaluation():
    print("=" * 60)
    print("Task 6: Retrieval Evaluation (Recall@K)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt_path = 'task6/best_model.pt'
    
    val_dataset = Flickr8kDataset(split='val', image_size=64, max_text_len=32)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = CLIPStyleModel(embed_dim=192, projection_dim=128)
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Note: No trained checkpoint found. Evaluating with model initialization.")
        
    metrics = evaluate_retrieval(model, val_loader, device=device)
    
    print("\n" + "-" * 45)
    print("Evaluation Results Summary:")
    print("-" * 45)
    print(f" Image -> Text Recall@1:  {metrics['i2t_R@1']:.2f}%")
    print(f" Image -> Text Recall@5:  {metrics['i2t_R@5']:.2f}%")
    print(f" Image -> Text Recall@10: {metrics['i2t_R@10']:.2f}%")
    print("-" * 45)
    print(f" Text -> Image Recall@1:  {metrics['t2i_R@1']:.2f}%")
    print(f" Text -> Image Recall@5:  {metrics['t2i_R@5']:.2f}%")
    print(f" Text -> Image Recall@10: {metrics['t2i_R@10']:.2f}%")
    print("=" * 60)
    
    return metrics

if __name__ == '__main__':
    run_evaluation()
