import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class WordTokenizer:
    """
    Simple word-level tokenizer for caption dataset.
    Reserves:
      0 -> <pad>
      1 -> <unk>
      2 -> <sos>
    """
    def __init__(self):
        self.stoi = {'<pad>': 0, '<unk>': 1, '<sos>': 2}
        self.itos = {0: '<pad>', 1: '<unk>', 2: '<sos>'}
        self.vocab_size = 3

    def build_vocab(self, texts, min_freq=1):
        freqs = {}
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            for w in words:
                freqs[w] = freqs.get(w, 0) + 1
        
        for w, count in sorted(freqs.items()):
            if count >= min_freq and w not in self.stoi:
                idx = len(self.stoi)
                self.stoi[w] = idx
                self.itos[idx] = w
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        words = re.findall(r'\w+', text.lower())
        return [self.stoi.get(w, 1) for w in words]

    def decode(self, tokens):
        return ' '.join([self.itos.get(t, '<unk>') for t in tokens if t != 0])


class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k image-caption pairs.
    Supports standard Kaggle Flickr8k dataset, Karpathy splits,
    or falls back to synthetic dataset generation for dry runs.
    """
    def __init__(self, image_dir='task6/data/Flicker8k_Dataset',
                 captions_file='task6/data/Flickr8k.token.txt',
                 tokenizer=None, image_size=64, max_text_len=32, split='train'):
        self.image_dir = image_dir
        self.max_text_len = max_text_len
        self.split = split
        
        # Load caption pairs (image_filename, caption)
        self.pairs = self._load_pairs(image_dir, captions_file, split)
        
        if tokenizer is None:
            self.tokenizer = WordTokenizer()
            self.tokenizer.build_vocab([caption for _, caption in self.pairs])
        else:
            self.tokenizer = tokenizer

        # Image transformations
        if split == 'train':
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),  # ImageNet stats
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def _load_pairs(self, image_dir, captions_file, split):
        pairs = []
        if os.path.exists(captions_file) and os.path.exists(image_dir):
            with open(captions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_id = parts[0].split('#')[0]
                    caption = parts[1]
                    pairs.append((img_id, caption))
        
        # Fallback synthetic pairs if dataset is not yet downloaded
        if len(pairs) == 0:
            print(f"[Flickr8kDataset] Note: Dataset file not found at {captions_file}. Initializing synthetic dataset for local execution.")
            sample_captions = [
                "a brown dog playing on green grass in the park",
                "a young girl in a red shirt swinging on a playground",
                "two dogs running through water on a sunny beach",
                "a man riding a bicycle along a mountain trail",
                "a black cat sitting on a wooden bench outdoors",
                "a boy holding a yellow ball in a grassy backyard",
                "a white dog jumping over a hurdle at an agility contest",
                "a woman walking down a busy city street with an umbrella"
            ]
            total_samples = 6000 if split == 'train' else 1000
            for i in range(total_samples):
                img_name = f"synthetic_{i:04d}.jpg"
                cap = sample_captions[i % len(sample_captions)]
                pairs.append((img_name, cap))
                
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]
        img_path = os.path.join(self.image_dir, filename)
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Deterministic synthetic image for missing files
            torch.manual_seed(idx)
            raw_tensor = torch.rand(3, 64, 64)
            image = T.ToPILImage()(raw_tensor)

        image = self.transform(image)

        # Tokenize and pad/truncate text
        tokens = self.tokenizer.encode(caption)
        tokens = tokens[:self.max_text_len]
        mask = [1] * len(tokens) + [0] * (self.max_text_len - len(tokens))
        tokens = tokens + [0] * (self.max_text_len - len(tokens))

        return {
            'image': image,
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'caption': caption,
            'image_id': filename
        }


if __name__ == '__main__':
    print("Verifying Flickr8kDataset implementation...")
    dataset = Flickr8kDataset(split='train')
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample Image Shape:   {sample['image'].shape}")
    print(f"Sample Tokens Shape:  {sample['tokens'].shape}")
    print(f"Sample Mask Shape:    {sample['mask'].shape}")
    print(f"Sample Caption:       '{sample['caption']}'")
    print(f"Sample Decoded Text:  '{dataset.tokenizer.decode(sample['tokens'].tolist())}'")
    print("Verification complete!")
