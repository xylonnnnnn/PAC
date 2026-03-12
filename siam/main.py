import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_samples(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Каталог не найден: {data_dir}')
    samples = {}
    for d in sorted([x for x in data_dir.iterdir() if x.is_dir() and x.name.startswith('s')], key=lambda x: int(x.name[1:])):
        label = int(d.name[1:]) - 1
        imgs = sorted(sum([list(d.glob(ext)) for ext in ('*.pgm', '*.png', '*.jpg', '*.jpeg')], []))
        if imgs:
            samples[label] = imgs
    if not samples:
        raise RuntimeError('Изображения не найдены')
    train, test = [], []
    for label, imgs in samples.items():
        train += [(p, label) for p in imgs[:8]]
        test += [(p, label) for p in imgs[8:]]
    return train, test


class FacePairs(Dataset):
    def __init__(self, samples, tfm, size, seed=42):
        self.samples = samples
        self.tfm = tfm
        self.size = size
        self.rng = random.Random(seed)
        self.by_class = {}
        for path, label in samples:
            self.by_class.setdefault(label, []).append(path)
        self.labels = list(self.by_class)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        same = i % 2 == 0
        if same:
            label = self.rng.choice(self.labels)
            p1, p2 = self.rng.sample(self.by_class[label], 2)
            y = 1.0
        else:
            a, b = self.rng.sample(self.labels, 2)
            p1 = self.rng.choice(self.by_class[a])
            p2 = self.rng.choice(self.by_class[b])
            y = 0.0
        x1 = self.tfm(Image.open(p1).convert('L'))
        x2 = self.tfm(Image.open(p2).convert('L'))
        return x1, x2, torch.tensor([y], dtype=torch.float32), str(p1), str(p2)


class FaceImages(Dataset):
    def __init__(self, samples, tfm):
        self.samples = samples
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.tfm(Image.open(path).convert('L')), label, str(path)


class SiameseNet(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 14 * 11, 128), nn.ReLU(), nn.Linear(128, emb_dim))

    def encode(self, x):
        return F.normalize(self.head(self.features(x)), p=2, dim=1)

    def forward(self, x1, x2):
        return self.encode(x1), self.encode(x2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        d = F.pairwise_distance(z1, z2)
        y = y.view(-1)
        return (y * d.pow(2) + (1 - y) * torch.clamp(self.margin - d, min=0).pow(2)).mean(), d


@torch.no_grad()
def evaluate(model, loader, device, mode, margin):
    model.eval()
    dists, labels, pairs = [], [], []
    for x1, x2, y, p1, p2 in loader:
        z1, z2 = model(x1.to(device), x2.to(device))
        d = F.pairwise_distance(z1, z2).cpu().numpy()
        dists.extend(d.tolist())
        labels.extend(y.view(-1).numpy().tolist())
        pairs.extend(list(zip(p1, p2)))
    dists, labels = np.array(dists), np.array(labels)
    preds = (torch.sigmoid(torch.tensor(-dists)).numpy() >= 0.5).astype(np.float32) if mode == 'bce' else (dists < margin / 2).astype(np.float32)
    return {
        'accuracy': float((preds == labels).mean()),
        'same_distance_mean': float(dists[labels == 1].mean()),
        'diff_distance_mean': float(dists[labels == 0].mean()),
        'dists': dists,
        'labels': labels,
        'pairs': pairs,
    }


@torch.no_grad()
def embed(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y, _ in loader:
        feats.append(model.encode(x.to(device)).cpu().numpy())
        labels.extend(y.numpy().tolist())
    return np.concatenate(feats), np.array(labels)


def train_epoch(model, loader, opt, device, mode, margin):
    model.train()
    bce = nn.BCELoss()
    contrast = ContrastiveLoss(margin)
    total = 0
    for x1, x2, y, _, _ in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        z1, z2 = model(x1, x2)
        d = F.pairwise_distance(z1, z2).unsqueeze(1)
        loss = bce(torch.sigmoid(-d), y) if mode == 'bce' else contrast(z1, z2, y)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(x1)
    return total / len(loader.dataset)


def plot_tsne(embs, labels, path, title):
    pts = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=max(5, min(30, len(embs) - 1))).fit_transform(embs)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab20', s=40)
    plt.legend(*sc.legend_elements(num=10), title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pairs(info, path, title, n=6):
    items = []
    for cls in [1, 0]:
        for (p1, p2), d, y in zip(info['pairs'], info['dists'], info['labels']):
            if y == cls:
                items.append((p1, p2, d, int(y)))
                if len(items) >= (n // 2 if cls == 1 else n):
                    break
    fig, ax = plt.subplots(len(items), 2, figsize=(6, 2.5 * len(items)))
    if len(items) == 1:
        ax = np.array([ax])
    for i, (p1, p2, d, y) in enumerate(items):
        ax[i, 0].imshow(np.array(Image.open(p1).convert('L')), cmap='gray')
        ax[i, 1].imshow(np.array(Image.open(p2).convert('L')), cmap='gray')
        ax[i, 0].set_title(f'label={y}')
        ax[i, 1].set_title(f'distance={d:.4f}')
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def run(mode, train_loader, test_pair_loader, test_img_loader, device, epochs, lr, margin, out_dir):
    model = SiameseNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, opt, device, mode, margin)
        info = evaluate(model, test_pair_loader, device, mode, margin)
        history.append({
            'epoch': epoch,
            'train_loss': loss,
            'test_accuracy': info['accuracy'],
            'same_distance_mean': info['same_distance_mean'],
            'diff_distance_mean': info['diff_distance_mean'],
        })
        print(f'[{mode}] epoch {epoch}/{epochs} loss={loss:.4f} acc={info["accuracy"]:.4f} same={info["same_distance_mean"]:.4f} diff={info["diff_distance_mean"]:.4f}')
    info = evaluate(model, test_pair_loader, device, mode, margin)
    embs, labels = embed(model, test_img_loader, device)
    plot_tsne(embs, labels, out_dir / f'tsne_{mode}.png', f't-SNE ({mode})')
    plot_pairs(info, out_dir / f'inference_{mode}.png', f'Inference ({mode})')
    return history, info


def main():
    epochs = 15
    batch_size = 32
    lr = 1e-3
    margin = 1.0
    seed = 42

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.Resize((112, 92)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train, test = read_samples('att_faces')
    train_loader = DataLoader(FacePairs(train, tfm, 4000, seed), batch_size=batch_size, shuffle=True)
    test_pair_loader = DataLoader(FacePairs(test, tfm, 800, seed + 1), batch_size=batch_size)
    test_img_loader = DataLoader(FaceImages(test, tfm), batch_size=batch_size)

    hist_bce, info_bce = run('bce', train_loader, test_pair_loader, test_img_loader, device, epochs, lr, margin, out_dir)
    hist_con, info_con = run('contrastive', train_loader, test_pair_loader, test_img_loader, device, epochs, lr, margin, out_dir)

    summary = {
        'device': str(device),
        'train_samples': len(train),
        'test_samples': len(test),
        'bce': {'history': hist_bce, 'final_accuracy': info_bce['accuracy'], 'same_distance_mean': info_bce['same_distance_mean'], 'diff_distance_mean': info_bce['diff_distance_mean']},
        'contrastive': {'history': hist_con, 'final_accuracy': info_con['accuracy'], 'same_distance_mean': info_con['same_distance_mean'], 'diff_distance_mean': info_con['diff_distance_mean']},
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for path in sorted(out_dir.iterdir()):
        print(path)


if __name__ == '__main__':
    main()