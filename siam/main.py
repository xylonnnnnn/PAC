import argparse
import json
import math
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


class PairDataset(Dataset):
    def __init__(self, samples, transform=None, pairs_per_epoch=4000, seed=42):
        self.samples = samples
        self.transform = transform
        self.pairs_per_epoch = pairs_per_epoch
        self.seed = seed
        self.class_to_paths = {}
        for path, label in samples:
            self.class_to_paths.setdefault(label, []).append(path)
        self.labels = sorted(self.class_to_paths.keys())
        self.rng = random.Random(seed)

    def __len__(self):
        return self.pairs_per_epoch

    def _make_pair(self, idx):
        same = idx % 2 == 0
        if same:
            label = self.rng.choice(self.labels)
            p1, p2 = self.rng.sample(self.class_to_paths[label], 2)
            target = 1.0
        else:
            l1, l2 = self.rng.sample(self.labels, 2)
            p1 = self.rng.choice(self.class_to_paths[l1])
            p2 = self.rng.choice(self.class_to_paths[l2])
            target = 0.0
        return p1, p2, target

    def __getitem__(self, idx):
        p1, p2, target = self._make_pair(idx)
        img1 = Image.open(p1).convert('L')
        img2 = Image.open(p2).convert('L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor([target], dtype=torch.float32), str(p1), str(p2)


class SingleImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, str(path)


class SiameseEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 11, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, target):
        target = target.view(-1)
        distance = F.pairwise_distance(z1, z2)
        loss = target * distance.pow(2) + (1.0 - target) * torch.clamp(self.margin - distance, min=0.0).pow(2)
        return loss.mean(), distance


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_samples(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Каталог с датасетом не найден: {data_dir}')
    samples = []
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith('s')], key=lambda p: int(p.name[1:]))
    for class_dir in class_dirs:
        label = int(class_dir.name[1:]) - 1
        images = sorted(list(class_dir.glob('*.pgm')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')))
        for img_path in images:
            samples.append((img_path, label))
    if not samples:
        raise RuntimeError(f'В каталоге {data_dir} не найдены изображения')
    return samples


def split_samples(samples, train_per_class=8):
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append(path)
    train_samples = []
    test_samples = []
    for label, paths in sorted(by_class.items()):
        paths = sorted(paths)
        train_paths = paths[:train_per_class]
        test_paths = paths[train_per_class:]
        train_samples.extend([(p, label) for p in train_paths])
        test_samples.extend([(p, label) for p in test_paths])
    return train_samples, test_samples


def build_transforms():
    return transforms.Compose([
        transforms.Resize((112, 92)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


@torch.no_grad()
def collect_pair_metrics(model, loader, device, mode='bce', margin=1.0):
    model.eval()
    distances = []
    labels = []
    probs = []
    path_pairs = []
    for x1, x2, target, p1, p2 in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        z1, z2 = model(x1, x2)
        distance = F.pairwise_distance(z1, z2)
        prob = torch.sigmoid(-distance)
        distances.extend(distance.cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())
        labels.extend(target.view(-1).cpu().numpy().tolist())
        path_pairs.extend(list(zip(p1, p2)))
    distances = np.array(distances)
    probs = np.array(probs)
    labels = np.array(labels)
    if mode == 'bce':
        preds = (probs >= 0.5).astype(np.float32)
    else:
        preds = (distances < margin / 2.0).astype(np.float32)
    acc = float((preds == labels).mean())
    same_dist = float(distances[labels == 1].mean()) if np.any(labels == 1) else math.nan
    diff_dist = float(distances[labels == 0].mean()) if np.any(labels == 0) else math.nan
    return {
        'accuracy': acc,
        'same_distance_mean': same_dist,
        'diff_distance_mean': diff_dist,
        'distances': distances,
        'labels': labels,
        'probs': probs,
        'path_pairs': path_pairs,
    }


@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    paths = []
    for x, label, path in loader:
        x = x.to(device)
        z = model.forward_once(x)
        embeddings.append(z.cpu().numpy())
        labels.extend(label.numpy().tolist())
        paths.extend(path)
    return np.concatenate(embeddings, axis=0), np.array(labels), paths


@torch.no_grad()
def select_demo_pairs(model, loader, device, count=6):
    model.eval()
    same_examples = []
    diff_examples = []
    for x1, x2, target, p1, p2 in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        z1, z2 = model(x1, x2)
        distance = F.pairwise_distance(z1, z2).cpu().numpy()
        target_np = target.view(-1).cpu().numpy()
        for i in range(len(distance)):
            item = (p1[i], p2[i], float(distance[i]), int(target_np[i]))
            if item[3] == 1 and len(same_examples) < count // 2:
                same_examples.append(item)
            if item[3] == 0 and len(diff_examples) < count - count // 2:
                diff_examples.append(item)
        if len(same_examples) + len(diff_examples) >= count:
            break
    return same_examples + diff_examples


def plot_tsne(embeddings, labels, out_path, title):
    perplexity = max(5, min(30, len(embeddings) - 1))
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=perplexity)
    pts = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab20', s=40)
    plt.legend(*scatter.legend_elements(num=10), title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_inference_grid(examples, out_path, title):
    rows = len(examples)
    fig, axes = plt.subplots(rows, 2, figsize=(6, 2.5 * rows))
    if rows == 1:
        axes = np.array([axes])
    for row, (p1, p2, distance, label) in enumerate(examples):
        img1 = np.array(Image.open(p1).convert('L'))
        img2 = np.array(Image.open(p2).convert('L'))
        axes[row, 0].imshow(img1, cmap='gray')
        axes[row, 1].imshow(img2, cmap='gray')
        axes[row, 0].set_title(f'label={label}')
        axes[row, 1].set_title(f'distance={distance:.4f}')
        axes[row, 0].axis('off')
        axes[row, 1].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def train_one_epoch(model, loader, optimizer, device, criterion_name='bce', contrastive_margin=1.0):
    model.train()
    bce_loss = nn.BCELoss()
    contrastive = ContrastiveLoss(margin=contrastive_margin)
    total_loss = 0.0
    total_items = 0
    for x1, x2, target, _, _ in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        z1, z2 = model(x1, x2)
        if criterion_name == 'bce':
            distance = F.pairwise_distance(z1, z2).unsqueeze(1)
            prob = torch.sigmoid(-distance)
            loss = bce_loss(prob, target)
        else:
            loss, _ = contrastive(z1, z2, target)
        loss.backward()
        optimizer.step()
        bs = x1.size(0)
        total_loss += loss.item() * bs
        total_items += bs
    return total_loss / max(total_items, 1)


def run_experiment(name, train_loader, test_pair_loader, test_single_loader, device, epochs, lr, margin, out_dir):
    model = SiameseEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion_name=name, contrastive_margin=margin)
        metrics = collect_pair_metrics(model, test_pair_loader, device, mode=name, margin=margin)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_accuracy': metrics['accuracy'],
            'same_distance_mean': metrics['same_distance_mean'],
            'diff_distance_mean': metrics['diff_distance_mean'],
        })
        print(f'[{name}] epoch {epoch}/{epochs} loss={train_loss:.4f} test_acc={metrics["accuracy"]:.4f} same_dist={metrics["same_distance_mean"]:.4f} diff_dist={metrics["diff_distance_mean"]:.4f}')
    final_metrics = collect_pair_metrics(model, test_pair_loader, device, mode=name, margin=margin)
    embeddings, labels, _ = collect_embeddings(model, test_single_loader, device)
    plot_tsne(embeddings, labels, out_dir / f'tsne_{name}.png', f't-SNE ({name})')
    examples = select_demo_pairs(model, test_pair_loader, device, count=6)
    plot_inference_grid(examples, out_dir / f'inference_{name}.png', f'Inference ({name})')
    return model, history, final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='att_faces')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--pairs-train', type=int, default=4000)
    parser.add_argument('--pairs-test', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(args.data_dir)
    train_samples, test_samples = split_samples(samples, train_per_class=8)
    transform = build_transforms()

    train_pair_dataset = PairDataset(train_samples, transform=transform, pairs_per_epoch=args.pairs_train, seed=args.seed)
    test_pair_dataset = PairDataset(test_samples, transform=transform, pairs_per_epoch=args.pairs_test, seed=args.seed + 1)
    test_single_dataset = SingleImageDataset(test_samples, transform=transform)

    train_loader = DataLoader(train_pair_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_pair_loader = DataLoader(test_pair_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_single_loader = DataLoader(test_single_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    _, history_bce, metrics_bce = run_experiment('bce', train_loader, test_pair_loader, test_single_loader, device, args.epochs, args.lr, args.margin, out_dir)
    _, history_contrastive, metrics_contrastive = run_experiment('contrastive', train_loader, test_pair_loader, test_single_loader, device, args.epochs, args.lr, args.margin, out_dir)

    summary = {
        'device': str(device),
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'bce': {
            'history': history_bce,
            'final_accuracy': metrics_bce['accuracy'],
            'same_distance_mean': metrics_bce['same_distance_mean'],
            'diff_distance_mean': metrics_bce['diff_distance_mean'],
        },
        'contrastive': {
            'history': history_contrastive,
            'final_accuracy': metrics_contrastive['accuracy'],
            'same_distance_mean': metrics_contrastive['same_distance_mean'],
            'diff_distance_mean': metrics_contrastive['diff_distance_mean'],
        },
    }

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\nГотово. Сохранены файлы:')
    for path in sorted(out_dir.iterdir()):
        print(path)


if __name__ == '__main__':
    main()
