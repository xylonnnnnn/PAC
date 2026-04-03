import random
import ssl
from pathlib import Path
from collections import defaultdict
from urllib.error import URLError

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pytorch_metric_learning import losses

try:
    import certifi
except ImportError:
    certifi = None


IMAGE_ROOT = Path("./EuroSAT_RGB")
MODEL_PATH = Path("./best_arcface_eurosat.pt")
PLOTS_DIR = Path("./plots")

BATCH_SIZE = 128
NUM_WORKERS = 2
EMBEDDING_SIZE = 128
EPOCHS = 5
BACKBONE_LR = 1e-4
ARCFACE_LR = 1e-2
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
VAL_SIZE = 0.1
IMAGE_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_ssl():
    if certifi is None:
        return
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def resolve_image_root():
    if IMAGE_ROOT.is_dir() and any(child.is_dir() for child in IMAGE_ROOT.iterdir()):
        return IMAGE_ROOT
    raise FileNotFoundError(f"Не найдена папка с изображениями: {IMAGE_ROOT}")


def save_plot(filename: str):
    PLOTS_DIR.mkdir(exist_ok=True)
    output_path = PLOTS_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {output_path.resolve()}")


def build_file_splits(image_root: Path, test_size=0.2, val_size=0.1):
    class_names = sorted([p.name for p in image_root.iterdir() if p.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    all_paths, all_labels = [], []
    for cls_name in class_names:
        files = sorted((image_root / cls_name).glob("*.jpg"))
        all_paths.extend(files)
        all_labels.extend([class_to_idx[cls_name]] * len(files))

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=test_size,
        stratify=all_labels,
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=val_size,
        stratify=train_labels,
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
    }


class EuroSATPathsDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128, use_pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        try:
            self.backbone = models.resnet18(weights=weights)
        except (URLError, ssl.SSLError) as error:
            print(
                "Не удалось загрузить предобученные веса ResNet18 "
                f"({error}). Продолжаю с случайной инициализацией."
            )
            self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_size),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def run_epoch(model, loader, loss_func, model_optimizer, loss_optimizer=None, train=True):
    if train:
        model.train()
        loss_func.train()
    else:
        model.eval()
        loss_func.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(train):
            embeddings = model(images)
            loss = loss_func(embeddings, labels)
            logits = loss_func.get_logits(embeddings)
            preds = logits.argmax(dim=1)

            if train:
                model_optimizer.zero_grad()
                if loss_optimizer is not None:
                    loss_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()
                if loss_optimizer is not None:
                    loss_optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    all_embeddings, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_embeddings, all_labels


@torch.no_grad()
def predict_classes(model, loss_func, loader):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        embeddings = model(images)
        logits = loss_func.get_logits(embeddings)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("История обучения")
    plt.legend()
    plt.grid(True)
    save_plot("history_loss.png")

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Точность классификации через ArcFace logits")
    plt.legend()
    plt.grid(True)
    save_plot("history_accuracy.png")


def plot_tsne(embeddings, labels, class_names, sample_size=3000):
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=12, alpha=0.7, label=class_name)

    plt.title("t-SNE по embedding'ам тестового датасета")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_plot("tsne_test_embeddings.png")


@torch.no_grad()
def visualize_pairs(model, dataset, num_same=3, num_diff=3):
    model.eval()

    groups = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        groups[label].append(idx)

    def get_embedding(img_tensor):
        emb = model(img_tensor.unsqueeze(0).to(DEVICE))
        return emb.squeeze(0).cpu()

    def cosine_distance(a, b):
        return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    pairs = []

    labels_list = list(groups.keys())
    for _ in range(num_same):
        label = random.choice(labels_list)
        i1, i2 = random.sample(groups[label], 2)
        pairs.append((i1, i2, "same class"))

    for _ in range(num_diff):
        label1, label2 = random.sample(labels_list, 2)
        i1 = random.choice(groups[label1])
        i2 = random.choice(groups[label2])
        pairs.append((i1, i2, "different classes"))

    plt.figure(figsize=(10, 3 * len(pairs)))
    for row, (i1, i2, pair_type) in enumerate(pairs, start=1):
        img1, label1 = dataset[i1]
        img2, label2 = dataset[i2]

        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
        dist = cosine_distance(emb1, emb2)

        img1_np = img1.permute(1, 2, 0).numpy()
        img2_np = img2.permute(1, 2, 0).numpy()
        img1_np = np.clip(img1_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        img2_np = np.clip(img2_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

        plt.subplot(len(pairs), 2, 2 * row - 1)
        plt.imshow(img1_np)
        plt.axis("off")
        plt.title(f"{pair_type}\nlabel={label1}")

        plt.subplot(len(pairs), 2, 2 * row)
        plt.imshow(img2_np)
        plt.axis("off")
        plt.title(f"label={label2}\ncosine distance={dist:.4f}")

    plt.tight_layout()
    save_plot("sample_pairs.png")


def main():
    configure_ssl()
    print("Device:", DEVICE)

    image_root = resolve_image_root()
    print("Датасет:", image_root.resolve())
    splits = build_file_splits(image_root, test_size=TEST_SIZE, val_size=VAL_SIZE)
    class_names = splits["class_names"]
    num_classes = len(class_names)
    print("Классы:", class_names)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = EuroSATPathsDataset(*splits["train"], transform=train_transform)
    val_dataset = EuroSATPathsDataset(*splits["val"], transform=test_transform)
    test_dataset = EuroSATPathsDataset(*splits["test"], transform=test_transform)

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    model = EmbeddingNet(embedding_size=EMBEDDING_SIZE).to(DEVICE)

    loss_func = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=EMBEDDING_SIZE,
        margin=28.6,
        scale=64,
    ).to(DEVICE)

    model_optimizer = torch.optim.AdamW(model.parameters(), lr=BACKBONE_LR, weight_decay=WEIGHT_DECAY)
    loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=ARCFACE_LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_func, model_optimizer, loss_optimizer, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, loss_func, model_optimizer, None, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "loss_state_dict": loss_func.state_dict(),
                    "class_names": class_names,
                },
                MODEL_PATH,
            )

    print(f"Лучшая val_acc = {best_val_acc:.4f}")
    plot_history(history)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_func.load_state_dict(checkpoint["loss_state_dict"])

    y_true, y_pred = predict_classes(model, loss_func, test_loader)
    print("\nClassification report на test:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    test_embeddings, test_labels = extract_embeddings(model, test_loader)
    plot_tsne(test_embeddings.numpy(), test_labels.numpy(), class_names, sample_size=3000)

    visualize_pairs(model, test_dataset, num_same=3, num_diff=3)


if __name__ == "__main__":
    main()
