import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from ml_workflow.train.utils import (
    CLASS_ID_TO_NAME,
    MaskDataset,
    calculate_metrics,
    ensure_dir,
    get_device,
    get_transforms,
    load_config,
    save_joblib,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train hybrid model: ResNet18 embeddings + LogisticRegression classifier."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--img-size", type=int, default=None, help="Image size for feature extractor.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding extraction.")
    parser.add_argument("--lr", type=float, default=None, help="Unused (for compatibility).")
    parser.add_argument("--no-face-crop", action="store_true", help="Disable face cropping.")
    return parser.parse_args()


def build_feature_extractor():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


def compute_embeddings(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Embed", leave=False):
            images = images.to(device)
            emb = model(images).squeeze(-1).squeeze(-1)
            feats.append(emb.cpu().numpy())
            labels.extend(lbls.numpy().tolist())
    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)
    return feats, labels


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    params = config["training"]["resnet18_lr"]

    dataset_root = Path(args.dataset_root or paths["dataset_root"])
    img_size = args.img_size or params["img_size"]
    batch_size = args.batch_size or params["batch_size"]
    apply_face_crop = not args.no_face_crop

    set_seed(config["training"]["seed"])
    device = get_device()

    transform = get_transforms(img_size=img_size, augment=False)
    train_ds = MaskDataset(dataset_root, split=paths["train_dir"], transform=transform, apply_face_crop=apply_face_crop)
    val_ds = MaskDataset(dataset_root, split=paths["val_dir"], transform=transform, apply_face_crop=apply_face_crop)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=config["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=config["training"]["num_workers"])

    extractor = build_feature_extractor().to(device)
    train_feats, train_labels = compute_embeddings(extractor, train_loader, device)
    val_feats, val_labels = compute_embeddings(extractor, val_loader, device)

    scaler = StandardScaler()
    train_feats_norm = scaler.fit_transform(train_feats)
    val_feats_norm = scaler.transform(val_feats)

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(train_feats_norm, train_labels)

    val_probs = clf.predict_proba(val_feats_norm)[:, 1]
    metrics = calculate_metrics(val_labels, val_probs)

    model_path = Path(paths["models_dir"]) / "resnet18_lr.joblib"
    ensure_dir(model_path.parent)
    save_joblib(
        {
            "classifier": clf,
            "scaler": scaler,
            "backbone": "resnet18",
            "img_size": img_size,
            "apply_face_crop": apply_face_crop,
            "classes": CLASS_ID_TO_NAME,
        },
        model_path,
    )

    metrics_path = Path(paths["outputs_dir"]) / "resnet18_lr_val_metrics.json"
    ensure_dir(metrics_path.parent)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved hybrid model to {model_path}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()

