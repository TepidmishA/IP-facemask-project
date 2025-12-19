import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from ml_workflow.train.utils import (
    CLASS_ID_TO_NAME,
    MaskDataset,
    calculate_metrics,
    get_transforms,
    load_config,
    load_joblib,
    load_split_images,
    plot_confusion_matrix,
    plot_roc_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained mask detection models on the test set.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["hog_lr", "mobilenet_v2", "resnet18_lr", "all"],
        help="Model to evaluate.",
    )
    return parser.parse_args()


def evaluate_hog_lr(config, dataset_root: Path):
    paths = config["paths"]
    artifact = load_joblib(Path(paths["models_dir"]) / "hog_lr.joblib")
    hog_params = artifact.get("hog_params", {})
    img_size = artifact["img_size"]
    apply_face_crop = artifact.get("apply_face_crop", True)

    X_test, y_test, _ = load_split_images(
        dataset_root,
        split=paths["test_dir"],
        img_size=img_size,
        apply_face_crop=apply_face_crop,
        hog_params=hog_params,
    )
    X_scaled = artifact["scaler"].transform(X_test)
    probs = artifact["model"].predict_proba(X_scaled)[:, 1]
    metrics = calculate_metrics(y_test, probs)
    preds = (probs >= 0.5).astype(int)

    plot_dir = Path(paths["outputs_dir"]) / "plots"
    plot_confusion_matrix(y_test, preds, [CLASS_ID_TO_NAME[0], CLASS_ID_TO_NAME[1]], plot_dir / "hog_lr_cm.png")
    plot_roc_curve(y_test, probs, plot_dir / "hog_lr_roc.png")
    return metrics


def load_mobilenet_v2_model(artifact_path: Path, device):
    checkpoint = torch.load(artifact_path, map_location=device)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    img_size = checkpoint.get("img_size", 224)
    apply_face_crop = checkpoint.get("apply_face_crop", True)
    return model, img_size, apply_face_crop


def evaluate_mobilenet_v2(config, dataset_root: Path):
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size, apply_face_crop = load_mobilenet_v2_model(Path(paths["models_dir"]) / "mobilenet_v2_best.pth", device)

    transform = get_transforms(img_size=img_size, augment=False)
    dataset = MaskDataset(dataset_root, split=paths["test_dir"], transform=transform, apply_face_crop=apply_face_crop)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=config["training"]["num_workers"])

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    preds = (y_prob >= 0.5).astype(int)
    metrics = calculate_metrics(y_true, y_prob)

    plot_dir = Path(paths["outputs_dir"]) / "plots"
    plot_confusion_matrix(y_true, preds, [CLASS_ID_TO_NAME[0], CLASS_ID_TO_NAME[1]], plot_dir / "mobilenet_v2_cm.png")
    plot_roc_curve(y_true, y_prob, plot_dir / "mobilenet_v2_roc.png")
    return metrics


def load_resnet18_lr_classifier(config, device):
    paths = config["paths"]
    artifact = load_joblib(Path(paths["models_dir"]) / "resnet18_lr.joblib")
    backbone_name = artifact.get("backbone", "resnet18")
    if backbone_name != "resnet18":
        raise ValueError("Only resnet18 backbone is supported in this artifact.")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model, artifact


def evaluate_resnet18_lr(config, dataset_root: Path):
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor, artifact = load_resnet18_lr_classifier(config, device)
    img_size = artifact["img_size"]
    apply_face_crop = artifact.get("apply_face_crop", True)

    transform = get_transforms(img_size=img_size, augment=False)
    dataset = MaskDataset(dataset_root, split=paths["test_dir"], transform=transform, apply_face_crop=apply_face_crop)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=config["training"]["num_workers"])

    feats, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            emb = feature_extractor(images).squeeze(-1).squeeze(-1)
            feats.append(emb.cpu().numpy())
            labels.extend(lbls.numpy().tolist())
    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    feats_scaled = artifact["scaler"].transform(feats)
    probs = artifact["classifier"].predict_proba(feats_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = calculate_metrics(labels, probs)

    plot_dir = Path(paths["outputs_dir"]) / "plots"
    plot_confusion_matrix(labels, preds, [CLASS_ID_TO_NAME[0], CLASS_ID_TO_NAME[1]], plot_dir / "resnet18_lr_cm.png")
    plot_roc_curve(labels, probs, plot_dir / "resnet18_lr_roc.png")
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_root = Path(args.dataset_root or config["paths"]["dataset_root"])

    results = {}
    if args.model in ["hog_lr", "all"]:
        results["hog_lr"] = evaluate_hog_lr(config, dataset_root)
    if args.model in ["mobilenet_v2", "all"]:
        results["mobilenet_v2"] = evaluate_mobilenet_v2(config, dataset_root)
    if args.model in ["resnet18_lr", "all"]:
        results["resnet18_lr"] = evaluate_resnet18_lr(config, dataset_root)

    print(json.dumps(results, indent=2))
    out_path = Path(config["paths"]["outputs_dir"]) / "test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

