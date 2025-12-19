import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm

from ml_workflow.train.utils import (
    CLASS_ID_TO_NAME,
    build_dataloaders,
    calculate_metrics,
    compute_class_weights_from_loader,
    ensure_dir,
    get_device,
    load_config,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train deep learning model (MobileNetV2) for mask detection.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--img-size", type=int, default=None, help="Image size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--early-stopping", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--no-face-crop", action="store_true", help="Disable face cropping.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers (set 0 for Windows).")
    return parser.parse_args()


def build_model(num_classes: int = 1):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            running_loss += loss.item() * images.size(0)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    return epoch_loss, metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    params = config["training"]["mobilenet_v2"]

    dataset_root = Path(args.dataset_root or paths["dataset_root"])
    epochs = args.epochs or params["epochs"]
    batch_size = args.batch_size or params["batch_size"]
    img_size = args.img_size or params["img_size"]
    lr = args.lr or params["lr"]
    weight_decay = args.weight_decay or params["weight_decay"]
    early_stopping = args.early_stopping or params["early_stopping_patience"]
    apply_face_crop = not args.no_face_crop
    num_workers = args.num_workers if args.num_workers is not None else config["training"]["num_workers"]

    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(
        root_dir=dataset_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=True,
        apply_face_crop=apply_face_crop,
        cascade_path=cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    )

    class_weights = compute_class_weights_from_loader(train_loader, show_progress=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

    model = build_model(num_classes=1).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=params["lr_patience"])

    best_f1 = -np.inf
    patience_counter = 0
    history = []

    epoch_iter = tqdm(range(1, epochs + 1), desc="Epochs", ncols=100)
    for epoch in epoch_iter:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            model_path = Path(paths["models_dir"]) / "mobilenet_v2_best.pth"
            ensure_dir(model_path.parent)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_to_idx": CLASS_ID_TO_NAME,
                    "img_size": img_size,
                    "apply_face_crop": apply_face_crop,
                },
                model_path,
            )
        else:
            patience_counter += 1
        if patience_counter >= early_stopping:
            print("Early stopping triggered.")
            break

    log_path = Path(paths["logs_dir"]) / "mobilenet_v2_training_log.csv"
    ensure_dir(log_path.parent)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    metrics_path = Path(paths["outputs_dir"]) / "mobilenet_v2_val_metrics.json"
    ensure_dir(metrics_path.parent)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_f1": best_f1, "history": history}, f, indent=2)
    print(f"Best F1: {best_f1:.4f}")
    print(f"Saved best model to {Path(paths['models_dir']) / 'mobilenet_v2_best.pth'}")


if __name__ == "__main__":
    main()

