import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import (
    CLASS_ID_TO_NAME,
    calculate_metrics,
    detect_and_crop_face,
    load_config,
    load_split_images,
    save_joblib,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train HOG + LogisticRegression model for mask detection.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root path.")
    parser.add_argument("--img-size", type=int, default=None, help="Image size for HOG extraction.")
    parser.add_argument("--C", type=float, default=None, help="Inverse regularization strength for LogisticRegression.")
    parser.add_argument("--apply-face-crop", action="store_true", help="Enable face detection and crop.")
    parser.add_argument("--no-face-crop", dest="apply_face_crop", action="store_false", help="Disable face cropping.")
    parser.set_defaults(apply_face_crop=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    params = config["training"]["hog_lr"]
    dataset_root = Path(args.dataset_root or paths["dataset_root"])
    img_size = args.img_size or params["img_size"]
    C = args.C or params["C"]
    apply_face_crop = args.apply_face_crop

    set_seed(config["training"]["seed"])
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    hog_params = {
        "orientations": params["hog_orientations"],
        "pixels_per_cell": tuple(params["pixels_per_cell"]),
        "cells_per_block": tuple(params["cells_per_block"]),
    }

    X_train, y_train, train_meta = load_split_images(
        dataset_root,
        split=paths["train_dir"],
        img_size=img_size,
        apply_face_crop=apply_face_crop,
        cascade=None,
        hog_params=hog_params,
        show_progress=True,
        cascade_path=cascade_path,
    )
    X_val, y_val, val_meta = load_split_images(
        dataset_root,
        split=paths["val_dir"],
        img_size=img_size,
        apply_face_crop=apply_face_crop,
        cascade=None,
        hog_params=hog_params,
        show_progress=True,
        cascade_path=cascade_path,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_scaled, y_train)

    val_probs = clf.predict_proba(X_val_scaled)[:, 1]
    metrics = calculate_metrics(y_val, val_probs)

    print(f"Train samples: {len(train_meta)}, Val samples: {len(val_meta)}")
    print(f"Val F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}")

    models_dir = Path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "hog_lr.joblib"
    save_joblib(
        {
            "model": clf,
            "scaler": scaler,
            "img_size": img_size,
            "apply_face_crop": apply_face_crop,
            "hog_params": hog_params,
            "classes": CLASS_ID_TO_NAME,
        },
        model_path,
    )

    metrics_path = Path(paths["outputs_dir"]) / "hog_lr_val_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved HOG+LR model to {model_path}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()

