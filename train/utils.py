import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import joblib
import numpy as np
import torch
import yaml
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import hog
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CLASS_ID_TO_NAME = {0: "without_mask", 1: "with_mask"}
CLASS_NAME_TO_ID = {"WithoutMask": 0, "WithMask": 1}
_CASCADE_CACHE: Dict[str, cv2.CascadeClassifier] = {}


def get_cascade(cascade_path: Optional[str] = None) -> cv2.CascadeClassifier:
    """Lazily load Haar cascade; avoids pickling cv2 objects across workers."""
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if cascade_path not in _CASCADE_CACHE:
        _CASCADE_CACHE[cascade_path] = cv2.CascadeClassifier(cascade_path)
    return _CASCADE_CACHE[cascade_path]
_CASCADE_CACHE: Dict[str, cv2.CascadeClassifier] = {}


def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_and_crop_face(
    image: np.ndarray,
    cascade: Optional[cv2.CascadeClassifier] = None,
    cascade_path: Optional[str] = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
    if cascade is None:
        cascade = get_cascade(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if len(faces) == 0:
        return image, False, None
    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    cropped = image[y : y + h, x : x + w]
    return cropped, True, (int(x), int(y), int(w), int(h))


def resize_image(image: np.ndarray, img_size: int) -> np.ndarray:
    return cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)


def get_transforms(img_size: int, augment: bool = False) -> A.Compose:
    if augment:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussianBlur(p=0.2),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    return A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])


class MaskDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        transform: Optional[A.Compose] = None,
        apply_face_crop: bool = True,
        cascade_path: Optional[str] = None,
        return_meta: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.apply_face_crop = apply_face_crop
        self.cascade_path = cascade_path
        self.return_meta = return_meta
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.samples = []
        for class_name, label in CLASS_NAME_TO_ID.items():
            for img_path in (split_dir / class_name).glob("*.png"):
                self.samples.append((img_path, label))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = read_image(img_path)
        face_detected = False
        bbox = None
        if self.apply_face_crop:
            cascade = get_cascade(self.cascade_path)
            image, face_detected, bbox = detect_and_crop_face(image, cascade=cascade)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if self.return_meta:
            return image, label, {"path": str(img_path), "face_detected": face_detected, "bbox": bbox}
        return image, label


def build_dataloaders(
    root_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int = 2,
    augment: bool = False,
    apply_face_crop: bool = True,
    cascade_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MaskDataset(
        root_dir=root_dir,
        split="Train",
        transform=get_transforms(img_size, augment=augment),
        apply_face_crop=apply_face_crop,
        cascade_path=cascade_path,
    )
    val_ds = MaskDataset(
        root_dir=root_dir,
        split="Validation",
        transform=get_transforms(img_size, augment=False),
        apply_face_crop=apply_face_crop,
        cascade_path=cascade_path,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def compute_class_weights_from_loader(loader: DataLoader, show_progress: bool = False) -> torch.Tensor:
    labels = []
    iterator = tqdm(loader, desc="Class weights", leave=True) if show_progress else loader
    for _, y in iterator:
        labels.extend(y.tolist())
    weights = compute_class_weight(class_weight="balanced", classes=np.array(list(CLASS_ID_TO_NAME.keys())), y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def extract_hog_features(
    image: np.ndarray,
    img_size: int,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    image_gray = cv2.cvtColor(resize_image(image, img_size), cv2.COLOR_RGB2GRAY)
    features = hog(
        image_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def load_split_images(
    root_dir: Path,
    split: str,
    img_size: int,
    apply_face_crop: bool,
    cascade: Optional[cv2.CascadeClassifier] = None,
    hog_params: Optional[Dict] = None,
    show_progress: bool = False,
    cascade_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    cascade = cascade or get_cascade(cascade_path)
    hog_params = hog_params or {}
    x_list, y_list, metas = [], [], []
    split_dir = root_dir / split
    for class_name, label in CLASS_NAME_TO_ID.items():
        paths = list((split_dir / class_name).glob("*.png"))
        iterator = tqdm(paths, desc=f"{split}:{class_name}", leave=False) if show_progress else paths
        for img_path in iterator:
            try:
                image = read_image(img_path)
                face_detected = False
                bbox = None
                if apply_face_crop:
                    image, face_detected, bbox = detect_and_crop_face(image, cascade)
                features = extract_hog_features(
                    image,
                    img_size=img_size,
                    orientations=hog_params.get("orientations", 9),
                    pixels_per_cell=tuple(hog_params.get("pixels_per_cell", (8, 8))),
                    cells_per_block=tuple(hog_params.get("cells_per_block", (2, 2))),
                )
                x_list.append(features)
                y_list.append(label)
                metas.append({"path": str(img_path), "face_detected": face_detected, "bbox": bbox})
            except Exception as exc:
                LOGGER.warning("Skipping image %s due to error: %s", img_path, exc)
    if len(x_list) == 0:
        raise RuntimeError(
            f"No images found for split '{split}' under '{split_dir}'. "
            "Check dataset_root and directory names (expected subfolders WithMask/WithoutMask)."
        )
    return np.array(x_list), np.array(y_list), metas


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc,
    }


def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path):
    return joblib.load(path)


def get_device() -> torch.device:
    try:
        if torch.cuda.is_available() and torch.version.cuda:
            return torch.device("cuda")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("CUDA check failed (%s); falling back to CPU", exc)
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

