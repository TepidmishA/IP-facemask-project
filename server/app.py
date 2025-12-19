import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Request
from pydantic import BaseModel
from torchvision import models

from ml_workflow.train.utils import (
    CLASS_ID_TO_NAME,
    detect_and_crop_face,
    extract_hog_features,
    get_device,
    get_transforms,
    load_config,
    load_joblib,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("server")

app = FastAPI(title="Face Mask Detection API", version="1.0")

CONFIG = load_config("config.yaml")
PATHS = CONFIG["paths"]
DEVICE = get_device()
USE_DUMMY_MODELS = os.getenv("USE_DUMMY_MODELS", "0") == "1"

_HOG_LR_CACHE: Optional[Dict[str, Any]] = None
_MOBILENET_V2_MODEL: Optional[torch.nn.Module] = None
_MOBILENET_V2_CFG: Optional[Dict[str, Any]] = None
_RESNET18_LR_CACHE: Optional[Dict[str, Any]] = None

LABEL_MAP = {0: "not_masked", 1: "masked"}


class PredictionResponse(BaseModel):
    prediction: str
    probability: float  # probability of masked (backward compatible)
    probability_masked: float
    probability_not_masked: float
    model: str
    details: Dict[str, Any]


def _load_image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_hog_lr():
    global _HOG_LR_CACHE
    if _HOG_LR_CACHE is not None:
        return _HOG_LR_CACHE
    if USE_DUMMY_MODELS:
        _HOG_LR_CACHE = {
            "model": None,
            "scaler": None,
            "img_size": 128,
            "apply_face_crop": False,
            "hog_params": {},
        }
        return _HOG_LR_CACHE
    path = Path(PATHS["models_dir"]) / "hog_lr.joblib"
    if not path.exists():
        raise FileNotFoundError("HOG+LR model not found. Train it first.")
    _HOG_LR_CACHE = load_joblib(path)
    return _HOG_LR_CACHE


def _load_mobilenet_v2():
    global _MOBILENET_V2_MODEL, _MOBILENET_V2_CFG
    if _MOBILENET_V2_MODEL is not None:
        return _MOBILENET_V2_MODEL, _MOBILENET_V2_CFG
    if USE_DUMMY_MODELS:
        _MOBILENET_V2_MODEL = torch.nn.Identity()
        _MOBILENET_V2_CFG = {"img_size": 224, "apply_face_crop": False}
        return _MOBILENET_V2_MODEL, _MOBILENET_V2_CFG
    path = Path(PATHS["models_dir"]) / "mobilenet_v2_best.pth"
    if not path.exists():
        raise FileNotFoundError("MobileNetV2 model not found. Train it first.")
    checkpoint = torch.load(path, map_location=DEVICE)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    _MOBILENET_V2_MODEL = model
    _MOBILENET_V2_CFG = {"img_size": checkpoint.get("img_size", 224), "apply_face_crop": checkpoint.get("apply_face_crop", True)}
    return _MOBILENET_V2_MODEL, _MOBILENET_V2_CFG


def _load_resnet18_lr():
    global _RESNET18_LR_CACHE
    if _RESNET18_LR_CACHE is not None:
        return _RESNET18_LR_CACHE
    if USE_DUMMY_MODELS:
        _RESNET18_LR_CACHE = {
            "extractor": torch.nn.Identity(),
            "artifact": {
                "classifier": None,
                "scaler": None,
                "img_size": 224,
                "apply_face_crop": False,
            },
        }
        return _RESNET18_LR_CACHE
    path = Path(PATHS["models_dir"]) / "resnet18_lr.joblib"
    if not path.exists():
        raise FileNotFoundError("ResNet18+LR model not found. Train it first.")
    artifact = load_joblib(path)
    extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    extractor.fc = nn.Identity()
    extractor.to(DEVICE).eval()
    _RESNET18_LR_CACHE = {"extractor": extractor, "artifact": artifact}
    return _RESNET18_LR_CACHE


def _predict_hog_lr(image: np.ndarray) -> Tuple[int, float, float, Dict[str, Any]]:
    artifact = _load_hog_lr()
    face_detected = False
    bbox = None
    if artifact.get("apply_face_crop", False):
        image, face_detected, bbox = detect_and_crop_face(image)
    features = extract_hog_features(
        image,
        img_size=artifact["img_size"],
        orientations=artifact["hog_params"].get("orientations", 9),
        pixels_per_cell=tuple(artifact["hog_params"].get("pixels_per_cell", (8, 8))),
        cells_per_block=tuple(artifact["hog_params"].get("cells_per_block", (2, 2))),
    )
    feat_scaled = artifact["scaler"].transform([features]) if artifact["scaler"] is not None else [features]
    proba = artifact["model"].predict_proba(feat_scaled)[0] if artifact["model"] is not None else np.array([0.5, 0.5])
    prob_not_masked = float(proba[0])
    prob_masked = float(proba[1])
    pred = int(prob_masked >= 0.5)
    return pred, prob_masked, prob_not_masked, {"face_detected": face_detected, "face_bbox": bbox}


def _predict_mobilenet_v2(image: np.ndarray) -> Tuple[int, float, float, Dict[str, Any]]:
    model, cfg = _load_mobilenet_v2()
    face_detected = False
    bbox = None
    if cfg.get("apply_face_crop", False):
        image, face_detected, bbox = detect_and_crop_face(image)
    transform = get_transforms(img_size=cfg["img_size"], augment=False)
    tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor).squeeze()
        prob_masked = torch.sigmoid(outputs).item() if not isinstance(model, torch.nn.Identity) else 0.5
    prob_not_masked = float(1.0 - prob_masked)
    pred = int(prob_masked >= 0.5)
    return pred, float(prob_masked), prob_not_masked, {"face_detected": face_detected, "face_bbox": bbox}


def _predict_resnet18_lr(image: np.ndarray) -> Tuple[int, float, float, Dict[str, Any]]:
    cache = _load_resnet18_lr()
    extractor = cache["extractor"]
    artifact = cache["artifact"]
    face_detected = False
    bbox = None
    if artifact.get("apply_face_crop", False):
        image, face_detected, bbox = detect_and_crop_face(image)
    transform = get_transforms(img_size=artifact["img_size"], augment=False)
    tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = extractor(tensor).squeeze().cpu().numpy()
    feat_scaled = artifact["scaler"].transform([feat]) if artifact["scaler"] is not None else [feat]
    proba = artifact["classifier"].predict_proba(feat_scaled)[0] if artifact["classifier"] is not None else np.array([0.5, 0.5])
    prob_not_masked = float(proba[0])
    prob_masked = float(proba[1])
    pred = int(prob_masked >= 0.5)
    return pred, prob_masked, prob_not_masked, {"face_detected": face_detected, "face_bbox": bbox}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: UploadFile = File(..., description="Image file (png, jpg)"),
    model: str = Query("mobilenet_v2", pattern="^(hog_lr|mobilenet_v2|resnet18_lr)$"),
):
    try:
        raw = await image.read()
        np_image = _load_image_from_bytes(raw)
        LOGGER.info("Request %s model=%s size=%s bytes", request.client.host if request.client else "?", model, len(raw))
    except Exception as exc:
        LOGGER.exception("Invalid image file: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    try:
        if model == "hog_lr":
            pred, prob_masked, prob_not_masked, details = _predict_hog_lr(np_image)
        elif model == "resnet18_lr":
            pred, prob_masked, prob_not_masked, details = _predict_resnet18_lr(np_image)
        else:
            pred, prob_masked, prob_not_masked, details = _predict_mobilenet_v2(np_image)
    except FileNotFoundError as exc:
        LOGGER.error("Model missing (%s): %s", model, exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        LOGGER.exception("Inference failed for model=%s", model)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    prob_masked = float(np.clip(prob_masked, 0.0, 1.0))
    prob_not_masked = float(np.clip(prob_not_masked, 0.0, 1.0))

    resp = PredictionResponse(
        prediction=LABEL_MAP[pred],
        probability=prob_masked,  # backward compatible
        probability_masked=prob_masked,
        probability_not_masked=prob_not_masked,
        model=model,
        details=details,
    )
    LOGGER.info("Response model=%s pred=%s prob=%.4f details=%s", model, resp.prediction, resp.probability, resp.details)
    return resp

