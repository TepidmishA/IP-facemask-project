import numpy as np

from train.utils import detect_and_crop_face


def test_detect_and_crop_face_returns_original_when_no_face():
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    cropped, detected, bbox = detect_and_crop_face(dummy)
    assert detected is False
    assert bbox is None
    assert cropped.shape == dummy.shape

