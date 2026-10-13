"""Unit tests for vehicle_speed.nrc (Neural Residual Calibrator) module."""

from pathlib import Path

import numpy as np
import pytest

from vehicle_speed.detector import Detection
from vehicle_speed.nrc.features import NRCFeatures, extract_nrc_features
from vehicle_speed.nrc.model import NeuralResidualCalibrator


def test_extract_nrc_features() -> None:
    """Detection nesnesinden 5 boyutlu NRC girdi vektörü çıkarımı."""
    det = Detection(
        track_id=1,
        bbox_xyxy=(10, 20, 110, 120),
        confidence=0.85,
        pixel_width=100.0,
        center_px=(60.0, 70.0),
        aspect_ratio=1.0,
        y_normalized=0.5,
    )
    feat = extract_nrc_features(det, raw_speed_ms=12.5)

    assert isinstance(feat, NRCFeatures)
    assert feat.v_raw_ms == 12.5
    assert feat.pixel_width == 100.0
    assert feat.y_normalized == 0.5
    assert feat.aspect_ratio == 1.0
    assert feat.confidence == 0.85

    arr = feat.to_numpy()
    assert arr.shape == (5,)
    assert arr.dtype == np.float32


def test_nrc_model_forward_pass_and_calibration() -> None:
    """NRC modelinin artık hız hesaplaması ve nihai hız kalibrasyonu."""
    calibrator = NeuralResidualCalibrator()

    feat = NRCFeatures(
        v_raw_ms=20.0,
        pixel_width=120.0,
        y_normalized=0.6,
        aspect_ratio=1.2,
        confidence=0.9,
    )

    delta_v = calibrator.predict_residual(feat)
    assert isinstance(delta_v, float)

    calibrated_speed = calibrator.calibrate(raw_speed_ms=20.0, features=feat)
    assert calibrated_speed >= 0.0
    assert calibrated_speed == pytest.approx(max(0.0, 20.0 + delta_v), rel=1e-5)


def test_nrc_model_save_and_load_weights(tmp_path: Path) -> None:
    """Model ağırlıklarının kaydedilip geri yüklenmesi testi."""
    model1 = NeuralResidualCalibrator()
    weight_file = tmp_path / "nrc_weights.npz"

    model1.save_weights(weight_file)
    assert weight_file.is_file()

    model2 = NeuralResidualCalibrator(weights_path=weight_file)

    feat = NRCFeatures(
        v_raw_ms=15.0,
        pixel_width=80.0,
        y_normalized=0.4,
        aspect_ratio=0.9,
        confidence=0.8,
    )

    pred1 = model1.predict_residual(feat)
    pred2 = model2.predict_residual(feat)

    assert pred1 == pytest.approx(pred2, rel=1e-6)
