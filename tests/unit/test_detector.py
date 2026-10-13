"""Unit tests for vehicle_speed.detector module."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vehicle_speed.detector import Detection, ObjectDetector, to_bbox


def test_to_bbox_converts_valid_sequence() -> None:
    """4 elemanlı liste veya array doğru (x1, y1, x2, y2) tuple'ına dönüşür."""
    assert to_bbox([10.4, 20.9, 100.1, 200.7]) == (10, 20, 100, 200)
    assert to_bbox((0, 0, 50, 60)) == (0, 0, 50, 60)


@pytest.mark.parametrize("invalid_raw", [[1, 2, 3], [1, 2, 3, 4, 5], 123])
def test_to_bbox_rejects_invalid_inputs(invalid_raw: object) -> None:
    """4 elemanlı olmayan veya yinelenebilir olmayan girdilerde ValueError fırlatılır."""
    with pytest.raises(ValueError):
        to_bbox(invalid_raw)


def test_detection_dataclass_immutable() -> None:
    """Detection nesnesi dondurulmuştur (frozen)."""
    det = Detection(
        track_id=1,
        bbox_xyxy=(10, 20, 110, 120),
        confidence=0.85,
        pixel_width=100.0,
        center_px=(60.0, 70.0),
        aspect_ratio=1.0,
        y_normalized=0.5,
    )
    with pytest.raises(FrozenInstanceError):
        det.confidence = 0.90  # type: ignore[misc]
    assert det.pixel_width == 100.0
    assert det.track_id == 1


class MockTensor:
    """Mock tensor mimicking PyTorch tensor cpu().numpy() chain."""

    def __init__(self, val: Any) -> None:
        self._val = val

    def cpu(self) -> "MockTensor":
        return self

    def numpy(self) -> Any:
        return self._val


class MockBoxes:
    """Mock YOLO boxes container."""

    def __init__(self) -> None:
        self.xyxy = [MockTensor(np.array([10.0, 20.0, 110.0, 120.0]))]
        self.id = [MockTensor(np.array(42))]
        self.conf = [MockTensor(np.array(0.92))]
        self.cls = [MockTensor(np.array(0))]

    def __len__(self) -> int:
        return 1


class MockResult:
    """Mock YOLO detection result."""

    def __init__(self) -> None:
        self.boxes: MockBoxes | None = MockBoxes()


class MockYOLO:
    """Mock YOLO model instance."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def track(self, **kwargs: Any) -> list[MockResult]:
        return [MockResult()]


def test_object_detector_track_with_mock_model() -> None:
    """ObjectDetector.track() metodu YOLO çıktılarını doğru Detection sözleşmesine çevirir."""
    detector = ObjectDetector(model_path=Path("dummy.pt"), mode="ball")
    detector._model = MockYOLO("dummy.pt")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.track(dummy_frame)

    assert len(dets) == 1
    d = dets[0]
    assert d.track_id == 42
    assert d.bbox_xyxy == (10, 20, 110, 120)
    assert d.confidence == pytest.approx(0.92)
    assert d.pixel_width == 100.0
    assert d.center_px == (60.0, 70.0)
    assert d.aspect_ratio == pytest.approx(1.0)
