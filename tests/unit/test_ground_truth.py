"""Unit tests for vehicle_speed.telemetry.ground_truth module."""

import pytest

from vehicle_speed.telemetry.ground_truth import (
    check_travel_boundary,
    ground_truth_speed,
    steps_to_metres,
)
from vehicle_speed.telemetry.protocol import StatusEvent


def test_steps_to_metres_conversion() -> None:
    """400 step/mm ile 400.000 adım tam 1.0 metre olmalıdır."""
    # 400 step/mm * 1000 mm/m = 400.000 step/m
    assert steps_to_metres(400000, steps_per_mm=400) == pytest.approx(1.0)
    assert steps_to_metres(200000, steps_per_mm=400) == pytest.approx(0.500)
    assert steps_to_metres(0, steps_per_mm=400) == pytest.approx(0.0)


def test_steps_to_metres_rejects_non_positive_ratio() -> None:
    """steps_per_mm <= 0 durumunda ValueError yükseltilir."""
    with pytest.raises(ValueError, match="steps_per_mm pozitif olmalıdır"):
        steps_to_metres(1000, steps_per_mm=0)


def test_ground_truth_speed_derivation() -> None:
    """İki StatusEvent arasından türetilen hız doğru fiziksel m/s vermelidir."""
    # 0. saniyede 0 mm (0 adım)
    e1 = StatusEvent(event="status", elapsed_time_ms=0, current_position=0)
    # 5. saniyede 200 mm (80.000 adım -> 0.2 metre)
    e2 = StatusEvent(event="status", elapsed_time_ms=5000, current_position=80000)

    # Hız = 0.2 m / 5.0 s = 0.04 m/s (40 mm/s)
    profile = ground_truth_speed([e1, e2])
    assert len(profile) == 1
    t_s, v_ms = profile[0]
    assert t_s == pytest.approx(5.0)
    assert v_ms == pytest.approx(0.04)


def test_ground_truth_speed_empty_on_single_event() -> None:
    """Tek bir olay veya boş listede hız türetilemez, boş liste döner."""
    assert ground_truth_speed([]) == []
    e1 = StatusEvent(event="status", elapsed_time_ms=0, current_position=0)
    assert ground_truth_speed([e1]) == []


def test_check_travel_boundary() -> None:
    """Vidalı mil 500 mm sınır kontrolü."""
    # 500 mm * 400 = 200.000 adım azami
    assert check_travel_boundary(0) is True
    assert check_travel_boundary(100000) is True
    assert check_travel_boundary(200000) is True
    assert check_travel_boundary(200001) is False
    assert check_travel_boundary(-1) is False
