"""Unit tests for scripts/collect_dataset.py tool."""

import sys
from pathlib import Path

# Add scripts directory to sys.path for import
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest  # noqa: E402

from collect_dataset import (  # noqa: E402
    build_parser,
    interpolate_ground_truth_speed,
    load_telemetry_status_events,
)
from vehicle_speed.telemetry.protocol import StatusEvent  # noqa: E402


def test_build_parser() -> None:
    """CLI argüman ayrıştırıcısı gereklilikleri ve varsayılanları doğru olmalıdır."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--video",
            "test.mp4",
            "--telemetry",
            "test.jsonl",
        ]
    )
    assert args.video == Path("test.mp4")
    assert args.telemetry == Path("test.jsonl")
    assert args.output == Path("data/nrc_dataset.npz")
    assert args.mode == "ball"
    assert args.filter == "none"
    assert args.time_offset_ms == 0.0


def test_load_telemetry_status_events() -> None:
    """Fixture dosyasından sadece StatusEvent nesneleri ve zaman sıralı yüklenmelidir."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "telemetry_sample.jsonl"
    events = load_telemetry_status_events(fixture_path)
    assert len(events) == 8
    # Zaman sırası denetimi
    for i in range(1, len(events)):
        assert events[i].elapsed_time_ms >= events[i - 1].elapsed_time_ms


def test_interpolate_ground_truth_speed() -> None:
    """Zaman damgalarına göre doğru hız interpolasyonu yapılmalıdır."""
    # 10 mm = 0.01 m in 1 s -> 0.01 m/s
    # 20 mm = 0.02 m in 1 s -> 0.02 m/s
    events = [
        StatusEvent(event="status", elapsed_time_ms=1000, current_position=0),
        StatusEvent(event="status", elapsed_time_ms=2000, current_position=4000),
        StatusEvent(event="status", elapsed_time_ms=3000, current_position=12000),
    ]

    # Aralık dışı sorgular None dönmelidir
    assert interpolate_ground_truth_speed(500, events) is None
    assert interpolate_ground_truth_speed(3500, events) is None

    # Aralık içi sorgu: 1500 ms (1000-2000 aralığı) -> 0.01 m/s
    v1 = interpolate_ground_truth_speed(1500, events)
    assert v1 == pytest.approx(0.01)

    # 2500 ms (2000-3000 aralığı) -> 0.02 m/s
    v2 = interpolate_ground_truth_speed(2500, events)
    assert v2 == pytest.approx(0.02)
