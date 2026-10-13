"""Integration test verifying offline telemetry log replay."""

from pathlib import Path

from vehicle_speed.telemetry.ground_truth import ground_truth_speed
from vehicle_speed.telemetry.protocol import (
    CompleteEvent,
    DirectionChangeEvent,
    StartupEvent,
    StatusEvent,
    parse_line,
)


def test_telemetry_fixture_replay() -> None:
    """Fixture dosyasındaki tüm satırların hatasız ayrıştırıldığını doğrular."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "telemetry_sample.jsonl"
    assert fixture_path.exists(), "Telemetry fixture dosyası bulunamadı."

    events = []
    with fixture_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            event = parse_line(line_str)
            assert event is not None, f"Ayrıştırılamayan satır: {line_str}"
            events.append(event)

    assert len(events) == 11
    assert isinstance(events[0], StartupEvent)
    assert isinstance(events[-1], CompleteEvent)
    assert any(isinstance(e, DirectionChangeEvent) for e in events)

    status_events = [e for e in events if isinstance(e, StatusEvent)]
    assert len(status_events) == 8

    # İki status olayı arasındaki yer gerçeği hız hesaplamasını doğrula
    speeds = ground_truth_speed(status_events)
    assert len(speeds) == 7
    # İlk aralık: t=2.0s, v = (6000 - 2500) / (400 * 1000) / 1.0 = 0.00875 m/s
    t0, v0 = speeds[0]
    assert t0 == 2.0
    assert abs(v0 - 0.00875) < 1e-5
