"""Unit tests for vehicle_speed.telemetry.protocol parser."""

from vehicle_speed.telemetry.protocol import (
    CompleteEvent,
    DirectionChangeEvent,
    StartupEvent,
    StatusEvent,
    parse_line,
)


def test_parse_startup_event() -> None:
    """Startup olayı ayrıştırma testi."""
    line = '{"event": "startup", "message": "5 dakikalik test basladi", "max_speed": 1000.0}'
    event = parse_line(line)
    assert isinstance(event, StartupEvent)
    assert event.event == "startup"
    assert event.message == "5 dakikalik test basladi"
    assert event.max_speed == 1000.0


def test_parse_direction_change_event() -> None:
    """Direction change olayı ayrıştırma testi."""
    line = (
        '{"event": "direction_change", "new_target_steps": 200000, '
        '"new_max_speed": 1500.0, "acceleration": 750.0}'
    )
    event = parse_line(line)
    assert isinstance(event, DirectionChangeEvent)
    assert event.new_target_steps == 200000
    assert event.new_max_speed == 1500.0
    assert event.acceleration == 750.0


def test_parse_status_event() -> None:
    """Status olayı ayrıştırma testi."""
    line = '{"event": "status", "elapsed_time_ms": 5002, "current_position": 14250}'
    event = parse_line(line)
    assert isinstance(event, StatusEvent)
    assert event.elapsed_time_ms == 5002
    assert event.current_position == 14250


def test_parse_complete_event() -> None:
    """Complete olayı ayrıştırma testi."""
    line = '{"event": "complete", "message": "300000ms doldu. Motor durduruldu."}'
    event = parse_line(line)
    assert isinstance(event, CompleteEvent)
    assert event.event == "complete"
    assert "Motor durduruldu" in event.message


def test_parse_malformed_json_returns_none() -> None:
    """Bozuk veya kısmi JSON satırlarında None dönmeli, istisna fırlatmamalıdır."""
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("{'event': 'status', broken") is None
    assert parse_line('{"event": "unknown_type"}') is None
    assert parse_line("sadece duz yazi") is None
    assert parse_line('{"event": "status", "elapsed_time_ms": "not_an_int"}') is None
