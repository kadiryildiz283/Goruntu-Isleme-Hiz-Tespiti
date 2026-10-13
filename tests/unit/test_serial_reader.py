"""Unit tests for vehicle_speed.telemetry.serial_reader module with mock serial."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from vehicle_speed.telemetry import serial_reader
from vehicle_speed.telemetry.protocol import StartupEvent, StatusEvent


class FakeSerial:
    """Mock serial.Serial device for headless unit testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.is_open = True
        self._lines = [
            b'{"event": "startup", "message": "test started", "max_speed": 1000.0}\n',
            b'{"event": "status", "elapsed_time_ms": 5000, "current_position": 80000}\n',
            b"",
        ]
        self._idx = 0

    def __enter__(self) -> "FakeSerial":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.is_open = False

    def readline(self) -> bytes:
        if self._idx < len(self._lines):
            line = self._lines[self._idx]
            self._idx += 1
            if not line:
                self.is_open = False
            return line
        self.is_open = False
        return b""


def test_stream_telemetry_with_mock_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock seri cihaz üzerinden telemetri olaylarının akışı doğrulanır."""
    import sys

    mock_serial_mod = MagicMock()
    mock_serial_mod.Serial = FakeSerial
    mock_serial_mod.SerialException = Exception
    monkeypatch.setitem(sys.modules, "serial", mock_serial_mod)

    events = list(serial_reader.stream_telemetry(port="MOCK_PORT"))
    assert len(events) == 2
    assert isinstance(events[0], StartupEvent)
    assert isinstance(events[1], StatusEvent)
    assert events[0].max_speed == 1000.0
    assert events[1].current_position == 80000


def test_record_telemetry_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Telemetri olaylarının dosyaya güvenle yazıldığı doğrulanır."""
    import sys

    mock_serial_mod = MagicMock()
    mock_serial_mod.Serial = FakeSerial
    mock_serial_mod.SerialException = Exception
    monkeypatch.setitem(sys.modules, "serial", mock_serial_mod)

    out_file = tmp_path / "test_telemetry.log"
    count = serial_reader.record_telemetry_to_file(
        output_path=out_file,
        port="MOCK_PORT",
        max_events=2,
    )
    assert count == 2
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "StartupEvent" in content
    assert "StatusEvent" in content
