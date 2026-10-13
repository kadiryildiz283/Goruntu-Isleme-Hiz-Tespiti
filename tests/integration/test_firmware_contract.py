"""Integration test verifying firmware C header and Python protocol constants match."""

import re
from pathlib import Path

from vehicle_speed.telemetry import protocol


def test_schema_constants_match_firmware() -> None:
    """firmware/include/telemetry_protocol.h ile protocol.py sabitleri eşit olmalıdır."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    header_path = repo_root / "firmware" / "include" / "telemetry_protocol.h"
    assert header_path.is_file(), f"Başlık dosyası bulunamadı: {header_path}"

    header_content = header_path.read_text(encoding="utf-8")

    firmware_values: dict[str, int] = {}
    for name in ("STEPS_PER_MM", "TRAVEL_DISTANCE_MM", "TELEMETRY_SCHEMA_VERSION"):
        match = re.search(rf"#define\s+{name}\s+(\d+)", header_content)
        assert match is not None, f"{name} sabiti C başlığında bulunamadı!"
        firmware_values[name] = int(match.group(1))

    assert firmware_values["STEPS_PER_MM"] == protocol.STEPS_PER_MM
    assert firmware_values["TRAVEL_DISTANCE_MM"] == protocol.TRAVEL_DISTANCE_MM
    assert firmware_values["TELEMETRY_SCHEMA_VERSION"] == protocol.SCHEMA_VERSION
