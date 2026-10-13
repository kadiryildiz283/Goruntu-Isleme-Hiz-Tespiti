"""Physical motion ground-truth derivations from ESP32 lead-screw stepper events.

Converts stepper motor step units to metric measurements and derives true physical speed.
"""

from collections.abc import Sequence

from vehicle_speed.config import STEPS_PER_MM, TRAVEL_DISTANCE_MM
from vehicle_speed.telemetry.protocol import StatusEvent


def steps_to_metres(steps: int, steps_per_mm: int = STEPS_PER_MM) -> float:
    r"""Step motor adımlarını metre cinsinden fiziksel mesafeye dönüştürür.

    Matematiksel formül:
        d = \frac{\text{steps}}{\text{steps\_per\_mm} \cdot 1000}

    Args:
        steps: Step motor adım sayısı.
        steps_per_mm: Milimetre başına düşen adım (varsayılan: 400).

    Returns:
        float: Metre cinsinden mesafe.

    Raises:
        ValueError: steps_per_mm <= 0 ise (Fail-Closed).
    """
    if steps_per_mm <= 0:
        msg = f"steps_per_mm pozitif olmalıdır, alınan: {steps_per_mm}"
        raise ValueError(msg)

    return steps / (steps_per_mm * 1000.0)


def ground_truth_speed(status_events: Sequence[StatusEvent]) -> list[tuple[float, float]]:
    r"""Ardışık durum telemetrilerinden yer gerçeği (ground truth) hızını türetir.

    Matematiksel formül:
        \Delta d = \text{metres}(pos_2) - \text{metres}(pos_1)
        \Delta t = \frac{t_2 - t_1}{1000}
        v = \frac{|\Delta d|}{\Delta t}

    Args:
        status_events: Zaman damgalı StatusEvent nesneleri dizisi.

    Returns:
        list[tuple[float, float]]: [(zaman_saniye, gercek_hiz_m_s), ...] listesi.
    """
    if len(status_events) < 2:
        return []

    speed_profile: list[tuple[float, float]] = []

    for i in range(1, len(status_events)):
        prev = status_events[i - 1]
        curr = status_events[i]

        delta_t_s = (curr.elapsed_time_ms - prev.elapsed_time_ms) / 1000.0
        if delta_t_s <= 0.0:
            continue

        prev_pos_m = steps_to_metres(prev.current_position)
        curr_pos_m = steps_to_metres(curr.current_position)

        delta_d_m = abs(curr_pos_m - prev_pos_m)
        v_ms = delta_d_m / delta_t_s

        timestamp_s = curr.elapsed_time_ms / 1000.0
        speed_profile.append((timestamp_s, v_ms))

    return speed_profile


def check_travel_boundary(steps: int) -> bool:
    """Konumun vidalı mil sınırları (0 - TRAVEL_DISTANCE_MM) içinde olup olmadığını doğrular."""
    pos_mm = steps / STEPS_PER_MM
    return 0.0 <= pos_mm <= float(TRAVEL_DISTANCE_MM)
