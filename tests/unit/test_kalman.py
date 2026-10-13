"""Unit tests for vehicle_speed.filters.kalman module."""

import numpy as np
import pytest

from vehicle_speed.filters.kalman import KalmanFilter3D


def test_kalman_initialization() -> None:
    """KalmanFilter3D başlangıç durum vektörü ve kovaryans doğrulaması."""
    kf = KalmanFilter3D(initial_position=(1.0, 2.0, 10.0))
    x, y, z = kf.position
    vx, vy, vz = kf.velocity

    assert (x, y, z) == (1.0, 2.0, 10.0)
    assert (vx, vy, vz) == (0.0, 0.0, 0.0)
    assert kf.speed_ms == 0.0


def test_kalman_rejects_non_positive_dt() -> None:
    """dt <= 0 durumunda predict metodu ValueError fırlatır."""
    kf = KalmanFilter3D(initial_position=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="dt pozitif olmalıdır"):
        kf.predict(dt=0.0)
    with pytest.raises(ValueError, match="dt pozitif olmalıdır"):
        kf.predict(dt=-0.1)


def test_kalman_converges_to_true_velocity() -> None:
    """Sabit 10 m/s hızla yaklaşan nesnenin hızına filtrenin yakınsaması."""
    true_v = 10.0
    dt = 1.0 / 30.0  # 30 FPS
    initial_z = 50.0

    kf = KalmanFilter3D(initial_position=(0.0, 0.0, initial_z))

    # 60 adım simülasyon (2 saniye)
    for step in range(1, 60):
        kf.predict(dt=dt)
        # Gerçek konum: z = 50 - 10 * (step * dt)
        true_pos = (0.0, 0.0, initial_z - true_v * (step * dt))
        kf.update(measurement=true_pos)

    # Filtre hız vektörü Z ekseninde -10 m/s büyüklükte yakınsamalı
    assert kf.speed_ms == pytest.approx(true_v, rel=0.05)


def test_kalman_attenuates_measurement_jitter() -> None:
    """Kalman filtresi gürültülü optik ölçümleri sönümlemelidir (varyans azalması)."""
    np.random.seed(42)
    dt = 0.033
    true_z = 20.0
    kf = KalmanFilter3D(initial_position=(0.0, 0.0, true_z), measurement_noise_std=1.0)

    raw_measurements = []
    filtered_positions = []

    for _ in range(50):
        # ±0.5 metre gürültülü ölçüm
        noisy_z = true_z + float(np.random.normal(0, 0.5))
        raw_measurements.append(noisy_z)

        kf.predict(dt=dt)
        kf.update((0.0, 0.0, noisy_z))
        filtered_positions.append(kf.position[2])

    raw_var = float(np.var(raw_measurements))
    filtered_var = float(np.var(filtered_positions[15:]))

    # Filtrelenmiş sinyalin varyansı ham ölçümden belirgin şekilde düşük olmalı
    assert filtered_var < raw_var
