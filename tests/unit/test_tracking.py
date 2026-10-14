"""Unit tests for vehicle_speed.tracking module."""

import pytest

from vehicle_speed.tracking import SpeedTracker


def test_first_update_returns_zero_speed() -> None:
    """İlk tespit anında delta_t veya delta_d olmadığı için hız 0.0 dönmelidir."""
    tracker = SpeedTracker()
    speed = tracker.update(track_id=1, position_m=(0.0, 0.0, 10.0), frame_index=0, fps=30.0)
    assert speed == 0.0
    assert tracker.active_track_count == 1


def test_constant_velocity_converges_to_true_speed() -> None:
    """30 FPS'te her karede 1 metre ilerleyen nesnenin hızı 30 m/s olmalıdır."""
    tracker = SpeedTracker(math_interval=1)
    fps = 30.0
    speeds: list[float] = []

    for frame in range(10):
        # Nesne her kare 1 metre yaklaşıyor
        z = 20.0 - (frame * 1.0)
        v = tracker.update(track_id=1, position_m=(0.0, 0.0, z), frame_index=frame, fps=fps)
        speeds.append(v)

    # İlk kareden sonraki tüm hızlar 30.0 m/s olmalı (1 m / (1/30 s))
    for v in speeds[1:]:
        assert v == pytest.approx(30.0, rel=1e-6)


def test_tracking_rejects_non_positive_fps() -> None:
    """Sıfır veya negatif FPS verildiğinde ValueError fırlatılır."""
    tracker = SpeedTracker()
    with pytest.raises(ValueError, match=r"FPS.*pozitif olmalıdır"):
        tracker.update(track_id=1, position_m=(0.0, 0.0, 10.0), frame_index=1, fps=0.0)


def test_math_interval_caching() -> None:
    """math_interval dolana kadar son hesaplanan hız önbellekten döndürülür."""
    tracker = SpeedTracker(math_interval=5)
    fps = 30.0

    tracker.update(1, (0.0, 0.0, 10.0), frame_index=0, fps=fps)
    # 5. karede hız hesaplanır: 5 metre / (5/30 s) = 30 m/s
    v_frame5 = tracker.update(1, (0.0, 0.0, 5.0), frame_index=5, fps=fps)
    assert v_frame5 == pytest.approx(30.0)

    # 6. ve 7. karede math_interval (5) henüz dolmadı; önceki hız korunur
    v_frame6 = tracker.update(1, (0.0, 0.0, 4.0), frame_index=6, fps=fps)
    v_frame7 = tracker.update(1, (0.0, 0.0, 3.0), frame_index=7, fps=fps)
    assert v_frame6 == v_frame5
    assert v_frame7 == v_frame5


def test_tracker_zero_delta_t_preserves_speed() -> None:
    """Aynı kare indeksinde art arda çağrıldığında sıfıra bölme engellenir."""
    tracker = SpeedTracker(math_interval=0)
    tracker.update(1, (0.0, 0.0, 10.0), frame_index=1, fps=30.0)
    # Aynı kare içinde tekrar çağrılırsa delta_t <= 0 olur ve önceki hız döner
    v = tracker.update(1, (0.0, 0.0, 9.0), frame_index=1, fps=30.0)
    assert v == 0.0


def test_prune_removes_stale_tracks_and_returns_count() -> None:
    """Belirli kare sayısı boyunca güncellenmeyen izler temizlenir ve silinen adet döner."""
    tracker = SpeedTracker()
    tracker.update(1, (0.0, 0.0, 10.0), frame_index=10, fps=30.0)
    tracker.update(2, (0.0, 0.0, 15.0), frame_index=100, fps=30.0)

    assert tracker.active_track_count == 2

    # 150 kare eşiği ile 170. karede prune çağrılır
    # ID 1 için (170 - 10) = 160 > 150 -> SİLİNMELİ
    # ID 2 için (170 - 100) = 70 <= 150 -> KALMALI
    pruned = tracker.prune(current_frame_index=170, max_age_frames=150)
    assert pruned == 1
    assert tracker.active_track_count == 1
    assert tracker.get_track(1) is None
    assert tracker.get_track(2) is not None


def test_tracker_clear() -> None:
    """clear() metodu tüm izleri temizler."""
    tracker = SpeedTracker()
    tracker.update(1, (0.0, 0.0, 10.0), frame_index=0, fps=30.0)
    tracker.clear()
    assert tracker.active_track_count == 0
