"""Object trajectory tracking and 3D velocity derivation.

Calculates finite difference speeds between consecutive spatial coordinates
and maintains active tracking histories with deterministic pruning.
"""

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrackState:
    """Tek bir izlenen nesnenin anlık uzamsal durumu ve geçmişi.

    Attributes:
        last_position_m: Kamera koordinatlarında son (X, Y, Z) metre konumu.
        last_frame_index: Nesnenin en son görüldüğü kare indeksi.
        last_speed_ms: En son hesaplanan anlık 3B hız (m/s).
        history: Son N konumun kuyruğu [(frame_idx, (X, Y, Z))].
    """

    last_position_m: tuple[float, float, float]
    last_frame_index: int
    last_speed_ms: float = 0.0
    history: deque[tuple[int, tuple[float, float, float]]] = field(
        default_factory=lambda: deque(maxlen=30),
    )


class SpeedTracker:
    """3 Boyutlu uzayda nesne takibi ve anlık hız hesaplama yöneticisi."""

    def __init__(self, math_interval: int = 1) -> None:
        """SpeedTracker ilklendirici.

        Args:
            math_interval: Hız hesabının kaç karede bir güncelleneceği.
                1 = her karede hesapla (akıcı).
        """
        self._tracks: dict[int, TrackState] = {}
        self.math_interval: int = max(1, math_interval)

    def update(
        self,
        track_id: int,
        position_m: tuple[float, float, float],
        frame_index: int,
        fps: float,
    ) -> float:
        r"""Nesne konumunu günceller ve 3B hızını türetir.

        Matematiksel formül:
            \Delta d = \sqrt{(X_2 - X_1)^2 + (Y_2 - Y_1)^2 + (Z_2 - Z_1)^2}
            \Delta t = \frac{\Delta \text{frame}}{fps}
            v = \frac{\Delta d}{\Delta t}

        Args:
            track_id: YOLO nesne takip kimliği.
            position_m: 3B kamera uzayındaki konum (X, Y, Z) metre.
            frame_index: Mevcut video kare indeksi.
            fps: Video akış kare hızı (kare/saniye).

        Returns:
            float: Anlık 3B hız (m/s). İlk tespit anında 0.0 döner.

        Raises:
            ValueError: fps <= 0 olduğunda (Fail-Closed).
        """
        if fps <= 0.0:
            msg = f"Video kare hızı (FPS) pozitif olmalıdır, alınan: {fps}"
            raise ValueError(msg)

        if track_id not in self._tracks:
            # İlk tespit: geçmişi başlat, hız henüz sıfır
            state = TrackState(
                last_position_m=position_m,
                last_frame_index=frame_index,
                last_speed_ms=0.0,
            )
            state.history.append((frame_index, position_m))
            self._tracks[track_id] = state
            return 0.0

        state = self._tracks[track_id]
        frame_diff = frame_index - state.last_frame_index

        if frame_diff < self.math_interval:
            # Hesaplama aralığı dolmadıysa son hızı koru
            state.history.append((frame_index, position_m))
            return state.last_speed_ms

        # 3B Öklid mesafesi
        x1, y1, z1 = state.last_position_m
        x2, y2, z2 = position_m
        delta_d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

        delta_t = frame_diff / fps
        if delta_t <= 0.0:
            return state.last_speed_ms

        speed_ms = delta_d / delta_t

        state.last_position_m = position_m
        state.last_frame_index = frame_index
        state.last_speed_ms = speed_ms
        state.history.append((frame_index, position_m))

        return speed_ms

    def prune(self, current_frame_index: int, max_age_frames: int = 150) -> int:
        """Kayıp veya ekrandan çıkmış izleri bellekten temizler.

        Args:
            current_frame_index: Mevcut video kare numarası.
            max_age_frames: Bir izin korunacağı azami sessiz kare sayısı.

        Returns:
            int: Bellekten silinen iz sayısı.
        """
        stale_ids = [
            tid
            for tid, state in self._tracks.items()
            if (current_frame_index - state.last_frame_index) > max_age_frames
        ]
        for tid in stale_ids:
            del self._tracks[tid]
        return len(stale_ids)

    def get_track(self, track_id: int) -> TrackState | None:
        """Belirli bir izin durumunu sorgular."""
        return self._tracks.get(track_id)

    @property
    def active_track_count(self) -> int:
        """Şu an takip edilen etkin iz sayısı."""
        return len(self._tracks)

    def clear(self) -> None:
        """Tüm takip geçmişini sıfırlar."""
        self._tracks.clear()
