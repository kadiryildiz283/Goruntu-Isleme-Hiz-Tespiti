"""Central configuration parameters and physical constants.

Defines default dimensions, camera specifications, and telemetry hardware limits.
Every parameter is strictly typed and documented with its physical unit.
"""

import math
from dataclasses import dataclass
from typing import Final

# ============================================================================
# Fiziksel Referans Boyutları (Metre cinsinden)
# ============================================================================

# TR standart araç plaka genişliği (520 mm)
DEFAULT_PLATE_WIDTH_M: Final[float] = 0.520

# Standart binek araç iki far arası ortalama mesafe (1500 mm)
DEFAULT_HEADLIGHT_WIDTH_M: Final[float] = 1.500

# Laboratuvar testlerinde kullanılan spor topu çapı (18.5 cm)
DEFAULT_BALL_WIDTH_M: Final[float] = 0.185

# ============================================================================
# Optik ve Kamera Varsayılanları
# ============================================================================

# Ölçülmemiş varsayılan HFOV değeri (derece).
# 1_donem_kadir.py içinde "placeholder" olarak işaretlenmişti.
# Gerçek kamera datasheet HFOV değeri ile değiştirilmeli.
DEFAULT_HFOV_DEG: Final[float] = 85.0
DEFAULT_HFOV_RAD: Final[float] = math.radians(DEFAULT_HFOV_DEG)

# Varsayılan video akış hızı (FPS)
DEFAULT_FPS: Final[float] = 30.0

# ============================================================================
# ESP32 Telemetri ve Mekanik Sabitler
# ============================================================================

# Step motor - vidalı mil dönüşüm oranı (firmware/src/main.cpp ile birebir eşleşik)
STEPS_PER_MM: Final[int] = 400

# Ray hareket mesafesi (mm)
TRAVEL_DISTANCE_MM: Final[int] = 500

# Seri port varsayılanları
DEFAULT_SERIAL_PORT: Final[str] = "/dev/ttyUSB0"
DEFAULT_BAUD_RATE: Final[int] = 115200
DEFAULT_SERIAL_TIMEOUT_S: Final[float] = 1.0


@dataclass(frozen=True)
class AppConfig:
    """Uygulama çalışma zamanı konfigürasyonu.

    Attributes:
        camera_hfov_deg: Kamera yatay görüş açısı (derece).
        target_fps: Beklenen video akış karesi / saniye.
        serial_port: ESP32 seri bağlantı port yolu.
        baud_rate: Seri haberleşme hızı.
        min_detection_confidence: YOLO filtreleme eşiği.
        max_track_age_frames: Bellekten silinmeden önceki kayıp kare sayısı.
    """

    camera_hfov_deg: float = DEFAULT_HFOV_DEG
    target_fps: float = DEFAULT_FPS
    serial_port: str = DEFAULT_SERIAL_PORT
    baud_rate: int = DEFAULT_BAUD_RATE
    min_detection_confidence: float = 0.40
    max_track_age_frames: int = 150
