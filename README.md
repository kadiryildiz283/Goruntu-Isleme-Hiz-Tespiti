# Vehicle Speed Estimation (Görüntü İşleme ile Hız Tespiti)

[![CI](https://github.com/kadiryildiz283/Goruntu-Isleme-Hiz-Tespiti/actions/workflows/ci.yml/badge.svg)](https://github.com/kadiryildiz283/Goruntu-Isleme-Hiz-Tespiti/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/)

Tek kamera (monoküler optik geometri) ve derin öğrenme tespiti kullanarak taşıtların 3 boyutlu hızını tahmin eden, doğrulamayı (ground truth) ise ESP32 kontrollü vidalı mil telemetrisi ile sağlayan açık kaynak araştırma ve mühendislik platformu.

---

## 📌 Mimari ve Modül Yapısı

```text
src/vehicle_speed/
├── __init__.py           # Paket sürümü (__version__)
├── py.typed              # PEP 561 tip uyumluluğu
├── config.py             # Merkezi parametreler ve fiziksel sabitler
├── geometry.py           # Saf matematiksel Pinhole ve 3B izdüşüm fonksiyonları
├── tracking.py           # Nesne izleme, yörünge ve 3B hız türetimi
├── calibration.py        # Kamera kalibrasyonu ve ölçek referansları
├── detector.py           # YOLO tabanlı tespit motoru (Soyutlanmış & Fail-Closed)
├── telemetry/            # ESP32 donanım telemetrisi ve yer gerçeği (Ground Truth)
│   ├── protocol.py       # ESP32 JSON telemetri şeması ve olay ayrıştırma
│   ├── ground_truth.py   # Adım (step) -> Milimetre -> m/s hız hesaplama
│   └── serial_reader.py  # Parametrik, güvenli seri port okuyucusu
├── filters/              # Kalman ve gürültü filtreleme modülleri (Planlanıyor)
├── nrc/                  # Nöral Artık Kalibratör (NRC) (Planlanıyor)
├── visualization.py      # OpenCV ekran ve görselleştirme katmanı (GUI izole)
└── cli.py                # Komut satırı çalıştırma arayüzü
```

---

## 🚀 Hızlı Başlangıç

### Gereksinimler

* Python `>=3.10` (Önerilen: `3.12`)
* [uv](https://github.com/astral-sh/uv) paket yöneticisi

### Kurulum

```bash
# Bağımlılıkları ve geliştirici araçlarını yükleyin
uv sync --extra dev

# Kod kalitesi kapılarını çalıştırın
uv run ruff check .
uv run mypy

# Testleri koşun
uv run pytest
```

---

## 📜 Mühendislik Sözleşmesi

Bu depoda çalışan tüm katkıcılar ve otomasyonlar [.coderules](.coderules) belgesine uymakla yükümlüdür.
Detaylı altyapı raporları [docs/infrastructure/](docs/infrastructure/README.md) dizininde yer almaktadır.
