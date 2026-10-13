# Mevcut Envanter: Diskte Gerçekte Ne Var?

**Kanıt tarihi:** 2026-10-13
**Yöntem:** `find`, `du`, `wc -l`, `diff`, `file`, araç zinciri taraması ve dosya okuma.

Bu rapor, altyapı kararlarının dayandığı gerçek durumu belgeler. Hiçbir satır
tahmin değildir; her bulgu bir komut çıktısından gelir.

---

## 1. Sürüm kontrolü durumu — `YOK` (bozuk)

Depoda `.git` dizini var ama git onu bir depo olarak tanımıyor.

```console
$ git status
fatal: Bir git deposu veya (/ bağlama noktasına kadar olan üstü) değil
$ echo $?
128
```

`.git/` içeriği:

```text
COMMIT_EDITMSG    9 bayt    (2025-07-02)
config          314 bayt    (2026-03-10)
description      73 bayt
HEAD             21 bayt
hooks/          556 bayt
index          8851 bayt    (2025-07-02)
info/            14 bayt
objects/          8 bayt    ← BOŞ
packed-refs     112 bayt
```

**Kök neden:** `objects/` dizini boş ve `refs/` dizini hiç yok. Nesne veritabanı
silinmiş veya hiç kopyalanmamış. `index` ve `packed-refs` yetim kalmış.

**Sonuç:** Commit geçmişi yok, branch yok, `git blame` yok, geri alma noktası yok.
CI/CD kurulmadan önce bu durumun çözülmesi zorunlu, çünkü GitHub Actions uzak bir
depoya push edilmiş commit'ler üzerinde çalışır.

**Çözüm yolu:** `06_veri_model_yonetimi/01_gitignore_lfs_veri_izlenebilirligi.md`
bölüm 4'te adım adım anlatıldı. Özet: mevcut `.git` silinmeden `.git.broken`
olarak yeniden adlandırılır, `.gitignore` yazılır, sonra `git init` ile temiz depo
başlatılır.

---

## 2. Kaynak kod envanteri — `VAR`

| Dosya | Satır | İçerik | Durum |
| --- | --- | --- | --- |
| `kadir.py` | 232 | `BallTracker3D` — top takibi, pinhole derinlik, 3B hız | `VAR` |
| `1_donem_kadir.py` | 243 | `SpeedEstimator3D` — araç plakası/far tabanlı hız | `VAR` |
| `listen.py` | 24 | Seri port dinleyici (ham `termios`) | `VAR` |
| `esp32.c` | 108 | ESP32 + TMC2208 vidalı mil sürücüsü (kök kopya) | `VAR` |
| `firmware/src/main.cpp` | ~110 | Aynı firmware'in PlatformIO sürümü | `VAR` |
| `firmware/platformio.ini` | 9 | `esp32doit-devkit-v1`, AccelStepper `^1.64` | `VAR` |

### 2.1 `kadir.py` — `BallTracker3D`

```python
BALL_WIDTH_CM: float = 18.5        # gerçek fiziksel genişlik
MATH_INTERVAL: int = 5             # ağır 3B hesap her N karede bir
TARGET_CLASS_ID: int = 32          # COCO 'sports ball'
```

Metotlar: `calibrate_camera`, `estimate_depth`, `calculate_3d_coords`,
`compute_speed`, `process_frame`, `_cleanup_history`.

Derinlik modeli: `Z = (f * BALL_WIDTH_CM) / pixel_w`. Kalibrasyon ilk tespitte
dinamik yapılıyor: `f = (pixel_w * known_distance_cm) / BALL_WIDTH_CM`.

Hız türetimi: iki ölçüm arası `scipy.spatial.distance.euclidean` ile 3B öklid
mesafesi, `delta_t = frame_diff / fps`. **Kalman filtresi yok**, **öğrenen
kalibratör yok**.

### 2.2 `1_donem_kadir.py` — `SpeedEstimator3D`

```python
PLATE_WIDTH_M: float = 0.520       # TR plaka genişliği
HEADLIGHT_WIDTH_M: float = 1.500   # farlar arası mesafe
KMH_CONVERSION: float = 3.6
```

İki YOLO modeli birlikte yükleniyor: plaka dedektörü (`detector.pt`) ve 14
anahtar noktalı poz modeli (`best.pt`). Plaka kutusu 15 pikselden küçükse
`_get_fallback_headlight_width` devreye girip far anahtar noktaları (indeks 4 ve 5)
arasındaki piksel mesafesini ölçüyor.

Kamera iç parametreleri HFOV'dan türetiliyor:

```python
estimator.initialize_camera(hfov=85.0, width=width, height=height, fps=fps)
```

Kodda bu değerin uydurma olduğu açıkça yazılı:

```python
# NOTE: 85.0 is a placeholder. Inject actual camera datasheet HFOV here.
```

Bu, makale açısından ölçülebilir bir belirsizlik kaynağı ve `PLANLANIYOR`
kalibrasyon prosedürünün gerekçesi.

### 2.3 `firmware/` — ESP32 telemetri kaynağı

`platformio.ini`:

```ini
[env:esp32doit-devkit-v1]
platform = espressif32
board = esp32doit-devkit-v1
framework = arduino
monitor_speed = 115200
upload_port = /dev/ttyUSB0
monitor_port = /dev/ttyUSB0
lib_deps =
    waspinator/AccelStepper @ ^1.64
```

Mekanik sabitler: `STEPS_PER_MM = 400`, `TRAVEL_DISTANCE_MM = 500`,
`TEST_DURATION_MS = 300000`, başlangıç hızı `1000 step/s`, artış `500 step/s`,
tavan `8000 step/s`.

Üretilen JSON telemetri şeması (4 olay tipi):

| `event` | Alanlar | Üretim koşulu |
| --- | --- | --- |
| `startup` | `message`, `max_speed` | `setup()` sonunda bir kez |
| `direction_change` | `new_target_steps`, `new_max_speed`, `acceleration` | her uç noktada |
| `status` | `elapsed_time_ms`, `current_position` | 5000 ms'de bir |
| `complete` | `message` | 300000 ms dolduğunda |

Bu şema, test katmanındaki mock seri port üreticisinin birebir taklit etmesi
gereken sözleşmedir. Detay: `04_test_mimarisi/02_mock_kamera_seri_port.md`.

### 2.4 Kopya kod: `esp32.c` ↔ `firmware/src/main.cpp`

```console
$ diff esp32.c firmware/src/main.cpp | grep -c '^[<>]'
10
```

Farkın tamamı yorum bloğu biçimi (` * *` yerine ` *`) ve `main.cpp`'deki ek
`#include <Arduino.h>` satırı. Yani iki dosya işlevsel olarak aynı kodun iki
kopyası. Tek kaynak `firmware/src/main.cpp` olmalı, `esp32.c` kaldırılmalı.
Aksi halde biri düzeltilip diğeri unutulacak.

---

## 3. İkili dosya ve medya envanteri — depo için `YASAKLI`

1 MB üzerindeki dosyalar:

| Boyut | Dosya |
| --- | --- |
| 273 MB | `Yapay zeka/yolov8x.onnx` |
| 137 MB | `Yapay zeka/yolov8x.pt` |
| 74 MB | `fotograflar/pciletop.mkv` |
| 53 MB | `best.pt` |
| 51 MB | `fotograflar/top.mp4` |
| 51 MB | `fotograflar/top2.mp4` |
| 51 MB | `fotograflar/22.mp4` |
| 24 MB | `Yapay zeka/output.mp4` |
| 23 MB | `yolov8s.pt` (kökte) |
| 23 MB | `Yapay zeka/yolov8s.pt` (kopya) |
| 19 MB | `Yapay zeka/Trafik.mp4` |
| 9.7 MB | `traffic_test.mp4`, `Yapay zeka/8.mp4` |
| 9.1 MB | `firmware/.pio/build/.../firmware.map` |
| 8.1 MB | `kadir_yildiz_2510101209_sunum.pdf` |
| 6.5 MB | `yolov8n.pt` |
| 6.2 MB | `detector.pt` |
| 6.1 MB | `firmware/.pio/build/.../firmware.elf` |

Dizin hacimleri:

```console
511M    Yapay zeka
303M    fotograflar
 23M    firmware        ← .pio derleme çıktıları dahil
188M    (kök dizin)
```

GitHub'ın dosya başına sert sınırı 100 MB, uyarı eşiği 50 MB. `yolov8x.onnx`
(273 MB) ve `yolov8x.pt` (137 MB) bu sınırı aşıyor; push doğrudan reddedilir.
Toplam ~1.3 GB'lık hacim ise Git LFS kotasını (ücretsiz 1 GB depolama / 1 GB
bant genişliği) tek başına tüketir. Politika: `06_veri_model_yonetimi/`.

---

## 4. Sanal ortam ve eski script arşivi

`Yapay zeka/` dizini 511 MB ve içinde bir sanal ortam var:

```text
Yapay zeka/my_new_venv/       ← Python venv, repoya girmesi yasak
Yapay zeka/yolov8x.onnx       ← 273 MB
Yapay zeka/pip                ← tek dosya, muhtemelen yanlış yönlendirme
Yapay zeka/displacement_calculator
Yapay zeka/test_opencv
Yapay zeka/deneme.cpp, sonkod.cpp
Yapay zeka/common.py, deneme.py, deneme2.py, hiz_testi.py, hocam.py,
             main.py, sonkod.py, speed.py, top.py, top_nocomment.py,
             top_nocomment2.py, tst_scene_render.py, video.py
```

Bunlar projenin önceki deneme aşamaları. Silinmemeli (araştırma geçmişi değerli),
ama ana paketin içine de karışmamalı. Öneri: `archive/legacy_scripts/` altına
taşınıp `.gitignore` ile depodan çıkarılsın ya da ayrı bir `research-archive`
dalında tutulsun.

---

## 5. Adlandırma anomalileri

| Öğe | Sorun |
| --- | --- |
| `,` (13658 bayt) | `file` çıktısı: "Python script, UTF-8 text executable". İçinde `import cv2 / numpy / ultralytics / math / time` var. Bir kabuk yönlendirme hatası sonucu virgül adıyla oluşmuş yetim script. |
| `ersanhocama` (311 bayt, uzantısız) | Uzantısız metin dosyası |
| `ersanhocam.md` (88 bayt) | Neredeyse boş not |
| `Yapay zeka/`, `fotograflar/` | Boşluk ve Türkçe karakter içeren dizin adları; CI'da tırnaklama hatalarına yol açar |
| `yapıştırılmış dosya.png` | Boşluklu + Türkçe karakterli dosya adı |
| `Araç Hız Tespiti Makale Tamamlama_(1).pdf` | Boşluk, Türkçe karakter, parantez |

Depo içi yollar ASCII ve boşluksuz olmalı. Gerekçe ve eşleme tablosu:
`01_dizin_yapisi/01_hedef_monorepo.md`.

---

## 6. Yerel araç zinciri

| Araç | Sürüm | Durum | Not |
| --- | --- | --- | --- |
| `python3` | 3.14.7 | `VAR` | Sistem Python'u; hedef matris (3.10-3.12) ile uyumsuz |
| `git` | 2.56.0 | `VAR` | |
| `uv` | 0.13.0 | `VAR` | Paket yöneticisi olarak kullanılabilir |
| `ruff` | 0.16.7 | `VAR` | CLI kurulu, ama paket olarak değil |
| `gh` | 2.102.0 | `VAR` | Label/issue/branch-protection otomasyonu için yeterli |
| `clang-format` | 23.1.1 | `VAR` | Firmware biçimlendirme için hazır |
| `mypy` | — | `YOK` | kurulmalı |
| `pytest` | — | `YOK` | kurulmalı |
| `pre-commit` | — | `YOK` | kurulmalı |
| `platformio` | — | `YOK` | firmware CI için gerekli |
| `cppcheck` | — | `YOK` | firmware statik analizi için gerekli |

Python paket durumu (sistem `python3`):

| Paket | Sürüm | Durum |
| --- | --- | --- |
| `opencv-python` | 5.0.0.93 | `VAR` |
| `numpy` | 2.5.1 | `VAR` |
| `pyserial` | 3.5 | `VAR` (ama `listen.py` kullanmıyor) |
| `torch` | — | `YOK` |
| `ultralytics` | — | `YOK` |
| `scipy` | — | `YOK` |

`kadir.py` hem `ultralytics` hem `scipy` import ediyor, ikisi de kurulu değil.
Yani script bugün bu makinede çalıştırılamaz. İzole bir sanal ortam
(`uv venv`) kurulumu ilk uygulama adımıdır.

---

## 7. Altyapı dosyaları — tamamı `YOK`

Diskte doğrulanmış yokluk listesi:

```text
pyproject.toml                  YOK
setup.py / setup.cfg            YOK
requirements.txt                YOK
uv.lock / poetry.lock           YOK
.gitignore                      YOK
.gitattributes                  YOK
.pre-commit-config.yaml         YOK
.github/                        YOK
  workflows/ci.yml              YOK
  ISSUE_TEMPLATE/               YOK
  PULL_REQUEST_TEMPLATE.md      YOK
  labeler.yml                   YOK
tests/                          YOK
conftest.py                     YOK
README.md                       YOK
LICENSE                         YOK
CONTRIBUTING.md                 YOK
SECURITY.md                     YOK
CODE_OF_CONDUCT.md               YOK
CITATION.cff                    YOK
Makefile / justfile             YOK
Dockerfile                      YOK
```

Mevcut tek dokümantasyon: `BAGLANTI_VE_KURULUM_REHBERI.md` (232 satır, donanım
bağlantı rehberi) ve `Q2_MAKALE_STRATEJISI_VE_YAPAY_ZEKA_RAPORU.md` (509 satır,
makale stratejisi). İkisi de korunmalı ve yeni `README.md`'den bağlanmalı.

**Sonuç:** Altyapı sıfırdan kurulacak. Bu bir dezavantaj değil; mevcut CI
kurallarına uyum sağlamak için eski yapılandırma yıkmak gerekmiyor.
