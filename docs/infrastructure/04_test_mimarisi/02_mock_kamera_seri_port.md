# Mock Katmanı: CI'da Donanım Olmadan Test Etmek

CI runner'ında kamera yok, ESP32 yok, seri port yok, görüntü sunucusu yok.
Bu dört fiziksel bağımlılığın her biri için ayrı bir mock stratejisi gerekir.

**Doğrulanmış bağımlılıklar:**

| Bağımlılık | Kod konumu | CI'da durumu |
| --- | --- | --- |
| `cv2.VideoCapture` | `kadir.py`, `1_donem_kadir.py` `__main__` | dosya yoksa `isOpened()` → False |
| `cv2.imshow` | her iki script | runner'da görüntü sunucusu yok |
| `YOLO(...)` | her iki script | `ultralytics` kurulu değilse ImportError; kurulsa bile ağırlık indirir |
| `/dev/ttyUSB0` | `listen.py`, `platformio.ini` | port yok |
| `termios` | `listen.py` satır 2 | **Windows'ta modül hiç yok** |

---

## 1. Seri port mock'u

### 1.1 Neden `termios` mock'lanamaz

`listen.py` ham sistem çağrıları kullanıyor:

```python
fd = os.open('/dev/ttyUSB0', os.O_RDWR | os.O_NOCTTY)
attrs = termios.tcgetattr(fd)
attrs[2] = termios.B115200 | termios.CS8 | termios.CREAD | termios.CLOCAL
termios.tcsetattr(fd, termios.TCSANOW, attrs)
```

`termios` POSIX'e özgü. Windows'ta bu modül **yok** — `import termios` satırında
`ModuleNotFoundError` verilir, mock'lamaya fırsat kalmaz. `unittest.mock.patch`
ile çözülemez, çünkü hata import anında oluşur.

İki seçenek:

1. Koşullu import + platform dalı — kodu karmaşıklaştırır, iki yol da test
   edilmeli
2. `pyserial`'e geç — `pyserial` zaten kurulu (3.5), platformlar arası, ve
   `serial.Serial` tek bir sınıfta mock'lanabilir

**Karar: seçenek 2.** Bu, `A4` borcunun çözümüdür ve Windows matrisinin ön
koşuludur (`03_ci_cd/03_test_matrisi_cache.md` bölüm 5).

### 1.2 Mock okuyucu tasarımı

Test edilebilirlik için okuyucu, veri kaynağını enjeksiyonla almalı:

```python
class TelemetryReader:
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        chunk_iter: Iterable[str] | None = None,   # test enjeksiyonu
    ) -> None:
        self._chunk_iter = chunk_iter
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

    def _open(self) -> Iterator[str]:
        if self._chunk_iter is not None:
            yield from self._chunk_iter
            return
        with serial.Serial(self._port, self._baudrate, timeout=self._timeout) as s:
            buffer = ""
            while True:
                chunk = s.read(256)
                if not chunk:
                    continue
                buffer += chunk.decode("utf-8", errors="replace")
                *lines, buffer = buffer.split("\n")
                yield from lines
```

`chunk_iter` parametresi, `serial`'i hiç mock'lamadan test yazmayı sağlar. Bu,
`unittest.mock` ile sahte nesne kurmaktan daha az kırılgan: mock'un hangi
metotlarının çağrılacağını bilmek gerekmez.

### 1.3 Testler

```python
def test_reader_parses_realistic_firmware_stream() -> None:
    stream = [
        '{"event": "startup", "message": "5 dakikalik test basladi", "max_speed": 1000.00}\n',
        '{"event": "status", "elapsed_time_ms": 5000, "current_position": 45320}\n',
        '{"event": "status", "elapsed_time_ms": 10000, "current_position": 17000}\n',
        '{"event": "complete", "message": "300000ms doldu. Motor durduruldu."}\n',
    ]
    reader = TelemetryReader(chunk_iter=iter(stream))
    events = [e for e in reader.events() if e is not None]
    assert [type(e).__name__ for e in events] == [
        "StartupEvent", "StatusEvent", "StatusEvent", "CompleteEvent",
    ]


def test_reader_reassembles_line_split_across_chunks() -> None:
    """listen.py'nin os.read(fd, 256) davranışının taklidi: satır ortasında bölünme."""
    full = '{"event": "status", "elapsed_time_ms": 5000, "current_position": 45320}\n'
    reader = TelemetryReader(chunk_iter=iter([full[:40], full[40:]]))
    events = [e for e in reader.events() if e is not None]
    assert len(events) == 1
    assert events[0].current_position == 45320


def test_reader_survives_garbage_without_losing_next_event() -> None:
    stream = [
        '{"event": "status", "elapsed_time_ms": 5000, "cur\n',   # bozuk
        'rupt json here\n',
        '{"event": "complete", "message": "bitti"}\n',
    ]
    reader = TelemetryReader(chunk_iter=iter(stream))
    events = [e for e in reader.events() if e is not None]
    assert len(events) == 1
    assert isinstance(events[0], CompleteEvent)


def test_reader_stops_on_complete_event() -> None:
    """complete olayı sonrası okuma sonlanmalı, sonsuz döngüye girmemeli."""
    stream = ['{"event": "complete", "message": "bitti"}\n', '{"event": "status"}\n']
    reader = TelemetryReader(chunk_iter=iter(stream))
    events = list(reader.events())
    assert len(events) == 1
```

Dördüncü test gerçek bir sorunu belgeliyor: mevcut `listen.py` `while True`
içinde sonsuza dek okuyor, `complete` olayını tanımıyor. Firmware testi
bitirdiğinde Python tarafı habersiz beklemeye devam ediyor. Elle `Ctrl-C`
gerekiyor — 5 dakikalık bir deney serisinde bu pratik değil.

### 1.4 `serial.Serial` gerçekten mock'lanacaksa

`chunk_iter` yerine `pytest-mock` kullanılacaksa:

```python
def test_reader_opens_port_with_correct_settings(mocker: MockerFixture) -> None:
    fake = mocker.MagicMock()
    fake.read.side_effect = [b'{"event": "complete"}\n', b""]
    fake.__enter__.return_value = fake
    serial_cls = mocker.patch("vehicle_speed.telemetry.serial_reader.serial.Serial",
                              return_value=fake)

    reader = TelemetryReader(port="/dev/ttyACM0", baudrate=115200, timeout=2.0)
    events = [e for e in reader.events() if e is not None]

    serial_cls.assert_called_once_with("/dev/ttyACM0", 115200, timeout=2.0)
    assert len(events) == 1
```

`assert_called_once_with` port ve baud ayarlarının doğru geçirildiğini kilitler.
`platformio.ini`'deki `monitor_speed = 115200` ile Python tarafının ayrışması
bu testle yakalanır.

---

## 2. Kamera ve video mock'u

### 2.1 `cv2.VideoCapture` yerine kare üreteci

Pipeline'ı video dosyasından ayırmak için giriş bir **kare üreteci** olmalı:

```python
FrameSource = Iterator[np.ndarray]

def frame_source_from_video(path: Path) -> FrameSource:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        msg = f"video açılamadı: {path}"
        raise FileNotFoundError(msg)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()
```

Bu fonksiyon **tek yerde** `cv2.VideoCapture` kullanır. Pipeline'ın geri kalanı
`FrameSource` tipiyle çalışır ve testte sahte üreteç enjekte edilir.

### 2.2 Sentetik kare üretimi

Gerçek video dosyası CI'a girmez (`06_veri_model_yonetimi/`). Test kareleri
üretilir:

```python
@pytest.fixture
def synthetic_frames() -> list[np.ndarray]:
    """640x480 gri kareler, ortada yaklaşan beyaz dikdörtgen."""
    frames = []
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        width = int(200 - i * 4)              # 200 px -> 84 px, yaklaşan nesne
        height = int(width * 0.4)
        x1 = 320 - width // 2
        y1 = 300 - height // 2
        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (255, 255, 255), -1)
        frames.append(frame)
    return frames
```

Bu fixture `cv2.rectangle` kullanıyor ama `cv2.imshow` kullanmıyor — headless
OpenCV'de sorunsuz çalışır.

### 2.3 Dedektör mock'u

```python
@pytest.fixture
def fake_detections() -> list[list[Detection]]:
    """YOLO'yu tamamen atlar. 30 kare, yaklaşan tek nesne, track_id=1."""
    out = []
    for i in range(30):
        pixel_width = 200.0 - i * 4.0
        x1 = int(320 - pixel_width / 2)
        bbox = (x1, 250, int(x1 + pixel_width), 250 + int(pixel_width * 0.4))
        out.append([
            Detection(
                track_id=1,
                bbox_xyxy=bbox,
                confidence=0.92,
                pixel_width=pixel_width,
                center_px=(320.0, 280.0),
                aspect_ratio=pixel_width / (pixel_width * 0.4),
                y_normalized=bbox[3] / 480.0,
            )
        ])
    return out
```

Bu fixture NRC'nin 5 girdisinin **dördünü** doğrudan üretiyor
(`pixel_width`, `center_px` → `y_normalized`, `aspect_ratio`, `confidence`).
Beşincisi (`v_raw`) pipeline çıktısından gelir. Yani NRC eklendiğinde
(`PLANLANIYOR`) yeni fixture yazmak gerekmez — mevcut olan eğitim verisi
üreteci olarak da kullanılabilir.

---

## 3. YOLO model ağırlığı mock'u

`ultralytics` ilk `YOLO("yolov8n.pt")` çağrısında dosya yoksa **internetten
indirir**. CI'da bu:

1. Ağ bağımlılığı yaratır (kesinti = kırmızı CI)
2. Runner'a 6.5 MB yazar
3. Test sonucunu model sürümüne bağlar (ultralytics güncellenirse ağırlık değişir)

```console
$ ls -la yolov8n.pt yolov8s.pt best.pt detector.pt
-rw-r--r-- 1 kadir kadir  6549796 yolov8n.pt
-rw-r--r-- 1 kadir kadir 22573363 yolov8s.pt
-rw-r--r-- 1 kadir kadir 53174570 best.pt
-rw-r--r-- 1 kadir kadir  6241454 detector.pt
```

**Karar:** testler YOLO'yu hiç import etmez. `ObjectDetector` bir protokol
(interface) arkasına alınır:

```python
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
    def track(self, frame: np.ndarray) -> list[Detection]: ...

class YoloDetector:
    """Gerçek implementasyon. Yalnızca bu dosya ultralytics import eder."""
    def __init__(self, model_path: Path, mode: DetectionMode) -> None:
        from ultralytics import YOLO          # geç import, testte tetiklenmez
        self._model = YOLO(str(model_path))
```

Geç import (`__init__` içinde) kritik: modül seviyesinde `from ultralytics
import YOLO` yazılırsa, `detector.py`'yi import eden her test ultralytics
ister. Fonksiyon içinde olunca yalnızca gerçekten nesne kurulduğunda gerekir.

Aynı desen `torch` için de geçerli (`nrc/model.py`, `PLANLANIYOR`).

`YoloDetector`'ın kendisi test edilmez — üçüncü taraf sarmalayıcı, kendi test
paketi var. Test edilen şey `Detector` protokolünü tüketen pipeline.

---

## 4. Görselleştirme ve headless OpenCV

CI'da `opencv-python-headless` kurulur. Bu paket `cv2.imshow` **sağlamaz**.
`visualization.py` import edilirse ne olur?

```python
>>> import cv2
>>> cv2.imshow
AttributeError: module 'cv2' has no attribute 'imshow'
```

Import anında hata vermez, çağrı anında verir. Yani `visualization.py`
import edilebilir ama test edilemez.

**Karar:** `visualization.py` kapsam dışı bırakılır
(`[tool.coverage.run] omit`, `02_ci_workflow_adimlari.md` bölüm 3).
Pipeline'ın geri kalanı bu modülü import etmez — `01_dizin_yapisi/01_hedef_monorepo.md`
bölüm 5'teki bağımlılık grafiğinde `visualization.py`'ye yalnızca `cli.py`
ok çizer.

Yerelde GUI ile çalıştırmak isteyen için:

```bash
uv sync --extra dev --extra gui      # opencv-python (GUI'li) kurar
uv run python -m vehicle_speed.cli --video video1.mp4 --show
```

```toml
[project.optional-dependencies]
gui = ["opencv-python>=4.9"]
```

**Çakışma uyarısı:** `gui` ve ana bağımlılıktaki `opencv-python-headless` aynı
anda kurulamaz (aynı `cv2` ad alanını doldururlar, hangisinin kazanacağı
kurulum sırasına bağlı). `uv sync --extra gui` çalıştırıldığında headless
kaldırılmalı. Bu, `CONTRIBUTING.md`'de açıkça belgelenmeli — sessiz bir arıza
kaynağı.

---

## 5. Firmware davranışının Python tarafında simülasyonu

`firmware/src/main.cpp` mantığının Python karşılığı yazılıp test edilebilir.
Bu, donanım olmadan firmware sözleşmesini doğrulamayı sağlar.

```python
def simulate_firmware(
    steps_per_mm: int = 400,
    travel_mm: int = 500,
    duration_ms: int = 300_000,
    start_speed: float = 1000.0,
    increment: float = 500.0,
    speed_limit: float = 8000.0,
    status_interval_ms: int = 5000,
) -> Iterator[str]:
    """main.cpp'nin davranışsal modeli. JSON satırları üretir."""
```

Bu simülatör iki işe yarar:

1. **Sözleşme testi:** ürettiği satırlar `parse_line` ile ayrıştırılabilmeli.
2. **Fiziksel tutarlılık testi:** `STEPS_PER_MM=400` ve `TRAVEL_DISTANCE_MM=500`
   ile tam bir git-gel 200000 adım. 1000 step/s hızda 200 saniye sürer.
   `TEST_DURATION_MS=300000` (300 sn) içinde ancak 1.5 tur tamamlanır.

İkinci madde gerçek bir bulgu üretiyor: firmware'in `speedIncrement=500` ile
hız artışı, ancak her uç noktaya varıldığında tetikleniyor. 5 dakikada kaç uç
noktaya varılacağı hesaplanabilir:

| Tur | Hız (step/s) | 200000 adım süresi | Birikimli |
| --- | --- | --- | --- |
| 1 | 1000 | 200 sn | 200 sn |
| 2 | 1500 | 133 sn | 333 sn |

İlk tur bile 200 saniye; 300 saniyelik testte ancak 1.5 tur. Yani
`direction_change` olayı **en fazla bir kez** üretilir ve hız artışı bir kez
gerçekleşir. `maxSpeedLimit = 8000.0`'e hiç ulaşılmaz.

```python
def test_5min_test_never_reaches_speed_limit() -> None:
    """TEST_DURATION_MS=300000 ile 8000 step/s tavanına ulaşılamıyor."""
    events = [parse_line(line) for line in simulate_firmware()]
    direction_changes = [e for e in events if isinstance(e, DirectionChangeEvent)]
    assert len(direction_changes) <= 2
    assert all(e.new_max_speed < 2500.0 for e in direction_changes)
```

Bu testin sonucu, deney tasarımını doğrudan etkiliyor: 5 dakikalık test,
kademeli hız artışını ölçmek için **yetersiz**. Makalede farklı hızlarda yer
gerçeği üretmek isteniyorsa ya `TEST_DURATION_MS` artırılmalı ya da
`TRAVEL_DISTANCE_MM` azaltılmalı. Bu, donanım laboratuvarında karar verilmesi
gereken bir ölçüm tasarımı sorunu ve `research_proposal.yml` şablonuyla
issue olarak açılmalı.

---

## 6. Fixture dosyası üretimi

`tests/fixtures/telemetry_sample.jsonl` **gerçek cihazdan** alınmalı, elle
yazılmamalı. Gerekçe: elle yazılan fixture, firmware'in gerçek davranışını
değil yazanın varsayımını test eder.

```bash
# 1. firmware'i yükle ve çalıştır
cd firmware && pio run -t upload && pio device monitor --baud 115200 --raw | tee /tmp/raw.log

# 2. test bitince (complete olayı) durdur, JSON satırlarını ayıkla
grep '^{' /tmp/raw.log | head -100 > tests/fixtures/telemetry_sample.jsonl

# 3. doğrula
wc -l tests/fixtures/telemetry_sample.jsonl
python -c "
import json, pathlib
for i, line in enumerate(pathlib.Path('tests/fixtures/telemetry_sample.jsonl').read_text().splitlines()):
    json.loads(line)
print('tüm satırlar geçerli JSON')
"
```

Mevcut `serial.log` yalnızca 23 bayt:

```console
$ wc -c serial.log
23 serial.log
```

Yani kullanılabilir bir kayıt yok. Fixture üretimi **donanım gerektiren bir ön
koşul** ve `07_uygulama_yol_haritasi/01_adim_adim_uygulama.md` içinde Faz 5'e
bağımlılık olarak yazıldı. Faz 1-4 (temizlik, paket, lint, CI) bu engel olmadan
tamamlanabilir; telemetri testleri geçici olarak sentetik satırlarla
(`04_test_mimarisi/01_test_piramidi.md` bölüm 5) yazılır ve gerçek kayıt
gelince değiştirilir.

Fixture dosyası depoya girer (birkaç KB), ham `serial.log` girmez.

---

## 7. Mock katmanının sınırları

Dürüst kayıt — mock'lar şunları **test etmez**:

| Test edilmeyen | Neden | Risk |
| --- | --- | --- |
| Gerçek seri port zamanlaması | mock anında döner | `os.read` blok davranışı, tampon taşması |
| Baud rate uyumsuzluğu | mock'ta baud anlamsız | 115200 dışında ayarlanmış cihaz |
| USB kopması | mock kesinti üretmez | `SerialException` ele alınmıyor |
| YOLO tespit doğruluğu | model mock'landı | mAP, yanlış pozitif oranı |
| Kamera kare hızı dalgalanması | sentetik üreteç sabit fps | `delta_t = frame_diff / fps` varsayımı bozulur |
| Gerçek ışık/gürültü koşulları | sentetik kare | `confidence` dağılımı gerçekçi değil |
| Step motor adım kaybı | simülatör ideal | mekanik yük altında `STEPS_PER_MM` sapması |

Son madde en önemlisi: yer gerçeği iddiası, vidalı mil sisteminin adım
kaybetmediği varsayımına dayanıyor. Bu varsayım **mock ile test edilemez**;
fiziksel ölçüm gerekir (kumpas/lazer mesafe ölçer ile tur sonu konum kontrolü).
Bu bir `hardware` testi olarak `tests/hardware/test_esp32_live.py` içine
yazılmalı ve makalede raporlanmalı.
