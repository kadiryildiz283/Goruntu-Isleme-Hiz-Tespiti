# Teknik Borç ve Gerçek Kod Hataları

**Kanıt tarihi:** 2026-10-13
**Kapsam:** `kadir.py`, `1_donem_kadir.py`, `listen.py`, `firmware/src/main.cpp`

Bu rapor iki şeyi ayırır:

- **A. Kod hataları** — bugün yanlış davranış üreten veya üretebilecek somut kusurlar.
- **B. Altyapı borcu** — kod doğru ama sürdürülebilirlik/CI önündeki engeller.

Her madde dosya ve satır düzeyinde kanıtlıdır. Önceliklendirme `05_yonetisim/01_label_taksonomisi.md`
içindeki `priority/*` şemasını kullanır.

---

## A. Kod hataları

### A1. `except` bloğunda atanmamış `track_id` kullanımı — `P1`

`kadir.py` ve `1_donem_kadir.py` dosyalarının ikisinde de aynı desen var:

```python
for i in range(len(boxes)):
    try:
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        track_id = int(boxes.id[i].cpu().numpy())
        ...
    except Exception as e:
        print(f"[DEBUG] Core processing failure for ID {track_id}: {e}")
```

İki ayrı arıza modu:

1. Hata `boxes.xyxy[i]` satırında, yani `track_id` atanmadan **önce** oluşursa ve
   bu döngünün ilk turuysa (`i == 0`), `track_id` hiç bağlanmamıştır. Except
   bloğunun içindeki f-string `UnboundLocalError` fırlatır. Bu yeni hata `try`
   bloğunun dışında kaldığı için yakalanmaz ve tüm `process_frame` çağrısı çöker.
   Yani asıl hatayı gizlemek yerine programı durdurur.
2. `i > 0` ise `track_id` **önceki turun** değerini taşır. Log'da yanlış nesne
   kimliği raporlanır; hata ayıklama sırasında yanlış iz sürülür.

**Düzeltme:** `track_id` döngü başında `None` ile ilklendirilmeli ve log mesajı
bunu tolere etmeli. Ayrıca geniş `except Exception` yerine beklenen istisnalar
daraltılmalı.

```python
for i in range(len(boxes)):
    track_id = None
    try:
        ...
    except (IndexError, ValueError) as exc:
        print(f"[DEBUG] Core processing failure for ID {track_id}: {exc}")
```

### A2. Kalibrasyon sessizce başarısız oluyor — `P1`

`kadir.py`, `calibrate_camera`:

```python
def calibrate_camera(self, pixel_w, known_distance_cm, width, height, fps) -> None:
    if pixel_w <= 0:
        return
```

`pixel_w <= 0` ise fonksiyon sessizce dönüyor, `self.is_calibrated` `False`
kalıyor. Ardından `process_frame` içinde:

```python
if not self.is_calibrated:
    continue   # Skip processing until calibrated
```

Sonuç: sistem hiçbir nesne üretmiyor ve hiçbir hata mesajı basmıyor. Dışarıdan
bakıldığında "model çalışmıyor" gibi görünür, oysa kök neden ilk tespitte sıfır
genişlikli bir kutu gelmesi. Telemetri çıktısı boş olan bir makale deneyinde bu
ayırt edilemez.

**Düzeltme:** `return` yerine açık uyarı logla ve başarısız kalibrasyon
denemelerini sayan bir sayaç tut. Belirli eşik aşılırsa `ValueError` yükselt.

### A3. `listen.py` içinde proje dışına işaret eden sabit yol — `P1`

```python
with open('/home/kadir/Projeler/yapay_zeka_yuksek_lisans/serial.log', 'w') as log:
```

Bu yol **bu proje dizini değil**. Çalışma dizini
`/home/kadir/Masaüstü/Goruntu-Isleme-Hiz-Tespiti`. Yani script, başka bir projenin
dizinine yazıyor. O dizin yoksa `FileNotFoundError` ile düşüyor ve hata mesajı
`serial.log`'a yazılmaya çalışıldığı için o da patlıyor — yani iki kez başarısız
oluyor, hiç iz bırakmadan.

Aynı dosyada `/dev/ttyUSB0` da sabit kodlanmış.

**Düzeltme:** yol ve port `argparse` ile parametrik olmalı, varsayılanlar proje
köküne göre `pathlib.Path(__file__).resolve().parent` üzerinden türetilmeli.

### A4. `listen.py` dosya tanımlayıcısı sızıntısı ve kaba kapanış — `P2`

```python
fd = os.open('/dev/ttyUSB0', os.O_RDWR | os.O_NOCTTY)
attrs = termios.tcgetattr(fd)
...
except Exception as e:
    with open(...) as log:
        log.write(f"ERROR: {e}\n")
```

`os.open` ile alınan `fd` hiçbir yolda `os.close(fd)` ile kapatılmıyor. Ayrıca
`except Exception` her şeyi yutuyor; `KeyboardInterrupt` dışında seri port
okumasındaki kalıcı hatalar (kablo çıkması) sonsuza dek sessiz kalıyor.

Ek olarak: `pyserial 3.5` bu makinede **kurulu**, ama script onu kullanmayıp ham
`termios` çağrılarıyla elle port yapılandırması yapıyor. `pyserial` timeout,
`in_waiting`, port listeleme ve platform bağımsızlığı sağlar. Elle yazılmış
`termios` kodu Windows'ta hiç çalışmaz — bu da planlanan `windows-latest` CI
matrisiyle doğrudan çelişir.

**Düzeltme:** `pyserial` tabanlı, `with serial.Serial(...) as port:` bağlam
yöneticisi kullanan, `argparse` ile yapılandırılan bir `telemetry` modülüne
dönüştür. `listen.py` kök dizinde kalmamalı.

### A5. `import sys` kullanılmıyor — `P3`

Ruff çıktısı `listen.py` için `1 F401 unused-import` veriyor. `sys` import edilmiş
ama hiç kullanılmıyor. `--fix` ile otomatik temizlenir.

### A6. `1_donem_kadir.py` içinde `exit()` kullanımı — `P3`

```python
if not cap.isOpened():
    print(f"[ERROR] Stream inaccessible: {VIDEO_SOURCE}")
    exit()
```

`exit()` `site` modülünün yardımcı nesnesidir, üretim kodunda `sys.exit()`
kullanılmalıdır. `site` devre dışı bırakılmış ortamlarda (`python -S`) `exit`
tanımsız olur ve `NameError` verir.

### A7. GUI bağımlılığı CI'da çalışmaz — `P2` (altyapı ile kesişim)

Her iki script de ana blokta `cv2.imshow(...)` ve `cv2.waitKey(1)` çağırıyor.
GitHub Actions runner'larında görüntü sunucusu yok. Ayrıca CI için önerilen
`opencv-python-headless` paketi `imshow` sembolünü hiç sağlamaz.

Bu bir kod hatası değil, mimari sınır: **çalıştırma bloğu ile iş mantığı ayrılmamış**.
`if __name__ == "__main__":` altındaki görselleştirme kodu, çekirdek
algoritmadan ayrılıp ayrı bir `cli` veya `visualization` modülüne alınmalı.
Böylece CI yalnızca algoritmayı test eder, GUI'ye hiç dokunmaz.

---

## B. Altyapı borcu

### B1. Lint taban çizgisi: 141 hata — `P1`

Doğrulanmış komut:

```console
$ ruff check --isolated \
    --select E,F,W,I,B,C4,UP,SIM,N,S,ARG,PTH,RUF \
    --line-length 100 --statistics \
    kadir.py 1_donem_kadir.py listen.py
```

Dosya bazında:

| Dosya | Toplam | Otomatik düzeltilebilir |
| --- | --- | --- |
| `kadir.py` | 65 | 54 |
| `1_donem_kadir.py` | 71 | 58 |
| `listen.py` | 5 | 3 |
| **Toplam** | **141** | **115** |

Kural bazında dağılım:

| Kod | Adet | Açıklama | Oto |
| --- | --- | --- | --- |
| `W293` | 84 | boş satırda boşluk karakteri | ✓ |
| `UP006` | 22 | `typing.Tuple` yerine `tuple` (PEP 585) | ✓ |
| `E501` | 16 | satır 100 karakterden uzun | ✗ |
| `UP035` | 6 | kullanımdan kaldırılmış `typing` importu | ✗ |
| `W291` | 4 | satır sonu boşluğu | ✓ |
| `I001` | 3 | import sırası bozuk | ✓ |
| `PTH123` | 2 | `open()` yerine `pathlib.Path.open()` | ✗ |
| `N806` | 2 | fonksiyon içinde büyük harfli değişken | ✗ |
| `UP045` | 1 | `Optional[X]` yerine `X \| None` | ✓ |
| `F401` | 1 | kullanılmayan import | ✓ |

Yorum: 115 hata `ruff check --fix` ile saniyeler içinde temizlenir. Kalan 26
hatanın 16'sı `E501` (uzun satır) — bunlar `ruff format` ile büyük ölçüde
sarılır. Elle müdahale gerektiren gerçek iş yükü ~10 satır: `UP035`, `PTH123`,
`N806`.

`N806`'nın 2 adedi `1_donem_kadir.py` içindeki `LEFT_HEADLIGHT_IDX` ve
`RIGHT_HEADLIGHT_IDX` sabitleridir. Bunlar modül seviyesine taşınmalı — o zaman
hem kural ihlali ortadan kalkar hem de `per-file-ignores` istisnasına gerek
kalmaz.

### B2. Paket yapısı yok — `P1`

Kod kök dizinde iki bağımsız script olarak duruyor. `__init__.py` yok, import
edilebilir bir paket yok. Sonuçları:

- Test yazılamaz: `from vehicle_speed.geometry import estimate_depth` gibi bir
  import hedefi yok.
- İki script arasında ortak mantık kopyalanmış: `estimate_depth`,
  `calculate_3d_coords`, `compute_speed`, `_cleanup_history` fonksiyonları
  `kadir.py` ve `1_donem_kadir.py` içinde neredeyse aynı biçimde iki kez yazılmış.
- Bağımlılık bildirimi yok: `torch`, `ultralytics`, `scipy` import ediliyor ama
  hiçbir manifest'te declared değil.

Kopyalanan dört fonksiyon, `src/vehicle_speed/geometry.py` ve
`src/vehicle_speed/tracking.py` altına tek kopya olarak taşınmalı.

### B3. Bağımlılık manifesti yok — `P1`

`requirements.txt`, `pyproject.toml`, `environment.yml` — hiçbiri yok. Bir
katkıcı depoyu klonladığında hangi paketleri kuracağını bilemez. Bugün bu
makinede `torch`, `ultralytics`, `scipy` kurulu olmadığı için `kadir.py`
çalıştırılamaz durumda. Bu, "bende çalışıyor" sınıfı bir borç.

### B4. Yapılandırma değerleri koda gömülü — `P2`

| Değer | Konum | Sorun |
| --- | --- | --- |
| `hfov=85.0` | `1_donem_kadir.py` `__main__` | Kodda "placeholder" diye yazılı |
| `initial_dist: 183.0 / 178.0` | `kadir.py` `VIDEO_SOURCES` | Video başına farklı, neden? belgesiz |
| `BALL_WIDTH_CM = 18.5` | `kadir.py` | Ölçü kaynağı belirtilmemiş |
| `PLATE_WIDTH_M = 0.520` | `1_donem_kadir.py` | TR plaka standardı, referans yok |
| `video1.mp4`, `traffic_test.mp4` | `__main__` | Göreli yol, çalışma dizinine bağımlı |
| `/dev/ttyUSB0` | `listen.py`, `platformio.ini` | Donanım yolu sabit |
| `detector.pt`, `best.pt`, `yolov8n.pt` | `__main__` | Model yolları sabit |

Bunlar bir yapılandırma katmanına (TOML/YAML + `pydantic-settings` veya basit
`dataclass`) taşınmalı. Makale tekrarlanabilirliği açısından da şart: hakem
"85.0 nereden geldi?" diye soracak.

### B5. Tek kaynak ilkesi ihlali (firmware) — `P2`

`esp32.c` ve `firmware/src/main.cpp` aynı kodun iki kopyası (fark: 10 satır,
tamamı yorum biçimi). `platformio.ini` yalnızca `src/main.cpp`'yi derler, yani
`esp32.c` ölü dosya. Silinmeli veya `docs/` altına referans olarak taşınmalı.

### B6. Derleme çıktıları çalışma ağacında — `P2`

`firmware/.pio/` dizini 23 MB ve içinde `firmware.map` (9.1 MB),
`firmware.elf` (6.1 MB), `libFrameworkArduino.a` (3.3 MB) gibi tamamen
üretilmiş dosyalar var. `.gitignore` olmadığı için bunlar ilk `git add .`
komutunda depoya girer. PlatformIO `.pio/` dizini her zaman yok sayılmalı.

### B7. Loglama `print` ile yapılıyor — `P3`

Her iki script de `print(f"[INFO] ...")`, `print(f"[ERROR] ...")`,
`print(f"[DEBUG] ...")` kullanıyor. Seviye filtreleme yok, dosyaya yazma yok,
zaman damgası yok. `logging` modülüne geçilmeli; böylece CI'da log seviyesi
`WARNING`'e çekilip gürültü azaltılabilir, deneylerde ise `DEBUG` ile tam iz
kaydı alınabilir.

### B8. Tip ipuçları eski sözdizimi — `P3`

`from typing import Tuple, List, Dict, Any, Optional` → `UP006` (22 adet) ve
`UP035` (6 adet). Python 3.10+ hedefleniyorsa `tuple[float, float]`,
`list[dict[str, Any]]`, `int | None` biçimi kullanılmalı. Otomatik düzeltilebilir.

---

## C. Borç → Issue eşlemesi

Bu maddeler uygulama aşamasında GitHub issue'ya dönüştürülmeli. Önerilen
başlıklar ve etiketler:

| Madde | Issue başlığı | Etiketler |
| --- | --- | --- |
| A1 | `[P1][BUG] except bloğunda atanmamış track_id kullanımı` | `type/bug`, `component/vision`, `priority/p1-high` |
| A2 | `[P1][BUG] Kalibrasyon başarısızlığı sessizce yutuluyor` | `type/bug`, `component/calibration`, `priority/p1-high` |
| A3 | `[P1][BUG] listen.py proje dışına sabit yol ile yazıyor` | `type/bug`, `component/telemetry`, `priority/p1-high` |
| A4 | `[P2][REFACTOR] termios yerine pyserial tabanlı telemetri okuyucu` | `type/refactor`, `component/telemetry`, `priority/p2-medium` |
| A6 | `[P3][BUG] exit() yerine sys.exit() kullanılmalı` | `type/bug`, `priority/p3-low` |
| A7 | `[P2][REFACTOR] Görselleştirme kodunu çekirdek algoritmadan ayır` | `type/refactor`, `component/vision`, `component/ci-cd` |
| B1 | `[P1][CHORE] 141 lint hatasını temizle ve ruff kapısını etkinleştir` | `type/chore`, `component/ci-cd`, `priority/p1-high` |
| B2 | `[P1][REFACTOR] Script'leri src/vehicle_speed paketine taşı` | `type/refactor`, `component/vision`, `priority/p1-high` |
| B3 | `[P1][FEATURE] pyproject.toml ile bağımlılık manifesti oluştur` | `type/feature`, `component/ci-cd`, `priority/p1-high` |
| B4 | `[P2][FEATURE] Yapılandırma değerlerini koddan ayır` | `type/feature`, `component/calibration` |
| B5 | `[P2][CHORE] esp32.c kopyasını kaldır, tek kaynak main.cpp olsun` | `type/chore`, `component/firmware` |
| B6 | `[P2][CHORE] .gitignore ekle, .pio ve venv'i depo dışına al` | `type/chore`, `component/ci-cd` |
| B7 | `[P3][REFACTOR] print yerine logging modülü` | `type/refactor`, `priority/p3-low` |

Bağımlılık zinciri: `B6` → `B3` → `B2` → `B1` → `A7`. Yani önce depo
temizlenir, sonra manifest yazılır, sonra paket yapısı kurulur, sonra lint
kapısı açılır. `A1`–`A4` kod hataları bu zincirden bağımsız olarak hemen
açılabilir, ama `B2` sonrasında taşınmış dosyalar üzerinde düzeltilmeleri daha
az çakışma üretir.

Issue gövde biçimi `05_yonetisim/02_issue_pr_sablonlari.md` içinde tanımlı.
