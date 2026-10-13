# Firmware Tarafı: C/C++ Statik Analiz ve Derleme Kapısı

**Kapsam:** `firmware/src/main.cpp`, `firmware/platformio.ini`, kökteki `esp32.c`
**Araç durumu (doğrulandı):** `clang-format 23.1.1` `VAR`; `platformio` ve
`cppcheck` `YOK`.

Python tarafındaki Ruff/Mypy kapılarının firmware'e uygulanamaz. ESP32 kodu ayrı
bir araç zinciri ve ayrı bir CI job'ı gerektirir. Bu rapor o job'ı tanımlar.

---

## 1. Mevcut firmware durumunun kanıtı

`firmware/platformio.ini`:

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

Derleme çıktısı çalışma ağacında duruyor:

```console
$ find firmware/.pio -type f -size +1M -printf '%10s  %p\n'
  9057508  firmware/.pio/build/esp32doit-devkit-v1/firmware.map
  6144936  firmware/.pio/build/esp32doit-devkit-v1/firmware.elf
  3337028  firmware/.pio/build/esp32doit-devkit-v1/libFrameworkArduino.a
```

Yani firmware **en az bir kez başarıyla derlenmiş**. Bu önemli: derleme kapısı
CI'a eklendiğinde sıfırdan başlamıyor, bilinen çalışan bir taban çizgisi var.

### Kopya dosya sorunu

```console
$ diff esp32.c firmware/src/main.cpp | grep -c '^[<>]'
10
```

Farkın tamamı yorum biçimi (` * *` → ` *`) ve `main.cpp`'deki ek
`#include <Arduino.h>`. PlatformIO yalnızca `firmware/src/` altını derler,
yani `esp32.c` **derlenmeyen ölü kopya**.

**Karar:** `esp32.c` silinir, `firmware/src/main.cpp` tek kaynak olur. Silme
öncesi içeriği `docs/` altına referans olarak kopyalanmamalı — git geçmişi
zaten bu işi görür (depo onarıldıktan sonra).

---

## 2. `upload_port` ve `monitor_port` sorunu

```ini
upload_port = /dev/ttyUSB0
monitor_port = /dev/ttyUSB0
```

Bu iki satır Linux'a özgü ve makineye özgü. Sonuçları:

1. Windows'ta port adı `COM3` gibidir, yapılandırma çalışmaz.
2. CI'da hiç port yok, derleme adımını etkilemez ama `pio run -t upload`
   denenirse başarısız olur.
3. `/dev/ttyUSB0` yerine `/dev/ttyACM0` takan bir katkıcı yapılandırmayı
   değiştirmek zorunda kalır — bu da gereksiz commit üretir.

**Çözüm:** portları `platformio.ini`'den çıkarıp `platformio.local.ini`
(`.gitignore`'da) veya ortam değişkenine taşı:

```ini
[env:esp32doit-devkit-v1]
platform = espressif32
board = esp32doit-devkit-v1
framework = arduino
monitor_speed = 115200
lib_deps =
    waspinator/AccelStepper @ ^1.64
; upload_port ve monitor_port KALDIRILDI
; Yerel ayar için platformio.local.ini kullanın (depoya girmez):
;   [env:esp32doit-devkit-v1]
;   upload_port = /dev/ttyUSB0
```

PlatformIO `platformio.local.ini` dosyasını otomatik olarak `platformio.ini`
üzerine bindirir ve bu dosya `.gitignore`'a eklenmelidir. Böylece port tercihi
katkıcıya kalır, depo makine bağımsız olur.

Aynı sorun `listen.py` içinde de var (`/dev/ttyUSB0` sabit) —
`00_durum_analizi/02_teknik_borc_ve_hatalar.md` A3/A4.

---

## 3. Derleme kapısı: `.github/workflows/firmware.yml`

```yaml
name: Firmware

on:
  push:
    branches: [main, develop]
    paths:
      - "firmware/**"
      - ".github/workflows/firmware.yml"
  pull_request:
    paths:
      - "firmware/**"
      - ".github/workflows/firmware.yml"

concurrency:
  group: firmware-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    name: PlatformIO build + static analysis
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Cache PlatformIO
        uses: actions/cache@v4
        with:
          path: ~/.platformio
          key: pio-${{ runner.os }}-${{ hashFiles('firmware/platformio.ini') }}

      - name: Install PlatformIO
        run: |
          python -m pip install --upgrade pip
          pip install platformio==6.1.16

      - name: Install cppcheck
        run: sudo apt-get update && sudo apt-get install -y cppcheck

      - name: Build firmware
        working-directory: firmware
        run: pio run

      - name: Report binary size
        working-directory: firmware
        run: |
          pio run -t size | tee ../build-size.txt

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: firmware-build
          path: |
            firmware/.pio/build/esp32doit-devkit-v1/firmware.bin
            firmware/.pio/build/esp32doit-devkit-v1/firmware.elf
            build-size.txt
          retention-days: 14

      - name: cppcheck static analysis
        working-directory: firmware
        run: |
          cppcheck \
            --enable=warning,style,performance,portability \
            --suppress=missingIncludeSystem \
            --inline-suppr \
            --error-exitcode=1 \
            --std=c++17 \
            -I ~/.platformio/packages/framework-arduinoespressif32/cores/esp32 \
            src/ include/

      - name: clang-format check
        run: |
          pip install clang-format==19.1.3
          clang-format --dry-run --Werror \
            firmware/src/*.cpp firmware/include/*.h 2>&1 | tee fmt.log
          [ ! -s fmt.log ] || (echo "Biçimlendirme farkı var" && exit 1)
```

### `paths:` filtresi neden önemli

Firmware job'ı her push'ta çalışırsa, yalnızca `docs/` değiştiğinde bile
PlatformIO kurulumu (~2 dakika) boşa gider. `paths` filtresi bunu engeller.
Aynı filtre `ci.yml` içinde de `docs/**` değişikliklerinde Python testlerini
atlamak için kullanılabilir — ama orada dikkatli olunmalı: `pyproject.toml`
değişikliği `paths` listesine eklenmezse bağımlılık güncellemesi test edilmez.

### `--error-exitcode=1`

cppcheck varsayılan olarak bulgu olsa bile `0` döner. Bu bayrak olmadan
statik analiz adımı **hiçbir zaman başarısız olmaz** — yani kapı değil,
sadece bilgi üretir. Kapı olması için bu bayrak şart.

---

## 4. cppcheck yapılandırması

Komut satırı yerine dosya ile:

`firmware/.cppcheck`:

```text
--enable=warning,style,performance,portability
--suppress=missingIncludeSystem
--suppress=unusedFunction:src/main.cpp
--inline-suppr
--std=c++17
--error-exitcode=1
```

`unusedFunction` bastırması gerekli: Arduino `setup()` ve `loop()` fonksiyonları
framework tarafından çağrılır, kaynak kodda çağıran görünmez. Bu bastırma
olmazsa cppcheck her ikisini de "kullanılmayan fonksiyon" diye raporlar.

### `main.cpp` üzerinde beklenen bulgular

Kod okunarak öngörülen gerçek sorunlar:

**4.1 `Serial.print` ile elle JSON kurma**

```cpp
Serial.print("{\"event\": \"startup\", \"message\": \"5 dakikalik test basladi\", \"max_speed\": ");
Serial.print(currentMaxSpeed);
Serial.println("}");
```

Dört ayrı yerde tekrarlanan elle JSON kurma. Riskler:

- `currentMaxSpeed` `float` ve `Serial.print(float)` varsayılan 2 ondalık basar.
  Python tarafı `max_speed: float` bekliyor, `1000.00` biçimi geçerli JSON ama
  `8000.00` ile `8000.0` farklı string. Ayrıştırıcı `json.loads` kullandığı
  için sorun değil, ama log dosyası karşılaştırmalarında gürültü üretir.
- Kaçış karakterleri elle yönetiliyor. Mesaj metninde `"` geçerse JSON bozulur.
  Şu anki mesajlarda yok, ama Türkçe karakter ve tırnak eklenirse sessiz bozulma.
- Dört olay tipi için dört ayrı `Serial.print` bloğu — şema değişikliği dört
  yerde yapılmalı, biri unutulursa Python ayrıştırıcısı patlar.

**Öneri:** `ArduinoJson` kütüphanesi:

```ini
lib_deps =
    waspinator/AccelStepper @ ^1.64
    bblanchon/ArduinoJson @ ^7.2.0
```

```cpp
#include <ArduinoJson.h>

void emitEvent(const char* event, JsonDocument& extra = {}) {
  JsonDocument doc;
  doc["event"] = event;
  for (auto kv : extra.as<JsonObject>()) doc[kv.key()] = kv.value();
  serializeJson(doc, Serial);
  Serial.println();
}
```

Bu değişiklik `01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md`
bölüm 2.5'teki `TelemetryEvent` sözleşmesini garanti altına alır. Şu an
sözleşme yalnızca iki tarafın da aynı biçimde yazıldığına dair bir umut.

**4.2 `loop()` içinde bloklama**

```cpp
Serial.print(...);  // 115200 baud'da ~100 bayt ≈ 8.7 ms
```

`status` logu 5000 ms'de bir, `direction_change` her uç noktada. 8000 step/s
hızda bir uç noktaya varış 25 saniye sürer, yani `direction_change` nadir.
Ama JSON çıktısı ~120 bayt ve 115200 baud'da ~10.4 ms blok yapar. Bu sürede
`stepper.run()` çağrılmaz → adım kaybı.

`AccelStepper`'ın adım zamanlaması `run()` çağrılarına bağlı. 10 ms'lik bir
gecikme, 8000 step/s hızda ~80 adım kaybı demek. `STEPS_PER_MM = 400` olduğu
için bu **0.2 mm konum hatası**. Yer gerçeği iddiası mikron düzeyinde olduğu
için bu, makalenin doğruluk iddiasını doğrudan zayıflatır.

**Öneri:** seri yazımı tamponlamak ya da log aralığını artırmak. Ölçülmesi
gereken bir ödünleşim — `04_test_mimarisi/02_mock_kamera_seri_port.md`
bölüm 5'te test senaryosu olarak tanımlandı.

**4.3 `return` ile `loop()` sonlandırma**

```cpp
if (currentTime >= TEST_DURATION_MS) {
    if (stepper.isRunning()) { ... }
    return;
}
```

`return` `loop()`'u bitirir ama Arduino framework `loop()`'u tekrar çağırır.
Yorum satırında "Programı sonlandır (loop döner ama içeriğe girmez)" yazılı —
bu **yanlış**. `currentTime >= TEST_DURATION_MS` koşulu kalıcı olarak doğru
olduğu için içeriğe girer, ama `stepper.isRunning()` artık `false` olduğundan
iç bloğu atlar. Yani davranış doğru, gerekçe yanlış. `complete` olayı yalnızca
bir kez basılıyor, o kısım doğru çalışıyor.

Bu tür bir yanlış yorum, cppcheck tarafından yakalanmaz. Kod incelemesi
maddesidir ve `05_yonetisim/04_branch_koruma_inceleme.md` içindeki inceleme
kontrol listesine eklendi.

---

## 5. clang-format

Sistemde `clang-format 23.1.1` var, ama pre-commit `v19.1.3` çekiyor
(`03_pre_commit.md` bölüm 3). Sürüm ayrışmasını önlemek için tek sürüm
seçilmeli ve her yerde o kullanılmalı.

`firmware/.clang-format`:

```text
BasedOnStyle: Google
IndentWidth: 2
ColumnLimit: 100
PointerAlignment: Left
AllowShortFunctionsOnASingleLine: Inline
SortIncludes: CaseSensitive
IncludeBlocks: Regroup
```

Uygulama:

```bash
clang-format -i firmware/src/main.cpp
git diff --stat firmware/
```

İlk çalıştırmada mevcut kodun büyük kısmı yeniden biçimlendirilecek. Bu,
`02_kod_kalitesi/01_ruff_linter_formatter.md` bölüm 3'teki Python
biçimlendirme adımı gibi **ayrı bir commit** olmalı:

```text
style(firmware): apply clang-format to main.cpp

No functional change. Removes the formatting drift between
firmware/src/main.cpp and the deleted esp32.c copy.
```

---

## 6. Yerel geliştirme

PlatformIO kurulu değil. Kurulum:

```bash
uv tool install platformio
# veya proje sanal ortamına
uv pip install platformio==6.1.16
```

Temel komutlar:

```bash
cd firmware
pio run                    # derle
pio run -t size            # flash/RAM kullanımı
pio run -t upload          # cihaza yükle (port gerekli)
pio device monitor         # 115200 baud seri izleme
pio device list            # bağlı portları listele
pio check                  # yerleşik statik analiz
```

`pio run -t size` çıktısı, flash doluluğunu izlemek için release workflow'una
eklendi. ESP32 flash'ı 4 MB; mevcut firmware'in ne kadar yer kapladığı
`firmware/.pio/build/.../firmware.map` (9.1 MB, sembol bilgisi dahil)
dosyasından okunabilir ama CI çıktısı daha pratik.

---

## 7. Firmware ve Python arasında şema sözleşmesi

En kırılgan nokta burası. İki taraf ayrı dillerde, ayrı depo bölümlerinde,
ayrı CI job'larında. Şema değişirse hangisinin önce güncelleneceği belirsiz.

**Öneri:** tek bir şema dosyası, iki tarafta da tüketilir.

`firmware/include/telemetry_protocol.h`:

```c
#pragma once
#define TELEMETRY_SCHEMA_VERSION 1
#define EVT_STARTUP          "startup"
#define EVT_DIRECTION_CHANGE "direction_change"
#define EVT_STATUS           "status"
#define EVT_COMPLETE         "complete"
#define STEPS_PER_MM         400
#define TRAVEL_DISTANCE_MM   500
#define TEST_DURATION_MS     300000UL
```

`src/vehicle_speed/telemetry/protocol.py`:

```python
from typing import Final

SCHEMA_VERSION: Final[int] = 1
EVENT_STARTUP: Final[str] = "startup"
EVENT_DIRECTION_CHANGE: Final[str] = "direction_change"
EVENT_STATUS: Final[str] = "status"
EVENT_COMPLETE: Final[str] = "complete"
STEPS_PER_MM: Final[int] = 400
TRAVEL_DISTANCE_MM: Final[int] = 500
```

İki dosyadaki sabitlerin eşleştiğini denetleyen bir test:

```python
def test_schema_constants_match_firmware() -> None:
    header = (REPO_ROOT / "firmware" / "include" / "telemetry_protocol.h").read_text()
    firmware_values = {
        name: int(re.search(rf"#define\s+{name}\s+(\d+)", header).group(1))
        for name in ("STEPS_PER_MM", "TRAVEL_DISTANCE_MM", "TELEMETRY_SCHEMA_VERSION")
    }
    assert firmware_values["STEPS_PER_MM"] == protocol.STEPS_PER_MM
    assert firmware_values["TRAVEL_DISTANCE_MM"] == protocol.TRAVEL_DISTANCE_MM
    assert firmware_values["TELEMETRY_SCHEMA_VERSION"] == protocol.SCHEMA_VERSION
```

Bu test Python CI job'ında çalışır, firmware derlemesi gerektirmez — yalnızca
başlık dosyasını okur. Böylece iki job arasında sözleşme kopması imkânsız hale
gelir. `04_test_mimarisi/01_test_piramidi.md` içinde bu test
`test_telemetry_protocol.py` altına yerleştirildi.

---

## 8. Doğrulama

```bash
# kopya dosya iddiasını yeniden test et
diff esp32.c firmware/src/main.cpp | grep -c '^[<>]'      # beklenen: 10

# derleme çıktılarının varlığını doğrula
ls -la firmware/.pio/build/esp32doit-devkit-v1/firmware.bin

# yerel derleme (platformio kurulduktan sonra)
cd firmware && pio run && pio run -t size

# biçimlendirme farkı var mı
clang-format --dry-run firmware/src/main.cpp | head
```

Bu raporun "derleme en az bir kez başarılı olmuş" iddiası
`firmware/.pio/build/` altındaki `firmware.elf` ve `firmware.map` dosyalarının
varlığına dayanır. Dosyalar diskte doğrulandı. Ancak **bu makinede yeniden
derleme yapılmadı**, çünkü `platformio` kurulu değil. Derlemenin bugün de
başarılı olacağı doğrulanmamış bir varsayımdır; `pio run` ilk kez
çalıştırıldığında teyit edilmelidir.
