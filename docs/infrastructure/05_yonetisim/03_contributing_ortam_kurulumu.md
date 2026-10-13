# CONTRIBUTING ve Geliştirme Ortamı Kurulumu

Bu rapor `CONTRIBUTING.md` dosyasının içeriğini ve yerel geliştirme ortamının
kurulum adımlarını tanımlar.

**Doğrulanmış yerel durum:** `uv 0.13.0` kurulu, `ruff 0.16.7` kurulu,
`gh 2.102.0` kurulu. `mypy`, `pytest`, `pre-commit`, `platformio` **yok**.
Sistem Python'u **3.14.7**, `torch`/`ultralytics`/`scipy` **kurulu değil**.

---

## 1. `CONTRIBUTING.md`

````markdown
# Katkı Rehberi

Bu depo, görüntü işleme ile araç/nesne hızı tahmini üzerine bir yüksek lisans
araştırmasının kod tabanıdır. Katkı biçimi endüstriyel bir açık kaynak projeden
farklı: algoritmik değişiklikler makalede raporlanan ölçütleri doğrudan
etkiler, bu yüzden her değişikliğin ölçüm kanıtı gerekir.

Katkıdan önce [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) ve
[SECURITY.md](SECURITY.md) dosyalarını okuyun.

## İçindekiler

- [Ortam kurulumu](#ortam-kurulumu)
- [Günlük iş akışı](#günlük-iş-akışı)
- [Kod kuralları](#kod-kuralları)
- [Test kuralları](#test-kuralları)
- [Commit biçimi](#commit-biçimi)
- [Branch ve PR süreci](#branch-ve-pr-süreci)
- [Firmware tarafı](#firmware-tarafı)
- [Sık karşılaşılan sorunlar](#sık-karşılaşılan-sorunlar)

## Ortam kurulumu

### Gereksinimler

| Araç | Sürüm | Kurulum |
| --- | --- | --- |
| `uv` | ≥ 0.13 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `git` | ≥ 2.40 | sistem paket yöneticisi |
| `gh` | ≥ 2.0 (opsiyonel) | `sudo pacman -S github-cli` |
| `clang-format` | ≥ 19 (yalnızca firmware) | pre-commit otomatik kurar |

Python'u ayrıca kurmanıza gerek yok; `uv` proje için doğru sürümü indirir.

### Kurulum

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
./scripts/bootstrap.sh
```

`bootstrap.sh` şunları yapar:

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. uv kurulu mu
command -v uv >/dev/null || {
    echo "uv bulunamadı. Kurulum: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}

# 2. Python 3.12 (sistem Python'u 3.14 olsa bile)
uv python install 3.12

# 3. sanal ortam + bağımlılıklar
uv sync --extra dev

# 4. yerel kalite kancaları
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push

# 5. doğrulama
uv run python -c "import vehicle_speed; print('paket', vehicle_speed.__version__)"
uv run ruff check .
uv run pytest tests/unit -q

echo "Kurulum tamam."
```

### İsteğe bağlı gruplar

```bash
uv sync --extra dev          # lint + test araçları (varsayılan)
uv sync --extra dev --extra ml   # torch + ultralytics (dedektör testleri için)
uv sync --extra dev --extra gui  # cv2.imshow ile görselleştirme
```

**`ml` ve `gui` aynı anda kurulmamalı.** `gui` grubu `opencv-python` (GUI'li)
çekiyor, ana bağımlılık ise `opencv-python-headless`. İkisi aynı `cv2` ad
alanını doldurur; hangisinin etkin olacağı kurulum sırasına bağlıdır ve
`cv2.imshow` varlığı öngörülemez hale gelir. Geçiş yaparken:

```bash
uv sync --extra dev --extra ml --no-extra gui   # headless'a dön
```

### Model ağırlıkları

Depo, `.pt` dosyalarını içermez (boyut sınırı). İndirmek için:

```bash
mkdir -p models
gh release download v0.1.0 --pattern "*.pt" --dir models/
```

`yolov8n.pt` ve `yolov8s.pt` ultralytics tarafından ilk kullanımda otomatik
indirilir; elle kurmaya gerek yok.

## Günlük iş akışı

```bash
git checkout -b fix/unbound-track-id      # branch aç
# ... kod değişikliği ...
uv run pytest tests/unit -q               # hızlı geri bildirim
git add -p                                # parça parça incele
git commit                                # pre-commit + commitizen çalışır
git push -u origin fix/unbound-track-id
gh pr create --fill                       # PR şablonu otomatik gelir
```

Yararlı komutlar:

```bash
uv run ruff check --fix .        # otomatik düzeltilebilir lint hataları
uv run ruff format .             # biçimlendir
uv run mypy                      # tip denetimi
uv run pytest -q                 # tüm testler
uv run pytest tests/unit -q --lf # yalnızca son başarısız olanlar
uv run pytest --cov=src --cov-report=term-missing
uv run pre-commit run --all-files
```

## Kod kuralları

Tam yapılandırma `pyproject.toml` içinde. Özet:

- Satır uzunluğu 100 karakter, 4 boşluk girinti, çift tırnak.
- Tüm fonksiyon imzaları tam tip notasyonlu (`mypy --strict`).
- Birimler değişken adının son ekinde: `z_m`, `v_ms`, `f_px`, `w_px`, `hfov_rad`.
  İç hesaplar **metre ve m/s**; cm ve km/h yalnızca CLI çıktısı ve
  görselleştirmede.
- Fizik formülleri docstring'de gösterilir: `Z = (f * W) / P`.
- `cv2.imshow` yalnızca `src/vehicle_speed/visualization.py` içinde.
- `torch` ve `ultralytics` yalnızca `detector.py` ve `nrc/**` içinde, ve
  **fonksiyon gövdesinde** import edilir (modül seviyesinde değil).
- Yapılandırma değerleri koda gömülmez; `config.py` üzerinden gelir.

İlk iki kuralın gerekçesi: `docs/infrastructure/01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md`.
Son üç kuralın gerekçesi: testlerin `torch` kurulmadan çalışabilmesi.

## Test kuralları

- Yeni davranış = yeni test. Test yoksa PR merge edilmez.
- Test adı biçimi: `test_<neyi>_<koşul>_<beklenen>`.
- Birim testler (`tests/unit`) donanım, video dosyası, internet veya `torch`
  gerektirmez. Gerektiriyorsa yanlış katmanda.
- Fiziksel donanım gerektiren testler `tests/hardware/` altına ve
  `@pytest.mark.hardware` ile işaretlenir. CI'da çalışmazlar.
- Fixture verisi elle yazılmaz; gerçek cihaz çıktısından üretilir
  (`docs/infrastructure/04_test_mimarisi/02_mock_kamera_seri_port.md` bölüm 6).
- Kapsam eşiği %85. Düşüyorsa PR açıklamasında gerekçe yazılır.

## Commit biçimi

[Conventional Commits](https://www.conventionalcommits.org/):

```text
<tip>(<kapsam>): <kısa özet>

<gövde: ne ve neden>

<altbilgi: Closes #12 / BREAKING CHANGE>
```

İzin verilen tipler: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
`test`, `build`, `ci`, `chore`, `revert`.

Kapsam adları `component/*` etiketleriyle aynı:
`vision`, `calibration`, `tracking`, `telemetry`, `firmware`, `nrc`, `ci-cd`,
`packaging`, `docs`, `data`.

Örnekler:

```text
fix(vision): guard against unbound track_id in detection loop

The except handler referenced track_id before assignment, so an error on
the first box raised UnboundLocalError and masked the original failure.
On later iterations it logged the previous box's id.

Initialize track_id to None before the try block and narrow the caught
exceptions to IndexError and ValueError.

Closes #12
```

```text
style(firmware): apply clang-format to main.cpp

No functional change. Removes the formatting drift between
firmware/src/main.cpp and the deleted esp32.c copy.
```

`commitizen` kancası biçimi zorunlu kılar; uymayan commit reddedilir.
`git commit --no-verify` **kullanmayın** — kanca engel oluyorsa ya kod ya
yapılandırma düzeltilir.

## Branch ve PR süreci

Branch adlandırma:

```text
feat/<kısa-açıklama>       yeni işlev
fix/<kısa-açıklama>        hata düzeltme
refactor/<kısa-açıklama>   yeniden yapılandırma
exp/<kısa-açıklama>        araştırma/deney (merge edilmeyebilir)
docs/<kısa-açıklama>       doküman
ci/<kısa-açıklama>         workflow
```

Akış:

1. `main`'den branch açın.
2. Küçük commit'ler; her commit tek bir mantıksal değişiklik.
3. Push edip `gh pr create --fill` ile PR açın.
4. PR şablonundaki **Doğrulama** bölümüne komut çıktılarını yapıştırın.
5. `vision`, `calibration`, `tracking`, `nrc`, `telemetry` bileşenlerine
   dokunuyorsanız **Akademik etki** tablosunu doldurun.
6. En az 1 onay + tüm zorunlu kontroller yeşil olduktan sonra squash merge.

`main`'e doğrudan push kapalıdır (branch protection).

## Firmware tarafı

```bash
cd firmware
pio run                    # derle
pio run -t size            # flash/RAM kullanımı
pio device list            # bağlı portlar
pio device monitor         # 115200 baud izleme
```

Port ayarınızı `firmware/platformio.local.ini` içine yazın (depoya girmez):

```ini
[env:esp32doit-devkit-v1]
upload_port = /dev/ttyUSB0
monitor_port = /dev/ttyUSB0
```

`platformio.ini`'yi port için düzenlemeyin — o dosya herkes için ortak.

Telemetri JSON şeması iki yerde sabit:
`firmware/include/telemetry_protocol.h` ve
`src/vehicle_speed/telemetry/protocol.py`. Birini değiştirirseniz diğerini de
değiştirin; `tests/integration/test_firmware_contract.py` eşleşmeyi denetler.

## Sık karşılaşılan sorunlar

### `import cv2` → `ImportError: libGL.so.1`

`opencv-python` (GUI'li) kurulu. Headless'a dönün:

```bash
uv sync --extra dev --no-extra gui
```

### `ModuleNotFoundError: termios` (Windows)

`termios` POSIX'e özgüdür. `pyserial` tabanlı `TelemetryReader` kullanın;
ham `termios` kodu kaldırıldı.

### `ModuleNotFoundError: ultralytics`

`ml` grubu kurulu değil:

```bash
uv sync --extra dev --extra ml
```

Birim testler `torch` gerektirmemeli. `tests/unit` altında bu hatayı
alıyorsanız, test yanlış katmanda — `tests/e2e`'ye taşıyın.

### mypy: `Cannot find implementation or library stub for module named "cv2"`

`[[tool.mypy.overrides]]` içinde `cv2.*` için `ignore_missing_imports = true`
olmalı. Hata yine çıkıyorsa `mypy --config-file=pyproject.toml` ile açıkça
belirtin.

### pre-commit: "fatal: not a git repository"

`pre-commit install` çalışan bir `.git` gerektirir. Depoyu klonladıysanız bu
hata çıkmaz; çıktıysa `.git` dizini bozuktur.

### Test asılı kalıyor, bitmiyor

Donanım testi yanlışlıkla toplanmış olabilir:

```bash
uv run pytest --collect-only -q | grep hardware
```

Çıktı boş olmalı. Boş değilse `pyproject.toml`'daki `testpaths` ayarını
kontrol edin.

### `uv sync` çok yavaş / torch 2.5 GB indiriyor

`[tool.uv.sources]` içindeki CPU index yönlendirmesi eksik veya `uv.lock`
commit edilmemiş. Kontrol:

```bash
grep -A3 "pytorch-cpu" pyproject.toml
ls -la uv.lock
uv run python -c "import torch; print(torch.__version__)"   # +cpu eki görünmeli
```
````

---

## 2. `SECURITY.md`

```markdown
# Güvenlik Politikası

## Açık bildirimi

**Güvenlik açıklarını public issue olarak AÇMAYIN.**

Bunun yerine:

1. GitHub Private Vulnerability Reporting:
   `https://github.com/OWNER/REPO/security/advisories/new`
2. Veya e-posta: `<güvenli iletişim adresi>`

48 saat içinde yanıt, 14 gün içinde durum değerlendirmesi taahhüt edilir.

## Kapsam

Bu bir araştırma kod tabanıdır; üretim servisi çalıştırmaz. Yine de şu
sınıflardaki sorunlar güvenlik kapsamında değerlendirilir:

| Sınıf | Örnek | Neden önemli |
| --- | --- | --- |
| Sabit kodlanmış kimlik bilgileri | RTSP URL içinde kullanıcı adı/şifre | depoya commit edilirse kalıcı olarak sızar |
| Yol gezinme (path traversal) | CLI'dan gelen dosya yolunun denetimsiz açılması | `--log-path /etc/passwd` |
| Güvensiz serileştirme | `torch.load` `weights_only=False` ile | `.pt` dosyası kod çalıştırabilir |
| Komut enjeksiyonu | `scripts/` içinde `shell=True` | |
| Denetimsiz ağ erişimi | ultralytics'in otomatik model indirmesi | tedarik zinciri |

### `torch.load` uyarısı

Eğitilmiş NRC checkpoint'leri (`PLANLANIYOR`) yüklendiğinde:

```python
# YANLIŞ — pickle üzerinden任意 kod çalıştırılabilir
state = torch.load(path)

# DOĞRU
state = torch.load(path, weights_only=True)
```

`weights_only=True` yalnızca tensör ve basit tipler yükler. Üçüncü taraftan
alınan bir `.pt` dosyası bu olmadan açılmamalı. Bu kural
`ruff`'un `S` (flake8-bandit) ailesiyle kısmen denetlenir; `S301`
(pickle) ve `S324` (hash) kuralları seçili.

### Otomatik model indirme

`ultralytics` ilk `YOLO("yolov8n.pt")` çağrısında dosya yoksa internetten
indirir. Bu bir tedarik zinciri yüzeyidir. Açık kaynak kullanıcıları için
`README.md`'de belgelenir ve model ağırlıklarının SHA256 değeri release
notlarında yayınlanır:

```bash
sha256sum models/*.pt
```

## Desteklenen sürümler

| Sürüm | Destek |
| --- | --- |
| `0.x` (güncel) | güvenlik düzeltmeleri alır |
| önceki `0.x` | almaz |

`1.0.0` öncesinde geriye dönük güvenlik yaması taahhüdü yoktur. Güvenlik
düzeltmesi en güncel sürüme uygulanır.

## Bu politikanın sınırları

Bu depo tek kişi tarafından yürütülen bir akademik projedir. Kurumsal bir
SRC (Security Response Center) kapasitesi yoktur. Bildirilen açıklar için
ödül programı, SLA veya CVE tahsisi taahhüt edilmez.
```

Son bölüm dürüstlük açısından gerekli. 48 saat yanıt sözü verilip
tutulamaması, hiç söz vermemekten daha kötü.

---

## 3. `CODE_OF_CONDUCT.md`

Contributor Covenant v2.1 tam metni kullanılır:

```bash
curl -o CODE_OF_CONDUCT.md \
  https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md
```

İndirildikten sonra değiştirilecek tek yer:

```markdown
## Uygulama

Taciz, tehdit veya diğer kabul edilemez davranışlar şu adrese bildirilmelidir:
<iletişim adresi>
```

Metnin geri kalanı **değiştirilmemeli**. Contributor Covenant'ın değeri,
standart ve tanınmış olması; yerelleştirilmiş bir sürüm bu faydayı yok eder.

Tek kişilik bir projede davranış kuralları gereksiz görünebilir. Gerekçe:
depo açık kaynak olduğunda issue'lara dışarıdan yorum gelir. Bir sınır ihlali
yaşandığında, moderasyon yetkisinin dayanağı bu belge. Sonradan yazmak, olay
sırasında yazmaktan daha zor.

---

## 4. `README.md` iskeleti

```markdown
# Araç/Nesne Hız Tahmini — Pinhole + YOLO + Mekatronik Yer Gerçeği

Tek bir kamera ile nesne hızı tahmini. Pinhole kamera modeli ile derinlik,
YOLOv8 ile tespit, ESP32 + TMC2208 vidalı mil sistemi ile mikrometre
düzeyinde yer gerçeği.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Durum

| Bileşen | Durum |
| --- | --- |
| Pinhole derinlik + 3B hız | çalışıyor |
| ESP32 yer gerçeği donanımı | çalışıyor |
| Kalman filtresi | planlanıyor |
| Nöral artık kalibratör (NRC) | planlanıyor |
| Makale | hazırlanıyor |

Bu tablo dürüst olmalı: "planlanıyor" satırlarını "çalışıyor" diye yazmak
kullanıcıyı yanıltır ve issue olarak geri döner.

## Hızlı başlangıç

```bash
git clone https://github.com/OWNER/REPO.git && cd REPO
./scripts/bootstrap.sh
uv run python -m vehicle_speed.cli --video video1.mp4 --initial-dist 1.83
```

## Nasıl çalışır

1. **Tespit** — YOLOv8 nesneyi bulur, `track()` ile kimlik atar.
2. **Derinlik** — `Z = (f · W) / P`. `W` bilinen fiziksel genişlik
   (plaka 0.52 m, far arası 1.5 m, top 0.185 m), `P` görüntüdeki piksel
   genişliği.
3. **Hız** — ardışık ölçümler arası 3B öklid mesafesi / geçen süre.
4. **Yer gerçeği** — ESP32 vidalı mil, 400 adım/mm çözünürlükle bilinen
   hızda hareket üretir; JSON telemetri ile raporlanır.

## Bilinen sınırlama

Derinlik piksel genişliği ile ters orantılı (`Z ∝ 1/P`). Hata yayılımı:

```
δZ = (Z² / (W·f)) · δP    ⟹    δv ∝ Z²·δP
```

Nesne uzaklaştıkça hız hatası uzaklığın karesiyle büyür. ±1 px titreme,
25 m'de metrelerce hataya yol açar. Kalman filtresi ve NRC bu sorunu
azaltmak için planlanıyor; ikisi de henüz kod tabanında yok.

## Depo yapısı

... (01_hedef_monorepo.md içindeki ağacın kısaltılmışı)

## Katkı

[CONTRIBUTING.md](CONTRIBUTING.md) · [Issue aç](../../issues/new/choose)

## Atıf

[CITATION.cff](CITATION.cff) — makale henüz yayınlanmadı; yazılım atfı için
bu dosyayı kullanın.

## Lisans

[Apache-2.0](LICENSE). `ultralytics` bağımlılığı AGPL-3.0 lisanslıdır;
dağıtım etkileri için bkz. `docs/infrastructure/05_yonetisim/05_lisans_citation.md`.

## Dokümantasyon

- [Donanım kurulumu](docs/hardware_setup.md)
- [Telemetri protokolü](docs/telemetry_protocol.md)
- [Altyapı raporları](docs/infrastructure/)
- [Makale stratejisi](docs/paper/Q2_makale_stratejisi.md)
```

**Rozet (badge) uyarısı:** `actions/workflows/ci.yml/badge.svg` bağlantısı
workflow var olmadan 404 döner ve rozet "no status" gösterir. Rozetler,
workflow push edildikten **sonra** eklenmeli.

---

## 5. Doğrulama

```bash
# bootstrap betiğini kuru çalıştır
bash -n scripts/bootstrap.sh                     # sözdizimi

# doküman bağlantıları ölü mü
uv pip install markdown-link-check
npx markdown-link-check README.md CONTRIBUTING.md  # veya
grep -oE '\]\([^)]+\)' CONTRIBUTING.md | sed 's/](//;s/)//' | while read -r l; do
  [ -e "$l" ] || echo "ÖLÜ BAĞLANTI: $l"
done
```

Bağlantı denetimi elle yapılabilir ama kolayca atlanır. `03_pre_commit.md`
içine bir `markdown-link-check` kancası eklenebilir; ancak yerel dosya
bağlantıları ile GitHub'a özgü bağlantıları (`../../actions/...`) ayırt
etmediği için yanlış pozitif üretir. Karar: kanca olarak eklenmez, release
öncesi elle denetlenir.
