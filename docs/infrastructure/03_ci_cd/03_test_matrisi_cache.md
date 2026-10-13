# Test Matrisi, Önbellekleme ve Bağımlılık Stratejisi

Bu rapor üç kararı belgeler: hangi OS × Python hücreleri test edilecek,
önbellekler nasıl kurulacak, ağır bağımlılıklar (torch, ultralytics, opencv)
CI'da nasıl izole edilecek.

---

## 1. Matris kararı

### 1.1 Yerel ortamın gerçek durumu

Doğrulandı:

| Öğe | Değer |
| --- | --- |
| Sistem Python | 3.14.7 |
| `uv` | 0.13.0 |
| `torch` | kurulu **değil** |
| `ultralytics` | kurulu **değil** |
| `scipy` | kurulu **değil** |
| `opencv-python` | 5.0.0.93 |
| `numpy` | 2.5.1 |
| `pyserial` | 3.5 |

`kadir.py` `ultralytics` ve `scipy` import ediyor, ikisi de yok. Yani script
bugün bu makinede **çalıştırılamaz**. CI matrisi tasarlanırken yerel sürümün
(3.14) referans alınmaması gerektiğinin somut kanıtı bu.

### 1.2 Hücre seçimi

| OS | Python | Karar | Gerekçe |
| --- | --- | --- | --- |
| `ubuntu-latest` | 3.10 | matriste | `requires-python` alt sınırı |
| `ubuntu-latest` | 3.11 | matriste | yaygın, kararlı |
| `ubuntu-latest` | 3.12 | matriste | birincil geliştirme sürümü |
| `ubuntu-latest` | 3.13 | **doğrulandıktan sonra** | torch tekerlek bulunabilirliği belirsiz |
| `ubuntu-latest` | 3.14 | **hayır** | yerel sürüm bu olsa da torch/ultralytics desteği doğrulanmadı |
| `windows-latest` | 3.10-3.12 | **Faz 4'te** | `termios` bağımlılığı `pyserial`'e geçmeden kırık |
| `macos-latest` | — | **hayır** | somut ihtiyaç yok, runner çarpanı 10× |

Başlangıç matrisi: **3 hücre** (ubuntu × 3.10/3.11/3.12). Faz 4'te Windows
eklenince 6 hücre.

### 1.3 Yeni sürüm ekleme protokolü

Bir Python sürümü matrise eklenmeden önce şu komut çalıştırılmalı:

```bash
uv run --python 3.13 --with torch --with ultralytics --with scipy \
  python -c "import torch, ultralytics, cv2, scipy, numpy; \
print('torch', torch.__version__); print('ultralytics', ultralytics.__version__)"
```

Çıktı `OK` ise eklenir, `No solution found` veya `ImportError` ise eklenmez ve
bu raporun tablosuna "doğrulandı, desteklenmiyor" notu düşülür.

Bu kontrolün otomatikleştirilmiş hali — haftalık çalışıp sürüm desteğini
izleyen bir workflow:

```yaml
name: Dependency compatibility probe

on:
  schedule:
    - cron: "0 6 * * 1"   # her pazartesi 06:00 UTC
  workflow_dispatch:

permissions:
  contents: read
  issues: write

jobs:
  probe:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: astral-sh/setup-uv@v3
      - name: Probe core deps
        id: probe
        continue-on-error: true
        run: |
          uv pip install --system \
            --index-strategy unsafe-best-match \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            torch ultralytics scipy numpy opencv-python-headless
          python -c "import torch, ultralytics, scipy, cv2, numpy; print('OK')"
      - name: Report unsupported versions
        if: steps.probe.outcome == 'failure'
        run: |
          echo "Python ${{ matrix.python-version }} icin torch/ultralytics cozumlenemedi."
          echo "Bu surum matrise EKLENMEMELI." >> "$GITHUB_STEP_SUMMARY"
```

`continue-on-error: true` önemli: bu bir **araştırma** job'ı, başarısızlık
beklenen bir sonuç. `if: steps.probe.outcome == 'failure'` ile sonuç özete
yazılır. Haftalık cron sayesinde "3.13 desteği ne zaman geldi?" sorusu elle
sorulmak zorunda kalmaz.

---

## 2. Bağımlılık grupları

Tek bir dev bağımlılık listesi yerine ayrıştırılmış gruplar:

```toml
[project]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "opencv-python-headless>=4.9",
    "pyserial>=3.5",
]

[project.optional-dependencies]
ml = [
    "torch>=2.2",
    "ultralytics>=8.0",
]
dev = [
    "ruff==0.16.7",
    "mypy==1.11.2",
    "pre-commit>=3.7",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.14",
    "coverage[toml]>=7.4",
]
docs = [
    "mkdocs-material>=9.5",
]
```

Grup ayrımının CI üzerindeki etkisi:

| Job | Kurulan grup | İndirme boyutu |
| --- | --- | --- |
| `lint` | hiçbiri (`uv tool install`) | ~15 MB |
| `typecheck` | `dev` | ~60 MB |
| `unit-tests` | `dev` | ~60 MB |
| `integration-matrix` | `dev` + `ml` | ~400 MB (CPU torch) |

`opencv-python-headless` ana bağımlılık, `opencv-python` değil. Fark:

- `opencv-python`: GUI desteği dahil, `libGL.so.1` ve `libglib2.0-0` gerektirir.
  Ubuntu runner'da bu kütüphaneler yoksa `import cv2` şu hatayı verir:
  `ImportError: libGL.so.1: cannot open shared object file`.
- `opencv-python-headless`: GUI yok, sistem kütüphanesi gerektirmez, `imshow`
  sağlamaz.

CI'da `imshow` zaten kullanılmayacak (`visualization.py` kapsam dışında,
`02_ci_workflow_adimlari.md` bölüm 3). Bu yüzden headless doğru seçim.

**Çakışma uyarısı:** `opencv-python` ve `opencv-python-headless` aynı anda
kurulursa `cv2` modülü ikisinden hangisinin son kurulduğuna bağlı olarak
davranır. Yerel makinede `opencv-python 5.0.0.93` kurulu. `uv sync`
çalıştırıldığında ikisi birden ortamda olmasın diye `CONTRIBUTING.md`'ye
açık uyarı konulmalı (`05_yonetisim/03_contributing_ortam_kurulumu.md`).

Ek not: yerelde kurulu sürüm **5.0.0.93**, `>=4.9` kısıtı bunu kabul eder.
Ama OpenCV 5.x ile 4.x arasında API farkları olabilir. `uv.lock` yerelde
5.0.0.93 ile çözümlenirse, CI'da da aynı sürüm kurulur — bu iyi. Ancak
`requires-python = ">=3.10"` ile OpenCV 5.x'in Python 3.10 desteği
doğrulanmalı. Bu, bölüm 1.3'teki probe workflow'unun işi.

---

## 3. Önbellekleme katmanları

Dört ayrı önbellek, hepsi farklı anahtarla:

| Katman | Yol | Anahtar | Sağlayıcı |
| --- | --- | --- | --- |
| uv araç önbelleği | `~/.cache/uv` | `hashFiles('uv.lock')` | `setup-uv` `enable-cache` |
| Python sürümü | runner yönetimli | `python-version` | `setup-python` (varsayılan açık) |
| mypy | `.mypy_cache` | `hashFiles('pyproject.toml','uv.lock')` | `actions/cache@v4` |
| pytest | `.pytest_cache` | — | **önbelleklenmez** |
| PlatformIO | `~/.platformio` | `hashFiles('firmware/platformio.ini')` | `actions/cache@v4` |

### `cache-suffix` ile matris ayrıştırması

```yaml
- uses: astral-sh/setup-uv@v3
  with:
    enable-cache: true
    cache-suffix: ${{ matrix.os }}-${{ matrix.python-version }}
```

Altı hücre aynı anahtarı paylaşırsa her koşu bir öncekinin önbelleğini
geçersiz kılar. Son ek olmadan önbellek isabet oranı ~%17'ye (1/6) düşer.

### `.pytest_cache` neden önbelleklenmez

`.pytest_cache` son başarısız testleri saklar ve `--lf` ile yeniden koşmayı
hızlandırır. CI'da her koşu temiz başlar; önbelleklenen bir `.pytest_cache`
"önceki koşuda başarısızdı" bilgisini taşır ve `--lf` kullanılırsa testlerin
atlanmasına yol açabilir. Bu sınıf bir hata çok zor ayıklanır. Değmez.

### Önbellek anahtarında `uv.lock` zorunlu

```yaml
key: mypy-${{ runner.os }}-${{ hashFiles('pyproject.toml', 'uv.lock') }}
```

Yalnızca `pyproject.toml` hash'lenirse, `uv.lock` değiştiğinde (transitif
bağımlılık güncellemesi) önbellek tazelenmez ve mypy eski stub'larla çalışır.
Sonuç: yerelde geçen, CI'da kalan tip hataları.

**`uv.lock` depoya commit edilmeli.** Uygulama (`pyproject.toml`) için lock
dosyası zorunludur; kütüphane için tartışmalıdır. Bu proje hem paket hem
uygulama olduğu için (CLI + `scripts/`) lock commit edilir. Bu aynı zamanda
makale tekrarlanabilirliği için şart: hakem iki yıl sonra aynı sonuçları
üretebilmeli.

---

## 4. Torch izolasyonu: üç yaklaşım

### 4.1 `--extra-index-url` (yapıştırılan planın yöntemi)

```bash
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

**Sorun:** `--extra-index-url` PyPI'yi kaldırmaz, **ekler**. Çözümlendirici en
yüksek sürümü seçer. PyPI'daki `torch` CUDA'lı tekerlektir (~2.5 GB,
`nvidia-*` bağımlılıklarıyla birlikte ~5 GB). CPU indeksi aynı sürümü
sunuyorsa sonuç belirsiz. Runner disk kotası (14 GB `ubuntu-latest`) dolar.

### 4.2 `--index-url` (PyPI'yi tamamen değiştirir)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CPU tekerleğini garanti eder, ama `ultralytics` gibi PyPI'dan gelen paketler
çözümlenemez. İki ayrı `pip install` komutu gerekir ve sürüm tutarlılığı
bozulur.

### 4.3 `[tool.uv.sources]` + `explicit = true` (önerilen)

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

`explicit = true` bu indeksin **yalnızca** açıkça yönlendirilen paketler için
kullanılmasını sağlar. `numpy` PyPI'dan, `torch` CPU indeksinden gelir. Tek
`uv sync` komutu, tek `uv.lock`, tekrarlanabilir.

Doğrulama (kurulumdan sonra):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# beklenen: 2.x.y+cpu None
```

`+cpu` son eki ve `cuda is None` görülmüyorsa CUDA tekerleği kurulmuş demektir.

### Karşılaştırma

| Yaklaşım | Tekrarlanabilir | CUDA riski | Disk | Karar |
| --- | --- | --- | --- | --- |
| `--extra-index-url` | hayır | yüksek | ~5 GB | reddedildi |
| `--index-url` | kısmi | yok | ~200 MB | reddedildi |
| `[tool.uv.sources]` | evet | yok | ~200 MB | **kabul** |

---

## 5. Runner disk ve bellek sınırları

`ubuntu-latest` (GitHub-hosted): ~14 GB kullanılabilir disk, 4 çekirdek, 16 GB RAM.
`windows-latest`: ~14 GB disk, 4 çekirdek, 16 GB RAM.

Bu proje için tahmini disk kullanımı:

| Bileşen | Boyut |
| --- | --- |
| runner temel imajı | ~9 GB (kullanılabilir alanı zaten düşer) |
| torch CPU + torchvision | ~200 MB |
| ultralytics + bağımlılıkları | ~50 MB |
| opencv-python-headless | ~60 MB |
| numpy/scipy | ~80 MB |
| depo checkout (LFS hariç) | ~5 MB |
| **Toplam ek** | **~400 MB** |

Güvenli. Ama **model ağırlıkları indirilirse** durum değişir: `yolov8n.pt`
6.5 MB (sorun değil), `best.pt` 53 MB, `yolov8x.pt` 137 MB. Testler gerçek
model ağırlığı gerektirmemeli.

`04_test_mimarisi/01_test_piramidi.md` içinde bu karar netleştirildi: testler
**küçük, üretilmiş bir model** veya **mock dedektör** kullanır, gerçek
ağırlıkları indirmez. Ultralytics'in ilk `YOLO("yolov8n.pt")` çağrısında
otomatik indirme yapması CI'da ağ bağımlılığı yaratır; bu da
`04_test_mimarisi/02_mock_kamera_seri_port.md` bölüm 3'te mock ile çözülür.

---

## 6. Zaman aşımı bütçesi

| Job | `timeout-minutes` | Gerekçe |
| --- | --- | --- |
| `lint` | 5 | ölçülen ~10 sn, 30× pay |
| `typecheck` | 10 | mypy + cache restore |
| `unit-tests` | 10 | mock tabanlı, hızlı |
| `integration-matrix` | 20 | torch kurulumu + Windows yavaşlığı |
| `firmware` | 20 | espressif32 toolchain soğuk kurulum ~2 dk |
| `summary` | 5 | yalnızca JSON denetimi |

Zaman aşımı yazılmazsa varsayılan 360 dk. Donanım bekleyen bir test asılı
kalırsa 6 saat runner yanar. `@pytest.mark.hardware` ile CI'da atlanan testler
bile yanlış yapılandırılırsa bekleme yapabilir
(`04_test_mimarisi/02_mock_kamera_seri_port.md` bölüm 1).

---

## 7. Eşzamanlılık ve iptal

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
```

`main` dışında iptal açık. Gerekçe: `01_pipeline_mimarisi.md` bölüm 4.

Bir ek ayrıntı: `cancel-in-progress` iptal edilen koşuyu `cancelled` sonucuyla
bitirir. `summary` job'ındaki denetim `cancelled`'ı başarısızlık sayıyor:

```bash
if echo "$results" | grep -qE '"result": *"(failure|cancelled)"'; then
```

Bu doğru davranış — iptal edilmiş bir koşu "yeşil" sayılmamalı. Ama branch
koruma kuralı `main` için ayarlanırken, iptal edilmiş koşuların merge'i
bloklamaması gerekir; `main`'de iptal kapalı olduğu için bu çelişki oluşmaz.

---

## 8. Ölçüm: CI süresi gerçekten ne kadar

Tahmin değil, ölçüm yapılmalı. İlk beş koşudan sonra:

```bash
gh run list --workflow ci.yml --limit 10 \
  --json databaseId,createdAt,updatedAt,conclusion,displayTitle

# job bazında süre
gh api repos/:owner/:repo/actions/runs/<RUN_ID>/jobs \
  --jq '.jobs[] | {name, started_at, completed_at}'
```

Beklenen:

| Koşu | Süre |
| --- | --- |
| 1. koşu (soğuk önbellek) | ~3 dk |
| 2. koşu (sıcak önbellek, aynı lock) | ~50 sn |
| `uv.lock` değişmiş koşu | ~2 dk |

Eğer 2. koşu da 3 dk sürüyorsa önbellek çalışmıyor demektir. En sık sebep:
`cache-suffix` yanlış, veya `uv.lock` commit edilmemiş (her koşuda yeniden
çözümleniyor).

---

## 9. Açık noktalar

1. **OpenCV 5.x uyumu.** Yerelde 5.0.0.93 kurulu. `cv2.VideoCapture`,
   `cv2.imshow` gibi API'lerin 5.x'te değişip değişmediği doğrulanmadı.
   `uv.lock` oluşturulurken 4.x'e sabitlemek daha güvenli olabilir:
   `opencv-python-headless>=4.9,<5`. Karar için bölüm 1.3'teki probe çalıştırılmalı.
2. **`ultralytics` lisansı AGPL-3.0.** Bu, projenin kendi lisans seçimini
   etkiler (`05_yonetisim/05_lisans_citation.md` bölüm 3). CI açısından sorun
   değil, ama dağıtım açısından önemli.
3. **GPU testi yok.** GitHub'ın ücretsiz runner'larında GPU yok. NRC eğitimi
   (`PLANLANIYOR`) CI'da test edilemez; yalnızca CPU üzerinde küçük batch ile
   şekil/gecikme testleri yapılabilir. Gerçek eğitim yerel makinede veya
   ayrı bir self-hosted GPU runner'da.
