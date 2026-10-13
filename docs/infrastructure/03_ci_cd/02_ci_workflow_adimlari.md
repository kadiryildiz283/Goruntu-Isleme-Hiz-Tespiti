# CI Workflow Adımları: `ci.yml` Satır Satır

Bu rapor `.github/workflows/ci.yml` dosyasının tam içeriğini ve her adımın
neden orada olduğunu açıklar.

**Doğrulama durumu:** Bu workflow henüz çalıştırılmadı. Depo bozuk
(`.git/objects` boş) ve uzak depo yok. Aşağıdaki YAML `check-jsonschema`
şemasına uygun yazıldı ama gerçek koşu doğrulaması push sonrasına kalır.
Beklenen ilk koşu süresi: soğuk önbellekle ~3 dk.

---

## 1. Tam dosya

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:
    inputs:
      reason:
        description: "Manuel çalıştırma sebebi"
        required: false
        default: "manual"

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}

env:
  UV_LINK_MODE: copy
  UV_PYTHON: "3.12"

jobs:
  lint:
    name: lint
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: "pyproject.toml"

      - name: Install ruff and mypy only
        run: uv tool install ruff==0.16.7 && uv tool install mypy==1.11.2

      - name: Ruff check
        run: ruff check --output-format=github .

      - name: Ruff format check
        run: ruff format --check --diff .

      - name: Verify pre-commit revisions match locked dev deps
        run: |
          set -euo pipefail
          python3 - <<'PY'
          import re, sys, tomllib, pathlib
          cfg = pathlib.Path(".pre-commit-config.yaml").read_text()
          py = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
          dev = py["project"]["optional-dependencies"]["dev"]
          pinned = {}
          for item in dev:
              m = re.match(r"([A-Za-z0-9_.-]+)\s*==\s*v?([0-9][0-9.]*)", item)
              if m:
                  pinned[m.group(1).lower()] = m.group(2)
          problems = []
          for pkg, frag in (("ruff", "ruff-pre-commit"), ("mypy", "mirrors-mypy")):
              if pkg not in pinned:
                  continue
              block = re.search(rf"{re.escape(frag)}.*?rev:\s*v?([0-9.]+)", cfg, re.S)
              if block and block.group(1) != pinned[pkg]:
                  problems.append(
                      f"{pkg}: pyproject={pinned[pkg]} pre-commit={block.group(1)}"
                  )
          if problems:
              print("Surum ayrismasi:\n" + "\n".join(problems))
              sys.exit(1)
          print("ruff/mypy surumleri eslesik.")
          PY

  typecheck:
    name: typecheck
    needs: lint
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Sync dev dependencies (no torch)
        run: uv sync --extra dev --no-install-package torch --no-install-package ultralytics

      - name: Cache mypy
        uses: actions/cache@v4
        with:
          path: .mypy_cache
          key: mypy-${{ runner.os }}-${{ hashFiles('pyproject.toml', 'uv.lock') }}

      - name: Mypy
        run: uv run mypy

  unit-tests:
    name: unit-tests
    needs: typecheck
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Sync dev dependencies (no torch)
        run: uv sync --extra dev --no-install-package torch --no-install-package ultralytics

      - name: Run unit tests
        run: uv run pytest tests/unit -q --no-cov

  integration-matrix:
    name: integration-matrix
    needs: unit-tests
    runs-on: ${{ matrix.os }}
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Install Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-suffix: ${{ matrix.os }}-${{ matrix.python-version }}

      - name: Sync with CPU-only torch
        run: uv sync --extra dev --extra torch-cpu --python "${{ env.PYTHON_PATH }}"
        env:
          PYTHON_PATH: ${{ env.pythonLocation }}/python

      - name: Run integration tests with coverage
        run: uv run pytest tests/ -q --cov=src --cov-report=xml --cov-report=term-missing

      - name: Upload coverage
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
          retention-days: 14

      - name: Enforce coverage threshold
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
        run: uv run coverage report --fail-under=85

  summary:
    name: All checks
    needs: [lint, typecheck, unit-tests, integration-matrix]
    if: always()
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Verify all required jobs succeeded
        run: |
          set -euo pipefail
          results='${{ toJson(needs) }}'
          echo "$results"
          if echo "$results" | grep -qE '"result": *"(failure|cancelled)"'; then
            echo "En az bir job basarisiz veya iptal edildi" >&2
            exit 1
          fi
          echo "Tum kontroller yesil."
```

---

## 2. Adım adım gerekçe

### 2.1 `lint` job'ı neden proje ortamını kurmuyor

```yaml
- name: Install ruff and mypy only
  run: uv tool install ruff==0.16.7 && uv tool install mypy==1.11.2
```

Yapıştırılan plan `uv pip install --system ".[dev]"` kullanıyordu. Bu, lint için
`torch`, `ultralytics`, `opencv` ve `scipy` kurmayı gerektirir — ~40 saniye ve
~1.5 GB disk. Oysa ruff ve mypy'nin çalışması için proje bağımlılıklarının
kurulu olması **gerekmiyor**:

- `ruff check` yalnızca kaynak dosyaları ayrıştırır, import çözmez.
- `mypy` üçüncü taraf stub'ları olmadan da çalışır; `ignore_missing_imports`
  zaten ayarlı (`02_mypy_strict.md` bölüm 2).

`uv tool install` izole bir araç ortamı kurar, saniyeler sürer. Bu değişiklik
lint job'ını ~40 sn'den ~10 sn'ye indirir ve lint her push'ta çalıştığı için
tasarruf çarpanı yüksek.

Ödünleşim: mypy, `numpy` gibi kurulu paketlerin tiplerini göremez ve
`import-untyped` uyarısı üretir. `[[tool.mypy.overrides]]` içindeki
`ignore_missing_imports = true` bunu bastırır. Tam tip doğruluğu `typecheck`
job'ında, gerçek ortam kurulduktan sonra denetlenir.

### 2.2 `ruff format --check --diff`

`--check` tek başına yalnızca "fark var" der. `--diff` eklenince CI log'unda
tam olarak hangi satırların değişmesi gerektiği görünür. Katkıcı log'dan
düzeltmeyi okuyabilir, yerelde yeniden çalıştırmak zorunda kalmaz.

### 2.3 Sürüm eşitleme denetimi

`lint` job'ındaki Python bloğu, `pyproject.toml`'daki `ruff==0.16.7` ile
`.pre-commit-config.yaml`'daki `rev: v0.16.7` değerlerinin ayrışmadığını
denetler. Gerekçe: `03_pre_commit.md` bölüm 7.

Bu adım olmadan iki yer arasındaki sürüm farkı sessizce büyür ve "yerelde
geçiyor, CI'da kalıyor" şikâyetleri başlar.

### 2.4 `--no-install-package torch`

```yaml
run: uv sync --extra dev --no-install-package torch --no-install-package ultralytics
```

`typecheck` ve `unit-tests` job'ları torch'a ihtiyaç duymaz, çünkü
`01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md` bölüm 4'teki import
sınırı sayesinde `geometry.py`, `tracking.py`, `calibration.py`, `telemetry/*`
modülleri torch'u hiç import etmez.

Tasarruf: torch CPU tekerleği ~200 MB, ultralytics ~700 KB + kendi bağımlılıkları.
İki job × her push = anlamlı.

**Risk:** Birisi `geometry.py`'ye yanlışlıkla `import torch` eklerse `typecheck`
job'ı `ModuleNotFoundError` verir. Bu **istenmeyen bir arıza değil, bir
özellik**: import sınırının ihlalini CI'da yakalar. Daha açık bir hata mesajı
için `import-linter` eklenebilir (bölüm 4'te opsiyonel adım).

### 2.5 Matris job'ında `setup-python` + `setup-uv` birlikte

`lint`/`typecheck`/`unit-tests` job'larında `setup-python` yok, çünkü
`env.UV_PYTHON: "3.12"` ile uv kendi Python'unu indiriyor. Matris job'ında ise
sürüm **runner tarafından** sağlanmalı, aksi halde her hücre aynı Python'u
kullanır ve matris anlamsızlaşır.

`--python "${{ env.pythonLocation }}/python"` ifadesi uv'ye `setup-python`'ın
kurduğu yorumlayıcıyı kullanmasını söyler. Bu satır atlanırsa uv kendi 3.12'sini
kullanır ve `python-version: "3.10"` hücreleri fiilen 3.12'de koşar — sessiz,
tehlikeli bir hata.

### 2.6 `cache-suffix`

```yaml
cache-suffix: ${{ matrix.os }}-${{ matrix.python-version }}
```

Altı matris hücresi aynı cache anahtarını paylaşırsa, birbirlerinin önbelleğini
geçersiz kılarlar (her hücre farklı bir `uv.lock` çözümlemesi yazar). Son ek
ayrıştırması bunu önler.

### 2.7 Kapsam eşiği

```yaml
- name: Enforce coverage threshold
  run: uv run coverage report --fail-under=85
```

`pytest --cov` kapsamı ölçer ama eşiği zorlamaz. `--cov-fail-under=85` da
kullanılabilir; ayrı adım olarak yazılmasının sebebi log'da kapsam özetinin
açıkça görünmesi ve eşik ihlalinin test hatasından ayrılması.

Eşik yalnızca tek hücrede (`ubuntu-latest, 3.12`) zorlanır. Altı hücrede de
zorlamak aynı ölçümü altı kez tekrarlar.

---

## 3. `pyproject.toml` tarafında gereken ekler

Workflow'un çalışması için `pyproject.toml`'da şu bölümler olmalı:

```toml
[project.optional-dependencies]
dev = [
    "ruff==0.16.7",
    "mypy==1.11.2",
    "pre-commit>=3.7",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.14",
    "coverage[toml]>=7.4",
]
torch-cpu = [
    "torch>=2.2",
    "ultralytics>=8.0",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform != 'darwin'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config"
markers = [
    "hardware: fiziksel ESP32 veya kamera gerektirir (CI'da atlanır)",
    "slow: 5 saniyeden uzun süren testler",
]

[tool.coverage.run]
source = ["src"]
branch = true
omit = ["*/visualization.py", "*/cli.py"]

[tool.coverage.report]
fail_under = 85
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

### `[tool.uv.sources]` ile CPU-only torch

Yapıştırılan plan `--extra-index-url https://download.pytorch.org/whl/cpu`
kullanıyordu. Bu çalışır ama iki sorunu var:

1. `--extra-index-url` ek bir indeks ekler, varsayılan PyPI'yi kaldırmaz. uv/pip
   en yüksek sürümü seçer, bu da CUDA'lı tekerlek olabilir → 2.5 GB indirme.
2. Komut satırında olduğu için `uv sync` ile tekrarlanamaz; `uv.lock` içine
   girmez.

`[[tool.uv.index]] explicit = true` yaklaşımı yalnızca açıkça yönlendirilen
paketlerin o indeksten gelmesini garanti eder. CUDA tekerleği hiç çözümlenmez.

### `omit = ["*/visualization.py", "*/cli.py"]`

Görselleştirme ve CLI giriş noktası kapsam hesabından çıkarılır. Gerekçe:
`cv2.imshow` CI'da çalışmaz (`A7` borcu), bu dosyaları kapsamaya çalışmak
mock'ların gerçeklikten kopmasına yol açar. Kararın bedeli dürüstçe
kaydedilmeli: bu iki modül **test edilmiyor**. İleride `cli.py` için
`typer.testing.CliRunner` benzeri bir yaklaşımla test eklenebilir.

---

## 4. Opsiyonel: import sınırı denetimi

```yaml
      - name: Verify module boundaries
        run: uv run lint-imports
```

`import-linter` kurulursa (`01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md`
bölüm 4), `geometry.py`'nin `torch`/`cv2` import etmediği açıkça denetlenir.
`--no-install-package torch` hilesi de gereksizleşir, çünkü kural ihlali
anlamlı bir hata mesajıyla raporlanır.

Öneri: Faz 3'te eklensin. İlk kurulumda kapsamı dar tutmak daha sağlıklı.

---

## 5. Windows'a özgü tuzaklar

Matris `windows-latest` içerdiği için bilinen farklar:

| Konu | Linux | Windows | Etki |
| --- | --- | --- | --- |
| Yol ayracı | `/` | `\` | `Path` kullanılmazsa bozulur |
| `termios` | var | **yok** | `listen.py` import anında `ModuleNotFoundError` |
| Dosya kilidi | gevşek | sıkı | açık dosya silinemez, `tmp_path` temizliği patlayabilir |
| Satır sonu | `LF` | `CRLF` | `.gitattributes` olmadan diff gürültüsü |
| Büyük/küçük harf | duyarlı | duyarsız | `check-case-conflict` kancası şart |
| `os.read` blok davranışı | kesilebilir | farklı | seri port okuma zaman aşımı |

`termios` sorunu somut: `listen.py` ilk satırlarda `import termios` yapıyor.
Windows'ta bu modül hiç yok. `pyserial` geçişi (`A4` borcu) yapılmadan Windows
matrisi **kesinlikle kırmızı** olur.

İki seçenek:

1. `pyserial` geçişini Windows matrisini açmadan **önce** yapmak (önerilen)
2. Matrise başlarken yalnızca `ubuntu-latest` almak, Windows'u `A4`
   tamamlandıktan sonra eklemek

Uygulama yol haritasında seçenek 2 benimsendi: Faz 2'de yalnızca Ubuntu,
Faz 4'te Windows eklenir. Gerekçe: `07_uygulama_yol_haritasi/01_adim_adim_uygulama.md`.

---

## 6. Doğrulama

Workflow push edilmeden önce yerelde doğrulanabilir:

```bash
# YAML şeması
uv pip install check-jsonschema
check-jsonschema --check-github-workflows .github/workflows/ci.yml

# expression sözdizimi
gh workflow list          # uzak depo kurulduktan sonra
gh run list --workflow ci.yml
gh run watch              # canlı izleme
gh run view --log-failed  # yalnızca başarısız adımların logu
```

İlk koşuda beklenen başarısızlıklar (kod taşınmadıysa):

1. `ruff check .` → kök dizindeki 141 hata
2. `mypy` → `src/` yoksa "no files to check"
3. `pytest tests/` → dizin yoksa hata

Bu yüzden workflow, `07_uygulama_yol_haritasi/01_adim_adim_uygulama.md`
içinde **Faz 2'nin sonunda** etkinleştirilir; Faz 1 (temizlik + paket yapısı)
bitmeden açılırsa ilk koşu kırmızı olur ve moral bozar.
