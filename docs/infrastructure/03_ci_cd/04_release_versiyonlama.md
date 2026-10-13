# Sürüm Yönetimi, Release ve Akademik Atıf

**Durum:** Depoda sürüm bilgisi yok. `pyproject.toml` yok, tag yok, `CHANGELOG.md` yok.
Bu rapor sıfırdan sürüm politikasını tanımlar.

---

## 1. Sürüm şeması: SemVer + `0.x` gerçekçiliği

```text
v0.1.0   → ilk çalıştırılabilir paket yapısı (Faz 2 sonu)
v0.2.0   → Kalman filtresi eklendi (PLANLANIYOR)
v0.3.0   → NRC modeli eklendi (PLANLANIYOR)
v1.0.0   → makale kabulü + tam veri seti + tekrarlanabilirlik paketi
```

`0.x` aşamasında API kırıcı değişiklikler `major` bump gerektirmez. Bu,
`commitizen` yapılandırmasında `major_version_zero = true` ile ifade edilir
(`02_kod_kalitesi/03_pre_commit.md` bölüm 4).

Neden `1.0.0` hemen değil: hakem incelemesi sırasında mimari değişebilir.
`1.0.0` "API kararlı" sözü verir; bu proje o aşamada değil.

### Bump kuralları

| Commit | Bump | Örnek |
| --- | --- | --- |
| `feat:` | minor | `feat(filters): add constant-velocity Kalman` → 0.2.0 |
| `feat!:` veya `BREAKING CHANGE:` footer | major (0.x'te minor) | `feat(geometry)!: change depth unit to metres` |
| `fix:` | patch | `fix(vision): guard unbound track_id` → 0.1.1 |
| `perf:` | patch | |
| `docs:`, `chore:`, `ci:`, `style:`, `test:` | **bump yok** | |
| `refactor:` | bump yok | davranış değişmedi |

`refactor` bump üretmemesi önemli: `B2` borcu (script'leri pakete taşıma)
büyük bir diff üretecek ama API'yi **yeni** oluşturuyor, kırmıyor. Taşıma
`feat` olarak işaretlenirse 0.1.0'a bump yapar — bu doğru davranış.

---

## 2. `release.yml`

```yaml
name: Release

on:
  push:
    tags: ["v*.*.*"]
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Yalnızca kuru çalıştırma, yayınlama"
        type: boolean
        default: true

permissions:
  contents: read

concurrency:
  group: release-${{ github.ref }}
  cancel-in-progress: false      # release asla iptal edilmemeli

jobs:
  verify:
    name: verify-tag
    runs-on: ubuntu-latest
    timeout-minutes: 10
    outputs:
      version: ${{ steps.version.outputs.version }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0        # CHANGELOG üretimi için tam geçmiş

      - name: Extract version from tag
        id: version
        run: |
          set -euo pipefail
          TAG="${GITHUB_REF_NAME#v}"
          echo "version=$TAG" >> "$GITHUB_OUTPUT"

      - name: Verify tag matches pyproject and __init__
        run: |
          set -euo pipefail
          TAG="${GITHUB_REF_NAME#v}"
          PY=$(grep -m1 '^version' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')
          INIT=$(grep -m1 '__version__' src/vehicle_speed/__init__.py | sed -E "s/.*'([^']+)'.*/\1/")
          echo "tag=$TAG pyproject=$PY __init__=$INIT"
          [ "$TAG" = "$PY" ] || { echo "pyproject.toml surumu tag ile eslesmiyor" >&2; exit 1; }
          [ "$TAG" = "$INIT" ] || { echo "__init__.py surumu tag ile eslesmiyor" >&2; exit 1; }

  build:
    name: build-distributions
    needs: verify
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Build sdist and wheel
        run: uv build

      - name: Verify wheel contents
        run: |
          set -euo pipefail
          uv tool install check-wheel-contents
          check-wheel-contents dist/

      - name: Install built wheel in clean env
        run: |
          set -euo pipefail
          uv venv /tmp/smoke --python 3.12
          uv pip install --python /tmp/smoke/bin/python dist/*.whl
          /tmp/smoke/bin/python -c "from vehicle_speed.geometry import estimate_depth; \
assert abs(estimate_depth(1200.0, 0.52, 100.0) - 6.24) < 1e-9; print('wheel OK')"

      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/*
          retention-days: 30

  firmware-artifacts:
    name: build-firmware-binary
    needs: verify
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Cache PlatformIO
        uses: actions/cache@v4
        with:
          path: ~/.platformio
          key: pio-release-${{ hashFiles('firmware/platformio.ini') }}
      - run: pip install platformio==6.1.16
      - name: Build
        working-directory: firmware
        run: pio run && pio run -t size | tee ../firmware-size.txt
      - uses: actions/upload-artifact@v4
        with:
          name: firmware-bin
          path: |
            firmware/.pio/build/esp32doit-devkit-v1/firmware.bin
            firmware-size.txt

  publish:
    name: publish-release
    needs: [verify, build, firmware-artifacts]
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: write          # release oluşturmak için
    environment: release
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/download-artifact@v4
        with:
          pattern: "*"
          merge-multiple: false

      - name: Generate release notes from CHANGELOG
        id: notes
        run: |
          set -euo pipefail
          V="${{ needs.verify.outputs.version }}"
          awk -v v="## \[$V\]" '
            $0 ~ v {flag=1; next}
            /^## \[/ {flag=0}
            flag {print}
          ' CHANGELOG.md > notes.md
          [ -s notes.md ] || echo "Bu surum icin CHANGELOG girdisi bulunamadi." > notes.md

      - name: Create GitHub Release
        if: ${{ !inputs.dry_run }}
        uses: softprops/action-gh-release@7b4da11513bf3f43f9999e90eabced41ab8c0451  # v2.2.0
        with:
          body_path: notes.md
          draft: false
          prerelease: ${{ contains(needs.verify.outputs.version, '-') }}
          files: |
            dist/*
            firmware-bin/*
          generate_release_notes: false
```

### Kritik noktalar

**`fetch-depth: 0`.** Varsayılan `1` (yalnızca son commit). CHANGELOG üretimi ve
`cz changelog` tam geçmiş gerektirir. Bu satır atlanırsa release notları boş çıkar.

**Sürüm üç yerde.** `tag`, `pyproject.toml`, `src/vehicle_speed/__init__.py`.
`verify-tag` job'ı üçünü karşılaştırır. Bu denetim olmadan `v0.2.0` tag'i
`0.1.0` koduyla yayınlanabilir ve kimse fark etmez.

**`check-wheel-contents`.** `uv build` başarılı olsa bile paket içeriği yanlış
olabilir: `py.typed` eksik, `src/` dizini tekerleğin içine olduğu gibi girmiş,
veri dosyaları dahil edilmemiş. Bu araç bu sınıf hataları yakalar.

**Temiz ortamda kurulum testi.** `dist/*.whl` dosyasını boş bir venv'e kurup
gerçekten import edilebildiğini doğrular. Bu adım olmadan "tekerlek üretildi ama
kurulamıyor" durumu release'ten sonra fark edilir.

**`prerelease` koşulu.** `v0.3.0-rc.1` gibi sürümler otomatik ön-sürüm olarak
işaretlenir. SemVer ön-sürüm eki `-` ile ayrılır.

**`softprops/action-gh-release` SHA ile sabitlendi.** Üçüncü taraf action,
`01_pipeline_mimarisi.md` bölüm 6 politikası gereği tag değil SHA.

---

## 3. Tag üretimi

Elle:

```bash
cz bump --dry-run          # hangi surum gelecek
cz bump --changelog        # CHANGELOG.md uret + version dosyalarini guncelle + tag at
git push origin main --follow-tags
```

`cz bump` şunları yapar: `pyproject.toml` ve `src/vehicle_speed/__init__.py`
içindeki sürümü günceller, `CHANGELOG.md`'ye yeni bölüm ekler, commit atar,
tag oluşturur. `--follow-tags` olmadan tag push edilmez ve release workflow'u
tetiklenmez — bu en sık yapılan hata.

`pyproject.toml`'daki `bump_files` yapılandırması iki dosyayı da kapsıyor
(`02_kod_kalitesi/03_pre_commit.md` bölüm 4). Üçüncü bir sürüm kaynağı
eklenirse (ör. `CITATION.cff`) bu listeye eklenmeli, aksi halde ayrışır.

---

## 4. `CITATION.cff` ve akademik atıf

Bu bir araştırma projesi; yazılım atıfının makale atıfından ayrılması gerekir.

```yaml
cff-version: 1.2.0
message: "Bu yazılımı kullanıyorsanız, lütfen aşağıdaki makaleye atıf yapın."
title: "Physics-Informed Hybrid Vehicle Speed Estimation with Neural Residual Calibration"
type: software
license: Apache-2.0
repository-code: "https://github.com/<kullanici>/<depo>"
version: 0.1.0
date-released: "2026-10-13"
keywords:
  - vehicle-speed-estimation
  - pinhole-camera-model
  - kalman-filter
  - physics-informed-machine-learning
  - esp32
  - yolo
authors:
  - family-names: "Yıldız"
    given-names: "Kadir"
    orcid: "https://orcid.org/0000-0000-0000-0000"
    affiliation: "<üniversite>"
preferred-citation:
  type: article
  title: "<makale başlığı — kabul sonrası doldurulacak>"
  authors:
    - family-names: "Yıldız"
      given-names: "Kadir"
  journal: "<dergi>"
  year: 2026
  doi: "<kabul sonrası>"
  notes: "Makale henüz yayınlanmadı. Bu alan kabul sonrası güncellenecek."
```

`preferred-citation` alanı bilinçli olarak boş bırakıldı. Makale henüz yayın
aşamasında (`Q2_MAKALE_STRATEJISI_VE_YAPAY_ZEKA_RAPORU.md`). Uydurma DOI veya
dergi adı yazmak, atıf dosyasının güvenilirliğini yok eder. `notes` alanı bu
durumu açıkça belirtir.

GitHub, `CITATION.cff` varlığında depo sayfasında otomatik "Cite this
repository" düğmesi gösterir. Bu, akademik görünürlük açısından somut fayda.

### Sürüm ↔ makale eşlemesi

`CITATION.cff` içindeki `version` alanı da `cz bump` ile güncellenmeli, yoksa
`0.1.0`'da kalır:

```toml
bump_files = [
    { pattern = "pyproject.toml", pattern_replacement = 'version = "$new_version"' },
    { pattern = "src/vehicle_speed/__init__.py", pattern_replacement = '__version__ = "$new_version"' },
    { pattern = "CITATION.cff", pattern_replacement = 'version: "$new_version"' },
]
```

Makalede hangi sürümün kullanıldığı **açıkça yazılmalı**. Öneri: makale kabul
edildiğinde o sürüm için `paper/<dergi>-v<sayı>` adında kalıcı bir tag atılır:

```bash
git tag -a paper/sensors-v1 -m "Makalede bildirilen sonuçların tam olarak üretildiği revizyon"
```

Bu tag, tekrarlanabilirlik iddiasının kanıtıdır. Hakem "sonuçları üretemiyorum"
derse, hangi commit'te denediği sorulur ve bu tag gösterilir.

---

## 5. `CHANGELOG.md` biçimi

`cz changelog` otomatik üretir. Elle yazılmaz. Biçim:

```markdown
## [0.2.0] - 2026-11-05

### Feat

- **filters**: sabit hızlı Kalman filtresi eklendi (#23)
- **telemetry**: pyserial tabanlı okuyucu, port argparse ile parametrik (#18)

### Fix

- **vision**: tespit döngüsünde atanmamış track_id kullanımı düzeltildi (#12)
- **calibration**: başarısız kalibrasyon artık sessizce yutulmuyor (#14)

### Refactor

- script'ler src/vehicle_speed paketine taşındı (#8)
```

`(#23)` biçimindeki issue referansları commit mesajı gövdesinden gelir.
Conventional Commits'te scope zorunlu tutulursa (`feat(telemetry): ...`)
CHANGELOG bileşen bazında gruplanır — bu, `01_label_taksonomisi.md` içindeki
`component/*` etiketleriyle aynı adlandırma.

---

## 6. Depo dışı dağıtım: model ağırlıkları

`best.pt` (53 MB), `detector.pt` (6.2 MB), `yolov8n.pt`, `yolov8s.pt` depoya
**girmez** (`06_veri_model_yonetimi/01_gitignore_lfs_veri_izlenebilirligi.md`).
Ama tekrarlanabilirlik için erişilebilir olmalılar.

Seçenekler:

| Yöntem | Boyut sınırı | Sürümleme | Karar |
| --- | --- | --- | --- |
| GitHub Release eki | 2 GB/dosya | tag'e bağlı | **kabul** |
| Git LFS | 1 GB kota (ücretsiz) | commit'e bağlı | reddedildi (kota) |
| Hugging Face Hub | geniş | model kartı | `PLANLANIYOR` için değerlendir |
| Zenodo | 50 GB | DOI verir | makale veri seti için **kabul** |

Önerilen bölünme:

- **Model ağırlıkları** → GitHub Release eki (`v0.1.0` altına `detector.pt`)
- **Ham video ve deney verileri** → Zenodo (DOI alır, makalede atıf yapılır)
- **Eğitilmiş NRC checkpoint** → Hugging Face Hub veya Release

Zenodo'nun DOI vermesi makale açısından belirleyici: hakem "veri seti nerede?"
diye sorduğunda kalıcı bir tanımlayıcı sunulabilir. GitHub Release URL'si depo
silinirse ölür; DOI ölmez.

Release eki yükleme (yerelden):

```bash
gh release upload v0.1.0 detector.pt best.pt --clobber
gh release view v0.1.0
```

İndirme tarafı `README.md`'de belgelenmeli:

```bash
gh release download v0.1.0 --pattern "*.pt" --dir models/
```

---

## 7. `dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels: ["type/chore", "component/ci-cd", "dependencies"]
    groups:
      dev-tools:
        patterns: ["ruff", "mypy", "pytest*", "pre-commit", "coverage"]
      runtime:
        patterns: ["numpy", "scipy", "opencv-python-headless", "pyserial"]

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels: ["type/chore", "component/ci-cd", "dependencies"]

  - package-ecosystem: "gitsubmodule"
    directory: "/"
    schedule:
      interval: "monthly"
```

`groups` yapılandırması önemli: gruplama olmadan Dependabot her paket için ayrı
PR açar ve haftada 8-10 PR birikir. `ruff`/`mypy`/`pytest` birlikte güncellenir,
tek PR.

**Dikkat:** `ruff` ve `mypy` sürümleri `.pre-commit-config.yaml`'da da sabit.
Dependabot `pyproject.toml`'u günceller ama pre-commit dosyasını **güncellemez**
(ayrı ekosistem). Sonuç: sürüm ayrışması. Tam da bu yüzden
`02_ci_workflow_adimlari.md` bölüm 2.3'teki eşitleme denetimi `lint` job'ında
çalışıyor. Dependabot PR'ı açıldığında bu denetim kırmızı olur ve geliştirici
iki dosyayı birlikte güncellemeye zorlanır.

PlatformIO bağımlılıkları (`AccelStepper`, `ArduinoJson`) Dependabot
tarafından izlenemez — `platformio` ekosistemi desteklenmiyor. Firmware
bağımlılıkları elle izlenmeli; `firmware.yml` içine eklenen bir adım
mevcut sürümleri raporlayabilir:

```yaml
- name: Report outdated firmware libs
  working-directory: firmware
  run: pio pkg list --only-libs && pio pkg outdated || true
```

`|| true` çünkü bu bilgi amaçlı, kapı değil.

---

## 8. Doğrulama

```bash
# yerel bump provası (hiçbir şey degistirmeden)
cz bump --dry-run --no-verify

# CHANGELOG onizleme
cz changelog --start-rev=v0.1.0 --dry-run

# release workflow'unu kuru calistir
gh workflow run release.yml -f dry_run=true

# tag ve surum eslesmesini elle kontrol et
git describe --tags
grep '^version' pyproject.toml
grep '__version__' src/vehicle_speed/__init__.py
grep '^version:' CITATION.cff
```

Dört kaynağın aynı sürümü göstermesi gerekir. `verify-tag` job'ı bunu CI'da
zorunlu kılar, ama yerel prova push öncesi yakalamayı sağlar.

---

## 9. Açık noktalar

1. **PyPI yayını yok.** `publish` job'ı yalnızca GitHub Release üretiyor,
   PyPI'ye yüklemiyor. Gerekçe: bu bir araştırma paketi, `pip install
   vehicle-speed-estimation` ile kurulmasının somut bir kullanıcısı yok.
   İleride gerekirse `uv publish` adımı eklenir ve `PYPI_API_TOKEN` secret'ı
   tanımlanır. Karar bilinçli olarak ertelendi.
2. **`environment: release` koruması.** `publish` job'ı bir GitHub environment
   kullanıyor. Bu environment için "required reviewers" ayarlanmalı, aksi halde
   koruma sağlamaz. Ayar depo web arayüzünden yapılır, YAML ile ifade edilemez.
3. **İmzalı commit/attestation.** `gh attestation` ile artefakt kökeni
   kanıtlanabilir. Akademik tekrarlanabilirlik için değerli ama bu fazda kapsam
   dışı.
