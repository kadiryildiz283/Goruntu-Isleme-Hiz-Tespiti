# Kabul Kriterleri ve "Bitti" Tanımı

Bu rapor her faz için nesnel kabul kriterlerini tanımlar. Amaç: "altyapı
kuruldu" beyanının doğrulanabilir olması. Her kriter bir komuta bağlı; komut
geçmiyorsa o faz bitmemiştir.

**İlke:** Bir fazın bittiği, o fazın doğrulama komutlarının çıktısıyla kanıtlanır.
"VALIDATION: PASS" yazmak kanıt değildir. Bu ilke,
`00_durum_analizi/01_mevcut_envanter.md` içinde belgelenen bozuk depo
durumunun ve yapıştırılan 50 adımlı plandaki doğrulanmamış beyanların
doğrudan sonucudur.

---

## Genel kural: üç kanıt türü

Bir işin bittiğini iddia etmek için üçünden en az ikisi gerekli:

| Kanıt türü | Örnek | Güç |
| --- | --- | --- |
| Komut çıktısı | `ruff check .` → `All checks passed!`, exit 0 | yüksek |
| Dosya sistemi | `ls src/vehicle_speed/geometry.py` → var | yüksek |
| CI koşusu | `gh run list --workflow ci.yml` → `success` | en yüksek |
| Beyan | "yapıldı", "PASS" | **kanıt sayılmaz** |

---

## Faz 0 — Depo hijyeni

| # | Kriter | Doğrulama | Eşik |
| --- | --- | --- | --- |
| 0.1 | `.git` çalışıyor | `git status` | exit 0 |
| 0.2 | Nesne veritabanı dolu | `find .git/objects -type f \| wc -l` | > 0 |
| 0.3 | En az bir commit var | `git log --oneline \| wc -l` | ≥ 1 |
| 0.4 | Büyük dosya sızmamış | `git ls-files -z \| xargs -0 stat -c '%s' \| sort -rn \| head -1` | < 5 MB |
| 0.5 | Depo hacmi makul | `du -sh .git` | < 20 MB |
| 0.6 | `.gitignore` ikili dosyaları yakalıyor | `git check-ignore best.pt video1.mp4 "Yapay zeka/yolov8x.onnx"` | üçü de listelenir |
| 0.7 | `docs/` görselleri istisna | `git check-ignore docs/x.png` | **çıktı yok** |
| 0.8 | Eski `.git` korunmuş | `ls -d .git.broken` | var |
| 0.9 | Kopya firmware kaldırılmış | `ls esp32.c` | yok |
| 0.10 | İzlenen dosya sayısı makul | `git ls-files \| wc -l` | 40-120 |

**Faz 0 bitti sayılır** ancak 0.1-0.7'nin tamamı geçerse. 0.8-0.10 hijyen
maddeleri, atlanırsa teknik borç olarak kaydedilir.

---

## Faz 1 — Paket yapısı

| # | Kriter | Doğrulama | Eşik |
| --- | --- | --- | --- |
| 1.1 | Paket import edilebilir | `uv run python -c "import vehicle_speed"` | exit 0 |
| 1.2 | Sürüm bildiriliyor | `... print(vehicle_speed.__version__)` | `0.1.0` |
| 1.3 | Pinhole formülü doğru | `estimate_depth(1200.0, 0.52, 100.0)` | `6.24` |
| 1.4 | `uv.lock` commit edilmiş | `git ls-files uv.lock` | listelenir |
| 1.5 | torch CPU sürümü | `python -c "import torch; print(torch.version.cuda)"` | `None` |
| 1.6 | Kopya kod yok | `grep -rc "def estimate_depth" src/ \| grep -v ':0' \| wc -l` | 1 |
| 1.7 | `imshow` tek dosyada | `grep -rl "cv2.imshow" src/ \| wc -l` | 1 |
| 1.8 | Modül seviyesi torch importu yok | `grep -rn "^import torch\|^from torch" src/` | çıktı yok |
| 1.9 | Gömülü `hfov=85.0` kalmadı | `grep -rn "85\.0" src/` | çıktı yok |
| 1.10 | Eski script'ler kaldırıldı | `ls kadir.py 1_donem_kadir.py listen.py` | hiçbiri yok |
| 1.11 | `py.typed` var | `ls src/vehicle_speed/py.typed` | var |

**1.9 hakkında not:** `85.0` değeri yok olmuyor, `config.py` içine
**kaynağı belirtilerek** taşınıyor. Kriter, değerin koda gömülü kalmamasını
denetliyor. Taşındığı yerde şu yorum olmalı:

```python
# Ölçülmemiş varsayılan. 1_donem_kadir.py içinde "placeholder" olarak işaretliydi.
# Gerçek kamera datasheet HFOV değeri ile değiştirilmeli. Bkz. issue #NN.
default_hfov_rad: float = math.radians(85.0)
```

Bu dürüst not, makalede hakemin soracağı sorunun cevabıdır.

---

## Faz 2 — Kod kalitesi

| # | Kriter | Doğrulama | Eşik |
| --- | --- | --- | --- |
| 2.1 | Lint temiz | `uv run ruff check . ; echo $?` | 0 |
| 2.2 | Biçim temiz | `uv run ruff format --check . ; echo $?` | 0 |
| 2.3 | Tip denetimi temiz | `uv run mypy ; echo $?` | 0 |
| 2.4 | `possibly-undefined` açık | `grep possibly-undefined pyproject.toml` | bulunur |
| 2.5 | Kancalar kurulu | `ls .git/hooks/pre-commit .git/hooks/commit-msg` | ikisi de var |
| 2.6 | Tüm kancalar geçiyor | `uv run pre-commit run --all-files ; echo $?` | 0 |
| 2.7 | Sürümler eşleşik | `grep -o 'ruff==[0-9.]*' pyproject.toml` ↔ `rev: v...` | aynı |
| 2.8 | Firmware biçimli | `clang-format --dry-run --Werror firmware/src/main.cpp ; echo $?` | 0 |
| 2.9 | Biçimlendirme ayrı commit | `git log --oneline \| grep -c "^.\{8\} style:"` | ≥ 1 |
| 2.10 | `CHANGELOG.md` var | `ls CHANGELOG.md` | var |

**2.9 neden önemli:** `02_kod_kalitesi/01_ruff_linter_formatter.md` bölüm 3'te
biçimlendirme ve davranış değişikliklerinin ayrı commit olması gerektiği
yazılı. Tek commit'te 141 lint düzeltmesi + kod taşıması olursa inceleme
imkânsız hale gelir.

---

## Faz 3 — Testler

| # | Kriter | Doğrulama | Eşik |
| --- | --- | --- | --- |
| 3.1 | Testler geçiyor | `uv run pytest tests/unit -q ; echo $?` | 0 |
| 3.2 | Test sayısı yeterli | `uv run pytest --collect-only -q \| tail -1` | ≥ 40 test |
| 3.3 | Kapsam eşiği | `uv run pytest --cov=src ... \| grep TOTAL` | ≥ %85 |
| 3.4 | Donanım testleri CI dışında | `uv run pytest --collect-only -q \| grep -c hardware` | 0 |
| 3.5 | Z² testi var | `uv run pytest -k quadratic -v` | 1 test geçti |
| 3.6 | A1 regresyonu eski kodu yakalıyor | `git stash && pytest -q test_detector.py; git stash pop` | **düşmeli** |
| 3.7 | A2 regresyonu var | `uv run pytest -k calibration_fails -v` | geçti |
| 3.8 | Telemetri kısmi satır testi | `uv run pytest -k split -v` | geçti |
| 3.9 | Firmware sözleşme testi | `uv run pytest -k firmware_contract -v` | geçti |
| 3.10 | Fixture gerçek veya açıkça sentetik | `head -1 tests/fixtures/telemetry_sample.jsonl` | geçerli JSON |
| 3.11 | Test süresi makul | `uv run pytest tests/unit --durations=5` | toplam < 10 sn |

**3.6 en kritik kriter.** Yeni yazılan bir test, düzelttiği hatanın eski
haliyle de geçiyorsa hiçbir şeyi korumuyor. Bu doğrulama yapılmadan "regresyon
testi eklendi" denemez.

**3.10 dürüstlük kriteri.** Fixture sentetik ise dosyanın başında yorumla
belirtilmeli:

```jsonl
# SENTETİK — firmware/src/main.cpp içindeki Serial.print biçiminden türetilmiştir.
# Gerçek cihaz kaydı değildir. Donanım erişimi sağlandığında değiştirilecek (issue #NN).
{"event": "startup", "message": "5 dakikalik test basladi", "max_speed": 1000.00}
```

JSONL yorum satırı desteklemez; bu not ayrı bir `README.md` dosyasında
(`tests/fixtures/README.md`) tutulmalı. Mevcut `serial.log` 23 bayt olduğu için
gerçek fixture **henüz üretilemedi** — bu bir açık ve gizlenmemeli.

---

## Faz 4 — CI/CD

| # | Kriter | Doğrulama | Eşik |
| --- | --- | --- | --- |
| 4.1 | Workflow geçerli | `check-jsonschema --check-github-workflows .github/workflows/*.yml` | exit 0 |
| 4.2 | Koşu başarılı | `gh run list --workflow ci.yml --json conclusion --jq '.[0].conclusion'` | `success` |
| 4.3 | Tüm job'lar yeşil | `gh api .../jobs --jq '.jobs[].conclusion' \| sort -u` | yalnızca `success`/`skipped` |
| 4.4 | Önbellek çalışıyor | ikinci koşu süresi / ilk koşu süresi | < %60 |
| 4.5 | Kapsam eşiği CI'da zorlanıyor | bilinçli bozuk PR aç → CI kırmızı olmalı | doğrulandı |
| 4.6 | Lint kapısı CI'da zorlanıyor | bilinçli lint hatası → CI kırmızı | doğrulandı |
| 4.7 | `permissions` kısıtlı | `grep -A2 "^permissions:" .github/workflows/ci.yml` | `contents: read` |
| 4.8 | `timeout-minutes` her job'da | `grep -c "timeout-minutes" ci.yml` | = job sayısı |
| 4.9 | Action SHA sabitlendi (3. taraf) | `grep "uses:" ci.yml` | `@vN` boşluksuz |
| 4.10 | `concurrency` `main`'de iptal etmiyor | `grep cancel-in-progress ci.yml` | koşullu ifade |
| 4.11 | Firmware job'ı paths filtreli | `grep -A3 "paths:" firmware.yml` | `firmware/**` |

**4.5 ve 4.6 negatif testler.** Kapının gerçekten kapı olduğunu kanıtlamanın
tek yolu, bilerek ihlal edip CI'ın kırmızıya döndüğünü görmek. Bu yapılmazsa
"CI yeşil" yalnızca "hiçbir şey denetlenmiyor" anlamına da gelebilir.

Negatif test nasıl yapılır:

```bash
git checkout -b test/verify-ci-gates
echo "import os" >> src/vehicle_speed/geometry.py    # F401 üretir
git commit -am "test: verify lint gate (revert me)"
git push -u origin test/verify-ci-gates
gh pr create --fill --draft
gh run watch                                         # KIRMIZI olmalı
gh pr close --delete-branch
```

Beklenen: `lint` job'ı `F401` ile düşer. Yeşil kalırsa kapı çalışmıyor.

---

## Faz 5 — Yönetişim

| # | Kriter | Doğrulama | Eşik |
| --- | --- | --- | --- |
| 5.1 | Lisans dosyası var | `ls LICENSE` + `head -1 LICENSE` | AGPL metni |
| 5.2 | Lisans tutarlı | `CITATION.cff` ↔ `LICENSE` ↔ `pyproject.toml` | üçü de AGPL |
| 5.3 | Etiketler yüklü | `gh label list --json name --jq 'length'` | 38 |
| 5.4 | Şablonlar render ediliyor | web arayüzü `/issues/new/choose` | 3 şablon **elle** |
| 5.5 | PR şablonu geliyor | boş PR aç → şablon içeriği görünmeli | **elle** |
| 5.6 | Branch protection aktif | `gh api .../branches/main/protection` | `contexts: ["All checks"]` |
| 5.7 | Admin de tabi | aynı API | `enforce_admins.enabled: true` |
| 5.8 | Force push engelli | `gh api .../protection --jq '.allow_force_pushes.enabled'` | `false` |
| 5.9 | Issue'lar açıldı | `gh issue list --json number --jq 'length'` | ≥ 13 |
| 5.10 | Ölü bağlantı yok | `01_adim_adim_uygulama.md` Faz 5 doğrulama betiği | çıktı yok |
| 5.11 | README durum tablosu dürüst | elle okuma | Kalman/NRC `planlanıyor` |
| 5.12 | CODEOWNERS var | `ls .github/CODEOWNERS` | var |
| 5.13 | Dependabot aktif | `gh api .../dependabot/repositories` veya Settings | etkin |

**5.4 ve 5.5 `gh` ile doğrulanamaz.** YAML geçerli olsa bile GitHub Issue Forms
şemasına uymayan bir dosya sessizce render edilmez. Bu iki kriter **elle** web
arayüzünden kontrol edilmeli ve sonuç buraya yazılmalı.

**5.11 kritik bir dürüstlük kriteri.** README'de Kalman filtresi ve NRC
"çalışıyor" diye yazılırsa, depoyu kullanan biri bu bileşenleri arar ve
bulamaz. `00_durum_analizi/01_mevcut_envanter.md` içinde belgelendiği gibi
ikisi de bugün **kod tabanında yok**.

---

## Faz 6 — Araştırma bileşenleri

| # | Kriter | Doğrulama | Eşik |
| --- | --- | --- | --- |
| 6.1 | Kalman yakınsıyor | `pytest -k kalman_converges -v` | geçti |
| 6.2 | Kovaryans tekilleşmiyor | `pytest -k covariance -v` | geçti |
| 6.3 | NRC tensör sözleşmesi | `[B,5] → [B,1]` testi | geçti |
| 6.4 | NRC parametre sayısı | `sum(p.numel() for p in model.parameters())` | ~850 |
| 6.5 | NRC CPU gecikmesi | 1000 çıkarım ölçümü | < 100 ms toplam |
| 6.6 | NaN üretmiyor | `torch.isnan(out).any()` | `False` |
| 6.7 | Artık-hata mimarisi | sıfır ağırlıkla `v_final == v_raw` | geçti |
| 6.8 | Yer gerçeği senkron | eşleme hatası ölçümü | belgelendi |
| 6.9 | Ablasyon 5 varyant | `baseline`, `+w_px`, `+y_norm`, `+AR`, `+conf` | 5 sonuç |
| 6.10 | İstatistiksel test | Wilcoxon | `p < 0.001` |
| 6.11 | SOTA tablosu | 5 yöntem × 3 metrik | dolu |
| 6.12 | Tekrarlanabilirlik paketi | Zenodo DOI | üretildi |
| 6.13 | `weights_only=True` | `grep -n "torch.load" src/` | hepsi bayraklı |

**6.7 hakkında:** NRC çökse bile sistemin Kalman hızına düşmesi gerekiyor
(fail-safe). Test: ağ ağırlıkları sıfırlanınca `Δv = 0`, dolayısıyla
`v_final = v_raw`. Bu test geçmiyorsa artık-hata mimarisi doğru kurulmamış.

**6.8 ölçülemeyebilir.** `01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md`
bölüm 5'te belirtildiği gibi iki akış arasında ortak saat yok. Senkronizasyon
çözümü bulunmadan 6.9-6.11 anlamlı sonuç üretmez. Bu, Faz 6'nın **gerçek
engelidir** ve kod yazarak çözülmez.

---

## Genel "bitti" tanımı (Definition of Done)

Her iş kalemi için, türünden bağımsız:

1. **Kod yazıldı** ve `ruff` + `mypy` temiz.
2. **Test eklendi** ve test, değişikliğin eski halini yakalıyor (negatif doğrulama).
3. **CI yeşil** — yerel değil, GitHub üzerinde.
4. **PR şablonu dolu** — doğrulama bölümünde komut çıktısı var.
5. **Akademik etki değerlendirildi** — `vision`/`calibration`/`tracking`/`nrc`/
   `telemetry` değiştiyse tablo dolduruldu.
6. **Doküman güncellendi** — davranış değiştiyse `README.md` veya `docs/`.
7. **Issue kapandı** — `Closes #NN` ile otomatik.

Bu yedi maddeden biri eksikse iş bitmemiştir.

---

## Bilinçli olarak kapsam dışı bırakılanlar

Bunlar "yapılmadı" değil, "bilinçli olarak yapılmayacak":

| Konu | Gerekçe | Nerede kayıtlı |
| --- | --- | --- |
| İmzalı commit zorunluluğu | tek kişilik akademik projede GPG yönetimi maliyeti faydayı aşıyor | `04_branch_koruma_inceleme.md` §7 |
| PyPI yayını | `pip install` ile kuracak somut kullanıcı yok | `04_release_versiyonlama.md` §9 |
| macOS CI | somut ihtiyaç yok, runner çarpanı 10× | `03_test_matrisi_cache.md` §1.2 |
| GPU testi | GitHub ücretsiz runner'da GPU yok | `03_test_matrisi_cache.md` §9 |
| `pydocstyle` (D kuralları) | 141 hatanın üzerine yüzlerce ekler; kod taşındıktan sonra değerlendirilecek | `01_ruff_linter_formatter.md` §2 |
| `visualization.py` / `cli.py` kapsamı | GUI CI'da çalışmaz; mock karmaşıklığı faydayı aşıyor | `04_test_mimarisi/01_test_piramidi.md` §8 |
| Git LFS | 1 GB kota, `best.pt` her yeniden eğitmede kota tüketir | `06_veri_model_yonetimi/...` §3 |
| `reuse lint` kapısı | PDF/görsel/fixture için gürültü üretir | `05_lisans_citation.md` §3 |

Bu tablo, ileride "neden X yok?" sorusuna cevap verir. Kapsam dışı bırakılan
bir konu yeniden gündeme gelirse, buraya gerekçe eklenerek karar güncellenir.

---

## Doğrulanmamış iddialar kaydı

Bu dizindeki raporlarda **kanıtlanmamış** ve uygulama sırasında teyit edilmesi
gereken iddialar:

| İddia | Nerede | Nasıl doğrulanır |
| --- | --- | --- |
| Firmware bugün de derleniyor | `04_firmware_c_statik_analiz.md` §8 | `cd firmware && pio run` (platformio kurulu değil) |
| `license = "AGPL-3.0-or-later"` (PEP 639) hatchling'de geçerli | `05_lisans_citation.md` §2 | `uv build` çıktısındaki uyarılar |
| OpenCV 5.x Python 3.10'u destekliyor | `03_test_matrisi_cache.md` §9 | probe workflow (§1.3) |
| Python 3.13'te torch tekerleği var | `01_hedef_monorepo.md` §3 | `uv run --python 3.13 --with torch ...` |
| AGPL + GPL-3.0-only birleşimi bu kullanımda sorunlu | `05_lisans_citation.md` §8 | hukuki danışmanlık |
| `top.mp4` ve `top2.mp4` kopya | `06_veri_model_yonetimi/...` §1 | `sha256sum` |
| `best.pt`/`detector.pt` eğitim veri seti | `05_lisans_citation.md` §5.2 | `Yapay zeka/` altındaki script'lerin incelenmesi |
| İkinci koşu %60 daha hızlı (önbellek) | `03_ci_cd/03_test_matrisi_cache.md` §8 | iki ardışık CI koşusu ölçümü |
| CI soğuk önbellekle ~3 dk | `03_ci_cd/01_pipeline_mimarisi.md` §8 | ilk koşu ölçümü |
| NRC ~850 parametre, 0.05 ms gecikme | `07_uygulama_yol_haritasi/...` Faz 6 | model kurulup ölçüldüğünde |

Bu tablo bilinçli olarak eklendi. Bir raporda "doğrulandı" yazmayan her sayısal
iddia ya ölçülmüş ya da tahmindir; tahmin olanlar burada listelenir. Yapıştırılan
50 adımlı plandaki "UNVALIDATED COMPONENTS: Yok" ve "REMAINING RISKS: Sıfır"
beyanlarının tersine, bu dizin doğrulanmamış kısımlarını açıkça söyler.

---

## Faz durum özeti (uygulama başlamadı)

| Faz | Adım sayısı | Durum | Kanıt |
| --- | --- | --- | --- |
| 0 — Depo hijyeni | 14 | başlamadı | `git status` → exit 128 |
| 1 — Paket yapısı | 15 | başlamadı | `ls src/` → yok |
| 2 — Kod kalitesi | 15 | başlamadı | `ruff check` → 141 hata |
| 3 — Testler | 16 | başlamadı | `ls tests/` → yok |
| 4 — CI/CD | 14 | başlamadı | `ls .github/` → yok |
| 5 — Yönetişim | 15 | başlamadı | `ls LICENSE` → yok |
| 6 — Araştırma | 17 | başlamadı | `ls src/vehicle_speed/nrc/` → yok |
| **Toplam** | **106** | **0 tamamlandı** | |

Bu tablo, uygulama ilerledikçe güncellenmeli ve her satırın "Kanıt" sütununa
gerçek komut çıktısı yazılmalı.
