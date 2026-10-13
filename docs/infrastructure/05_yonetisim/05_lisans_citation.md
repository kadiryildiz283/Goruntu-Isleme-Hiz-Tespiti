# Lisans, Atıf ve Bağımlılık Uyumu

**Durum:** Depoda `LICENSE` dosyası `YOK`, `CITATION.cff` `YOK`.
Bu rapor lisans seçimini ve bağımlılık uyum risklerini belgeler.

---

## 1. Bağımlılık lisans envanteri

Lisans seçimi, kullanılan kütüphanelerin lisanslarına bağlı. Bu proje için
belirleyici olan tek bir bağımlılık var.

| Paket | Lisans | Etki |
| --- | --- | --- |
| `ultralytics` (YOLOv8) | **AGPL-3.0** | **belirleyici** |
| `torch` | BSD-3-Clause | serbest |
| `opencv-python-headless` | Apache-2.0 | serbest |
| `numpy` | BSD-3-Clause | serbest |
| `scipy` | BSD-3-Clause | serbest |
| `pyserial` | BSD-3-Clause | serbest |
| `AccelStepper` (firmware) | GPL-2.0+ | firmware ayrı dağıtım |
| `ArduinoJson` (önerilen) | MIT | serbest |
| `ruff`, `mypy`, `pytest` | MIT / PSF | geliştirme aracı, dağıtılmaz |

### AGPL-3.0 neden belirleyici

AGPL'in farkı "ağ kullanımı" maddesidir: yazılımı bir ağ servisi üzerinden
kullandırsanız bile kaynak kodu sunmak zorundasınız. Normal GPL yalnızca
dağıtımda tetiklenir.

Bu proje için pratik sonuç:

1. `ultralytics`'i import eden bir paketi **MIT veya Apache-2.0 ile
   lisanslayamazsın**. Ultralytics'in kendi lisans sayfası bu konuda açık:
   AGPL koşullarına uymak ya da ticari lisans satın almak gerekiyor.
2. Depo **public** olursa AGPL'in kaynak sunma koşulu zaten karşılanmış olur.
   Yani açık kaynak akademik proje için AGPL uyumu fiilen sorun değil.
3. Ama **kapalı bir türev** (ör. bir şirkete devredilmiş sürüm) üretilemez.

### Üç seçenek

| Seçenek | Lisans | Sonuç |
| --- | --- | --- |
| A | **AGPL-3.0** | ultralytics ile tam uyumlu; ticari kullanım kapalı kalır |
| B | **Apache-2.0** + ultralytics bağımlılığı | lisans çelişkisi; dağıtım hakkı belirsiz |
| C | **Apache-2.0** + ultralytics'i opsiyonel yap | çekirdek MIT/Apache, YOLO entegrasyonu ayrı eklenti |

**Öneri: A (AGPL-3.0).**

Gerekçe:

- Proje zaten açık kaynak olacak; AGPL'in getirdiği ek yükümlülük (ağ
  kullanımı durumunda kaynak sunma) akademik bir depoda zaten karşılanıyor.
- B seçeneği gerçek bir hukuki belirsizlik yaratır. Akademik makale için
  sorun olmayabilir ama depoyu bir başkası kullanmak istediğinde çelişki
  ortaya çıkar.
- C seçeneği mimari olarak temiz ama bugün için gereksiz iş yükü: tüm proje
  YOLO üzerine kurulu, çekirdeği ondan ayırmak anlamlı bir kullanım
  senaryosu üretmiyor.

C seçeneği ileride değerlendirilebilir —örneğin geometri ve telemetri
modüllerinin (`geometry.py`, `tracking.py`, `telemetry/*`) ultralytics'e
hiç bağımlı olmaması sayesinde bu modüller ayrı bir Apache-2.0 paket olarak
çıkarılabilir. `01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md`
bölüm 4'teki import sınırları tam olarak bu ayrımı mümkün kılıyor. Bu,
planlanmış bir sonuç değil, mimarinin yan faydası.

---

## 2. `LICENSE` dosyası

AGPL-3.0 tam metni:

```bash
curl -o LICENSE https://www.gnu.org/licenses/agpl-3.0.txt
```

Metin **hiç değiştirilmez**. Lisans metninin başına telif satırı eklemek
yaygın bir yanlış; AGPL metni olduğu gibi kalmalı, telif bildirimi ayrı
dosyalarda (`README.md`, kaynak başlıkları) yer almalı.

`pyproject.toml`:

```toml
[project]
classifiers = [
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
]
license = "AGPL-3.0-or-later"
```

`license = "AGPL-3.0-or-later"` biçimi PEP 639 (yeni SPDX string) sözdizimidir.
Eski biçim `license = { text = "AGPL-3.0-or-later" }` veya
`license = { file = "LICENSE" }`. Hangisinin geçerli olduğu `hatchling`
sürümüne bağlı; `uv build` sırasında uyarı çıkarsa eski biçime dönülmeli.
Bu, doğrulanmamış bir ayrıntı — ilk `uv build` koşusunda kontrol edilmeli.

---

## 3. Kaynak dosya başlıkları

AGPL bölüm 5, her kaynak dosyasında kısa bir bildirim önerir. Bu proje için:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 Kadir Yıldız
#
# Bu program ücretsiz bir yazılımdır: Free Software Foundation tarafından
# yayınlanan GNU Affero Genel Kamu Lisansı (sürüm 3 veya sonraki) koşulları
# altında yeniden dağıtabilir ve/veya değiştirebilirsiniz.
```

C dosyaları için aynı metin `/* */` içinde.

**Ölçülülük notu:** 20 dosyalık bir araştırma deposunda her dosyaya 8 satırlık
lisans başlığı eklemek gürültü üretir. Yaygın pratik, yalnızca bir satırlık
SPDX tanımlayıcısı kullanmak:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
```

SPDX tanımlayıcısı makine tarafından okunabilir, `reuse` aracı ile
denetlenebilir ve tam metin tekrarını gerektirmez. Öneri: **tek satır SPDX**.

Denetim:

```bash
uv tool install reuse
reuse lint
```

`reuse lint` eksik lisans başlığı ve kayıtsız dosyaları raporlar. Bu bir
kalite kapısı olarak CI'a eklenebilir ama ilk kurulumda gürültü üretir
(PDF'ler, görseller, fixture'lar da lisans bilgisi ister). Karar: Faz 5'te
değerlendirilsin, başlangıçta zorunlu olmasın.

---

## 4. `CITATION.cff`

`03_ci_cd/04_release_versiyonlama.md` bölüm 4'te tam metin verildi. Burada
yalnızca lisans/atıf kesişimi:

```yaml
license: AGPL-3.0-or-later
```

`CITATION.cff` içindeki `license` alanı SPDX tanımlayıcısı alır ve
`LICENSE` dosyasıyla tutarlı olmalı. Ayrışma olursa GitHub'ın "Cite this
repository" paneli yanlış lisans gösterir.

Denetim (release workflow'una eklenebilir):

```bash
python3 - <<'PY'
import pathlib, re, sys
lic = pathlib.Path("LICENSE").read_text(errors="replace")
cff = pathlib.Path("CITATION.cff").read_text()
cff_lic = re.search(r"^license:\s*(\S+)", cff, re.M)
if not cff_lic:
    sys.exit("CITATION.cff icinde license alani yok")
is_agpl = "GNU AFFERO GENERAL PUBLIC LICENSE" in lic.upper()
if is_agpl and not cff_lic.group(1).startswith("AGPL"):
    sys.exit(f"LICENSE AGPL ama CITATION.cff {cff_lic.group(1)} diyor")
print("lisans tutarli:", cff_lic.group(1))
PY
```

---

## 5. Üçüncü taraf kod ve veri atıfları

### 5.1 YOLOv8 ağırlıkları

`yolov8n.pt` ve `yolov8s.pt` ultralytics tarafından eğitilmiş ve COCO veri
seti üzerinde doğrulanmış. Bu ağırlıklar depoya girmez
(`06_veri_model_yonetimi/`), ama indirildiklerinde atıf gerekir. `README.md`
içinde:

```markdown
## Üçüncü taraf bileşenler

Bu proje aşağıdaki bileşenleri kullanır:

- **YOLOv8** — Ultralytics, AGPL-3.0. https://github.com/ultralytics/ultralytics
  COCO-pretrained ağırlıklar (`yolov8n.pt`, `yolov8s.pt`) ilk kullanımda
  otomatik indirilir.
- **COCO veri seti** — Lin vd., 2014. https://cocodataset.org/#acknowledgements
- **AccelStepper** — Mike McCauley, GPL-2.0+. Firmware'de kullanılır.
```

COCO atfı kolayca unutulur. `TARGET_CLASS_ID = 32` (sports ball) doğrudan
COCO sınıf listesinden geliyor; yani COCO veri seti bu projenin bir
bağımlılığı, yalnızca ultralytics değil.

### 5.2 Kendi eğitilmiş modeller

`best.pt` (53 MB, 14 anahtar noktalı CarFusion poz modeli) ve `detector.pt`
(6.2 MB, plaka dedektörü) — bunlar bu projede eğitilmiş. Kaynakları:

- Eğitim veri seti nereden geldi? `Yapay zeka/` dizinindeki dosyalar arasında
  `coco.names` ve `coco.txt` var, bu COCO türevi bir eğitim olabileceğini
  düşündürüyor.
- Eğitim kodu depoda yok. `Yapay zeka/` altındaki `deneme.py`, `main.py`,
  `hocam.py` gibi dosyalar eğitim script'i olabilir ama doğrulanmadı.

**Bu bir tekrarlanabilirlik boşluğu.** Makalede "kendi eğittiğimiz plaka
dedektörü" deniyorsa, eğitim veri seti ve eğitim kodu da erişilebilir olmalı.
Aksi halde hakem "modeli yeniden eğitemiyorum" diyebilir ve bu Q2 düzeyinde
reddedilme sebebi olabilir.

Karar: `06_veri_model_yonetimi/01_gitignore_lfs_veri_izlenebilirligi.md`
bölüm 5'te ele alındı — eğitim veri seti Zenodo'ya, eğitim kodu
`src/vehicle_speed/nrc/train.py` ve `scripts/` altına alınmalı.

---

## 6. Firma/üniversite sahipliği

Yüksek lisans tezi kapsamında üretilen kodun fikri mülkiyet sahipliği
üniversite politikasına bağlı olabilir. Depoyu public yapmadan **önce**
kontrol edilmeli:

1. Üniversitenin fikri mülkiyet politikası yüksek lisans öğrencisi kodunu
   kapsıyor mu?
2. Danışman (Ersan Hoca) ortak yazar mı, kod sahibi mi?
3. Bir TÜBİTAK/BAP projesi kapsamında finanse edildiyse, açık kaynak
   yayınlama koşulu veya kısıtı var mı?

Bu soruların cevabı `LICENSE` dosyasındaki telif satırını doğrudan etkiler:

```text
Copyright (C) 2026 Kadir Yıldız
```

yerine

```text
Copyright (C) 2026 Kadir Yıldız ve <Üniversite Adı>
```

gerekebilir.

**Bu rapor hukuki tavsiye değildir.** Depo public yapılmadan önce üniversitenin
teknoloji transfer ofisi veya danışman ile görüşülmeli. Geri alınamaz bir
durum: public bir depoya push edilen kod, silinse bile fork'lar ve arşivler
üzerinden kalıcı hale gelir.

---

## 7. `.gitattributes` ve lisans denetimi

```text
# normalleştirme
* text=auto eol=lf

# ikili dosyalar (diff üretmesin)
*.pt    binary
*.onnx  binary
*.png   binary
*.jpg   binary
*.jpeg  binary
*.mp4   binary
*.mkv   binary
*.pdf   binary
*.bin   binary
*.elf   binary

# LFS (yalnızca gerçekten depoya girecek ikili dosyalar için)
*.pt    filter=lfs diff=lfs merge=lfs -text
*.onnx  filter=lfs diff=lfs merge=lfs -text
```

**Dikkat:** `*.pt` için LFS kuralı yazılıp dosyalar `.gitignore`'da olursa
çelişki oluşmaz (ignore öncelikli), ama ikisi birden yazıldığında niyet
belirsizleşir. `06_veri_model_yonetimi/01_gitignore_lfs_veri_izlenebilirligi.md`
bölüm 3'te karar netleştirildi: **`.pt` dosyaları LFS'e değil, GitHub
Release'e** konur. Dolayısıyla `.gitattributes` içindeki LFS satırları
**kullanılmamalı** — yalnızca `binary` işaretlemesi kalmalı.

`eol=lf` zorunlu: `03_ci_cd/02_ci_workflow_adimlari.md` bölüm 5'te
belirtildiği gibi Windows matrisi var. `text=auto` olmadan Windows'ta
checkout edilen dosyalar `CRLF` olur, Linux'ta diff gürültüsü üretir ve
`pre-commit`'teki `mixed-line-ending` kancası her commit'te dosyaları
değiştirir.

---

## 8. Bağımlılık lisans denetiminin otomasyonu

```yaml
  license-check:
    name: license-check
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - name: Scan dependency licenses
        run: |
          set -euo pipefail
          uv pip install pip-licenses
          uv run pip-licenses --format=json --with-urls > licenses.json
          python3 - <<'PY'
          import json, pathlib, sys
          ALLOWED = {"MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0",
                     "PSF-2.0", "Python-2.0", "ISC", "MPL-2.0", "AGPL-3.0",
                     "AGPL-3.0-or-later", "LGPL-2.1-or-later", "Unlicense",
                     "Zlib", "unknown"}
          FORBIDDEN = {"GPL-3.0-only", "SSPL-1.0", "EUPL-1.2"}
          rows = json.loads(pathlib.Path("licenses.json").read_text())
          bad = [r for r in rows
                 if any(f in str(r["License"]) for f in FORBIDDEN)]
          unlisted = [r for r in rows
                      if not any(a in str(r["License"]) for a in ALLOWED | FORBIDDEN)]
          for r in unlisted:
              print(f"INCELENMELI: {r['Name']} {r['Version']} -> {r['License']}")
          if bad:
              for r in bad:
                  print(f"YASAKLI: {r['Name']} {r['License']}")
              sys.exit(1)
          print(f"{len(rows)} paket tarandı, yasaklı lisans yok.")
          PY
      - uses: actions/upload-artifact@v4
        with:
          name: dependency-licenses
          path: licenses.json
```

`unknown` izin verilenler listesinde — çünkü `pip-licenses` bazı paketlerin
lisansını çözemiyor ve bu bir hata değil, inceleme konusu. `unlisted` çıktısı
log'da görünür, job'ı kırmaz. Yalnızca açıkça yasaklı lisanslar
(`GPL-3.0-only`, `SSPL-1.0`) job'ı düşürür.

**AGPL'in kendi listesinde olması** başta tuhaf görünebilir: proje zaten
AGPL, dolayısıyla AGPL bağımlılığı uyumlu. Ama `GPL-3.0-only` yasaklı, çünkü
AGPL-3.0 ile GPL-3.0-only birleştirilebilir olsa da bu proje "or later"
kullandığı için sürüm uyumu ayrıca denetlenmeli. Bu ince bir hukuki ayrım ve
**doğrulanmamış**; bir avukata veya üniversite TTO'ya danışılmalı
(bölüm 6).

---

## 9. Açık noktalar

1. **Hukuki doğrulama yapılmadı.** Bu rapor mühendislik perspektifinden
   lisans uyumunu belgeler, hukuki tavsiye değildir. `ultralytics` AGPL
   koşullarının bu kullanım biçimine nasıl uygulandığı bir avukatla
   netleştirilmeli.
2. **Üniversite fikri mülkiyet politikası araştırılmadı.** Depo public
   yapılmadan önce zorunlu kontrol.
3. **Eğitim veri seti kaynağı belirsiz.** `best.pt` ve `detector.pt`'nin nasıl
   eğitildiği depoda belgelenmemiş. Makale tekrarlanabilirliği açısından
   kapanması gereken bir boşluk.
4. **`AccelStepper` GPL-2.0+.** Firmware ayrı bir derleme ürünü; GPL
   bulaşıcılığı firmware binary'sine uygulanır. Firmware `.bin` dosyası
   GitHub Release'te dağıtılacaksa (`03_ci_cd/04_release_versiyonlama.md`
   bölüm 2), GPL kaynak sunma koşulu geçerli — firmware kaynak kodu zaten
   depoda public olduğu için koşul karşılanıyor.
