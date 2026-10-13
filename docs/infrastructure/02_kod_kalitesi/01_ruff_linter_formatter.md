# Ruff: Linter ve Formatter Yapılandırması

**Araç sürümü (doğrulandı):** `ruff 0.16.7`
**Taban çizgisi (doğrulandı):** 141 hata, 115'i otomatik düzeltilebilir

Bu rapor, yapıştırılan plandaki Ruff yapılandırmasının neden çalışmadığını
kanıtlarıyla gösterir ve yerine geçen, bu makinede gerçekten çalıştırılıp
doğrulanmış yapılandımayı verir.

---

## 1. Plandaki yapılandırma neden çöküyor

Yapıştırılan planın verdiği blok:

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "C4", "UP", "SIM", "TCH", "RUFF"]
```

Gerçek çalıştırma:

```console
$ ruff check .
ruff failed
  Cause: Unknown rule selector `RUFF` in `select` from `/tmp/rufftest/pyproject.toml`
```

`RUFF` diye bir seçici yok. Ruff'un kendi kurallarının ön eki **`RUF`**
(ör. `RUF001`, `RUF100`). Tek harf farkı, ama sonuç yapılandırmanın tamamının
reddedilmesi — yani linter hiç çalışmıyor.

### Diğer üç sorun

**1.1 `TCH` → `TC`.** `flake8-type-checking` kuralları yeniden adlandırıldı.
Doğrulama:

```console
$ ruff rule TC003
# typing-only-standard-library-import (TC003)
Derived from the **flake8-type-checking** linter.
```

`TCH` hâlâ geriye dönük takma ad olarak kabul ediliyor (`ruff check --select TCH`
hata vermeden çalıştı), ama yeni yapılandırmada `TC` yazılmalı. Takma adlar
ileride kaldırılabilir.

**1.2 Ölü `per-file-ignores`.** Plan şu istisnaları tanımlıyordu:

```toml
"tests/*" = ["S101", "ARG"]
"src/vehicle_speed/geometry.py" = ["N803", "N806"]
```

Ama `select` listesinde ne `S` (flake8-bandit) ne `N` (pep8-naming) ne de `ARG`
(flake8-unused-arguments) var. Seçilmemiş bir kuralın istisnası tanımlanamaz;
bu satırlar hiçbir etki üretmez.

Bu somut olarak önemli: planın gerekçesi "fizik sembolleri $W, H, Z$ büyük harfle
kalabilsin" idi. Ama `N` seçilmediği için `N806` zaten hiç tetiklenmiyordu —
yani istisna yazmaya gerek yoktu, gerekçe de boşa düşmüştü. Gerçek kod tabanında
`N806` yalnızca `N` seçildiğinde ortaya çıkıyor ve 2 adet:

```console
$ ruff check --select N --statistics 1_donem_kadir.py
2      N806    [ ] non-lowercase-variable-in-function
```

Bunlar `_get_fallback_headlight_width` içindeki `LEFT_HEADLIGHT_IDX` ve
`RIGHT_HEADLIGHT_IDX`. Doğru çözüm istisna yazmak değil, bu iki sabiti modül
seviyesine taşımak — zaten sabitler, fonksiyon içinde tanımlanmaları yanlış.

**1.3 `magic-trailing-comma` yanlış bölümde.** Plan `[tool.ruff]` altına
yazıyordu; bu bir `[tool.ruff.format]` seçeneğidir.

---

## 2. Doğrulanmış yapılandırma

Aşağıdaki blok bu makinede `ruff 0.16.7` ile çalıştırıldı ve geçerli bulundu.
`pyproject.toml` içine yazılacak hali:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests", "scripts"]
extend-exclude = ["archive", "firmware/.pio", "docs"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle hataları
    "W",    # pycodestyle uyarıları
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "TC",   # flake8-type-checking (eski ad: TCH)
    "ARG",  # flake8-unused-arguments
    "PTH",  # flake8-use-pathlib
    "S",    # flake8-bandit (güvenlik)
    "RUF",  # ruff'a özgü kurallar
]
ignore = [
    "B008",   # fonksiyon varsayılanında çağrı — CLI argparse kalıbı için gerekli
    "S101",   # assert — yalnızca testlerde serbest, aşağıda per-file ile
    "PTH123", # open() — mevcut listen.py refactor edilene kadar geçici
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "ARG001", "ARG002", "N802"]
"scripts/**/*.py" = ["S"]
"src/vehicle_speed/cli.py" = ["B008"]

[tool.ruff.lint.isort]
known-first-party = ["vehicle_speed"]
combine-as-imports = true

[tool.ruff.lint.flake8-tidy-imports]
ban-relative-imports = "all"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
magic-trailing-comma = true
line-ending = "auto"
docstring-code-format = true
```

### Seçilmeyen kural aileleri ve gerekçesi

| Aile | Neden seçilmedi |
| --- | --- |
| `D` (pydocstyle) | Bu aşamada her fonksiyona docstring zorunluluğu, 141 hatanın üzerine yüzlerce hata daha ekler. Kod taşındıktan sonra Faz 5'te değerlendirilmeli. |
| `ANN` (flake8-annotations) | `mypy --strict` zaten tüm imzaları tip zorunluluğuna tabi tutuyor; `ANN` gereksiz tekrar. |
| `COM` (flake8-commas) | `magic-trailing-comma = true` biçimlendiricide aynı işi görüyor. |
| `ERA` (eradicate) | Yorum satırı içindeki eski kodları silmek, akademik notları yok edebilir. Otomatik uygulanmamalı. |
| `TRY`/`EM` | İyi kurallar ama mevcut `except Exception` desenlerinin tamamını yeniden yazmayı gerektirir. `A1` hatası düzeltildikten sonra eklenmeli. |
| `FBT` (boolean-trap) | Mevcut imzalarda boolean parametre yok, şu an etkisiz. |

---

## 3. Taban çizgisini sıfırlama sırası

141 hatayı tek komutla temizlemeye çalışmak, büyük ve incelenemez bir diff
üretir. Sıra önemli:

### Adım 1 — biçimlendirme (gürültüyü at)

```bash
ruff format src/ tests/ scripts/
ruff check --select W293,W291,I001 --fix src/ tests/ scripts/
```

Beklenen: 84 `W293` + 4 `W291` + 3 `I001` = **91 hata** temizlenir.
Kalan: 50.

Bu adımın çıktısı neredeyse tamamen boşluk karakteri değişikliği olduğu için
kod incelemesinde `?w=1` (GitHub'da "Hide whitespace") ile atlanabilir. Bu
yüzden **ayrı bir commit** olmalı:

```text
style: normalize whitespace and import order

ruff format + ruff check --fix applied. No behavior change.
91 of 141 baseline findings resolved (W293: 84, W291: 4, I001: 3).
```

### Adım 2 — modern sözdizimi

```bash
ruff check --select UP --fix src/
```

Beklenen: 22 `UP006` + 1 `UP045` = 23 otomatik düzeltilir. 6 `UP035`
(`from typing import Tuple, List, ...` satırının kaldırılması) kalan `UP006`
düzeltmelerinden sonra kendiliğinden çözülür; çözülmezse import satırı elle
silinir.

Bu commit davranış değiştirmez ama tip notasyonlarını değiştirir
(`Tuple[float, float]` → `tuple[float, float]`). Python 3.9'da çalışmaz;
`requires-python = ">=3.10"` bunu zaten garanti ediyor.

### Adım 3 — satır uzunluğu

```bash
ruff format src/    # zaten sarmış olmalı
ruff check --select E501 src/
```

16 `E501`'in çoğu `ruff format` tarafından sarılır. Kalanlar genellikle uzun
`print(f"...")` satırlarıdır. Bunlar `logging` dönüşümüyle (borç `B7`) birlikte
ele alınmalı, ayrı ayrı uğraşılmamalı.

### Adım 4 — adlandırma ve pathlib

```bash
# 1_donem_kadir.py → src/vehicle_speed/detector.py taşındıktan sonra
# LEFT_HEADLIGHT_IDX / RIGHT_HEADLIGHT_IDX modül seviyesine çıkar
ruff check --select N,PTH src/
```

2 `N806` ve 2 `PTH123` burada çözülür. `PTH123` `listen.py` refactor'una
(`pyserial` geçişi) bağlı; o yüzden Adım 2'de `ignore` listesine geçici olarak
eklendi. Refactor bitince `ignore`'dan çıkarılmalı — bu bir takip issue'su
gerektirir.

### Adım 5 — kapıyı aç

```bash
ruff check .          # sıfır hata beklenir
ruff format --check . # sıfır fark beklenir
```

İkisi de temiz geçtikten sonra `pre-commit` ve CI'a eklenir.

**Sıra neden böyle:** kapı (Adım 5) en sona bırakılır. Kapı önce açılırsa, her
commit CI'ı kırar ve geliştirme durur. "Önce sıfırla, sonra kilitle" kuralı.

---

## 4. Fizik sembolleri ve adlandırma

Ruff'un `N` kuralları büyük harfli yerel değişkenleri reddeder. Fizik
formüllerinde ise $W$, $Z$, $f$ gösterimi makaleyle birebir örtüşmeli —
hakem kodu okuduğunda formülü tanıyabilmeli.

Çözüm `per-file-ignores` değil, **adlandırma sözleşmesi**:

| Fiziksel nicelik | Kod adı | Birim |
| --- | --- | --- |
| derinlik | `z_m` | metre |
| odak uzaklığı | `f_px` | piksel |
| gerçek genişlik | `real_width_m` | metre |
| piksel genişliği | `pixel_width_px` | piksel |
| görüntü genişliği | `image_width_px` | piksel |
| hız | `v_ms` | m/s |
| yatay görüş açısı | `hfov_rad` | radyan |

Bu isimlerin hepsi küçük harfle başlar, yani `N806` tetiklenmez. Aynı zamanda
birim son eki taşıdıkları için `01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md`
bölüm 1'deki birim karışıklığı borcu da kapanır. Formül yorumda gösterilir:

```python
def estimate_depth(f_px: float, real_width_m: float, pixel_width_px: float) -> float:
    """Pinhole derinlik kestirimi.

    Z = (f * W) / P

    Returns:
        Derinlik, metre.

    Raises:
        ValueError: Girdilerden biri pozitif değilse.
    """
    if f_px <= 0.0 or real_width_m <= 0.0 or pixel_width_px <= 0.0:
        msg = (
            f"positive values required, got f_px={f_px}, "
            f"real_width_m={real_width_m}, pixel_width_px={pixel_width_px}"
        )
        raise ValueError(msg)
    return (f_px * real_width_m) / pixel_width_px
```

Bu biçim `EM101`/`TRY003` kuralları ileride seçildiğinde de geçerli kalır
(mesaj `msg` değişkeninde, `raise ... from` gerekmiyor çünkü girdi doğrulaması).

---

## 5. Yerel çalıştırma

```bash
# tüm kontroller
ruff check .
ruff format --check .

# tek dosya
ruff check src/vehicle_speed/geometry.py

# otomatik düzeltme + biçimlendirme
ruff check --fix .
ruff format .

# hangi kural neden tetiklendi
ruff check --select B006 --show-fixes src/
ruff rule N806
```

`ruff` şu an sistemde CLI olarak kurulu (`ruff 0.16.7`), ama paket olarak
kurulu değil. `pyproject.toml` `[project.optional-dependencies] dev` grubuna
eklenmeli, böylece sürüm sabitlenir:

```toml
dev = [
    "ruff==0.16.7",
    ...
]
```

Sürüm sabitleme (`==`) bilinçli: ruff yeni kurallar ekliyor ve `>=` kullanılırsa
bir katkıcının makinesinde farklı sürüm farklı sonuç üretir. `pre-commit`
tarafında da aynı sürüm sabitlenmeli — iki yerdeki sürümün ayrışması, yerelde
geçen kodun CI'da kalması demek.

---

## 6. Doğrulama komutları

Bu raporun iddialarını yeniden doğrulamak için:

```bash
# 1. RUFF seçicisinin geçersiz olduğunu gör
printf '[tool.ruff.lint]\nselect = ["RUFF"]\n' > /tmp/t/pyproject.toml
cd /tmp/t && ruff check . ; echo "exit: $?"

# 2. Önerilen yapılandırmanın geçerli olduğunu gör
ruff check --select E,W,F,I,N,UP,B,C4,SIM,TC,ARG,PTH,S,RUF --statistics src/

# 3. Taban çizgisini yeniden ölç (kod taşınmadan önce)
ruff check --isolated --select E,F,W,I,B,C4,UP,SIM,N,S,ARG,PTH,RUF \
  --line-length 100 --statistics kadir.py 1_donem_kadir.py listen.py
# beklenen: 141 hata, 115 fixable
```

Üçüncü komutun çıktısı bu raporun sayısal iddialarının kaynağıdır. Kod
taşındıktan sonra sayılar değişecek; bu beklenen bir durumdur.
