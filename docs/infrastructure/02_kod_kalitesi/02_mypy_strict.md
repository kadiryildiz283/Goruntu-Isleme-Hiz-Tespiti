# Mypy: Sıkı Statik Tip Denetimi

**Araç durumu (doğrulandı):** `mypy` bu makinede **kurulu değil**. İlk adım kurulum.
**Hedef:** `src/` altında sıfır tip hatası, `PLANLANIYOR` modüller dahil.

Mypy, Ruff'un tamamlayıcısıdır: Ruff sözdizimi ve stil denetler, mypy anlam
denetler. `δv ∝ Z²·δP` gibi hata yayılımı hesaplarında birim ve tip karışıklığı
test yazmadan yakalanamaz; mypy bu sınıf hataların çoğunu derleme zamanında
yakalar.

---

## 1. Kurulum

```bash
uv pip install "mypy>=1.11"
# veya dev grubuyla birlikte
uv sync --extra dev
```

`pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "ruff==0.16.7",
    "mypy==1.11.2",
    "pre-commit>=3.7",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.14",
    "types-pyserial>=3.5",
]
```

`types-pyserial` ayrı bir stub paketi değil; `pyserial` kendi tip ipuçlarını
kısmen taşır. Doğrulama:

```bash
python -c "import serial, os; print(os.path.dirname(serial.__file__))"
ls $(python -c "import serial,os;print(os.path.dirname(serial.__file__))")/py.typed
```

`py.typed` yoksa `[[tool.mypy.overrides]]` ile `ignore_missing_imports` gerekir
(aşağıda yazılı).

---

## 2. Yapılandırma

```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
no_implicit_reexport = true
disallow_untyped_decorators = true
disallow_any_generics = true
check_untyped_defs = true
show_error_codes = true
pretty = true
files = ["src", "tests", "scripts"]
mypy_path = "src"
namespace_packages = true
explicit_package_bases = true

# Üçüncü taraf stub'ı olmayan paketler
[[tool.mypy.overrides]]
module = [
    "cv2.*",
    "ultralytics.*",
    "scipy.*",
    "torch.*",
]
ignore_missing_imports = true
follow_untyped_imports = true
```

### `strict = true` ne getirir

`strict` şu bayrakları topluca açar: `disallow_untyped_defs`,
`disallow_incomplete_defs`, `disallow_untyped_calls`, `warn_no_return`,
`strict_equality`, `no_implicit_optional`, ve diğerleri.

Mevcut kodda `disallow_untyped_defs` açısından durum iyi: `kadir.py` ve
`1_donem_kadir.py` içindeki tüm metotlar zaten tip notasyonlu
(`-> None`, `-> float`, `-> List[Dict[str, Any]]`). Yani `strict` geçişi
düşük maliyetli. Beklenen gerçek sorunlar aşağıda.

### Neden `follow_untyped_imports`

`ultralytics` ve `torch` tam tipli değil. `ignore_missing_imports = true`
tek başına bu modüllerden gelen değerleri `Any` yapar ve `Any` her yere bulaşır.
`follow_untyped_imports = true` mypy'nin bu paketlerin kaynaklarını okuyup
kısmi çıkarım yapmasını sağlar. Maliyet: mypy yavaşlar. `src/` için kabul
edilebilir.

---

## 3. Öngörülen bulgular ve çözümleri

Mevcut kod okunarak tespit edilen, `strict` altında kesin hata verecek noktalar:

### 3.1 `numpy` diziden çoklu atama

```python
x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
```

`boxes.xyxy[i].cpu().numpy()` dönüş tipi `Any` (ultralytics tipli değil).
`x1..y2` de `Any` olur. Ardından:

```python
pixel_w = float(x2 - x1)
```

`float(Any)` geçerli, hata vermez. Ama `Any` bulaşması, `disallow_any_expr`
açılırsa patlar. **Karar:** `disallow_any_expr` açılmamalı — ultralytics
kullanıldığı sürece bu savaş kaybedilir. Bunun yerine sınırda tek bir
dönüşüm fonksiyonu tanımlanmalı:

```python
def _to_bbox(raw: object) -> tuple[int, int, int, int]:
    """YOLO kutusunu tipli tuple'a çevirir. Any bulaşmasını burada durdurur."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (4,):
        msg = f"expected shape (4,), got {arr.shape}"
        raise ValueError(msg)
    x1, y1, x2, y2 = (int(v) for v in arr)
    return x1, y1, x2, y2
```

Bu fonksiyon hem `Any` sınırını kapatır hem de `A1` hatasının (beklenmeyen
şekil) daraltılmış istisnayla ele alınmasını sağlar.

### 3.2 `track_id` atanmamış olabilir

`00_durum_analizi/02_teknik_borc_ve_hatalar.md` A1 maddesindeki hata, mypy
tarafından da yakalanır — ama yalnızca doğru biçimde yazılırsa. Mevcut kodda
`track_id` `try` içinde atanıyor ve `except` içinde okunuyor; mypy bu akışı
"possibly undefined" olarak işaretler mi?

`strict` altında **işaretlemez**, çünkü mypy'nin `possibly-undefined` kontrolü
ayrı bir bayrak:

```toml
enable_error_code = ["possibly-undefined", "ignore-without-code", "redundant-expr"]
```

`possibly-undefined` açılırsa A1 hatası derleme zamanında yakalanır:

```console
error: Name "track_id" may be undefined  [possibly-undefined]
```

**Bu bayrak açılmalı.** A1 tam olarak bu sınıf bir hata ve mypy'nin onu
otomatik bulması, kod incelemesine bırakmaktan çok daha güvenli.

### 3.3 `Optional` dönüşler

```python
def _get_fallback_headlight_width(self, frame, bbox_2d) -> Optional[float]:
    ...
    return None
```

Çağıran taraf:

```python
hl_pixel_w = self._get_fallback_headlight_width(frame, boxes.xyxy[i].cpu().numpy())
if hl_pixel_w:
    pixel_w = hl_pixel_w
```

`if hl_pixel_w:` kontrolü `0.0` değerini de reddeder. Far mesafesi 0.0 piksel
olamaz ama bu bir **niyet** meselesi: `is not None` yazılmalı. `strict` altında
`warn_unused_ignores` ve daraltma kuralları bunu yakalamaz; ama
`--strict-equality` ve kod incelemesi yakalar. Öneri: `if hl_pixel_w is not None:`.

### 3.4 Sözlük tipleri

Mevcut:

```python
detections.append({
    "vehicle_id": track_id,
    "bbox_2d": (int(x1), int(y1), int(x2), int(y2)),
    "speed_kmh": speed_kmh
})
```

Dönüş tipi `List[Dict[str, Any]]` olarak bildirilmiş. `disallow_any_generics = true`
açık olduğunda `Any` içeren generic'ler uyarı üretir. Zaten
`01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md` içinde `Detection`
veri sınıfı tanımlandı; sözlük yerine bu sınıf kullanılınca sorun kendiliğinden
kalkar.

### 3.5 `cv2` sabitleri

`cv2.CAP_PROP_FRAME_WIDTH`, `cv2.FONT_HERSHEY_SIMPLEX` gibi sabitler
`opencv-python` stub'larında `Any`. `int(cap.get(...))` dönüşümü zaten
yapılıyor, sorun yok.

---

## 4. Kademeli geçiş planı

`strict = true`'yu ilk günden açmak, 141 lint hatasının üzerine tip hataları
yığar ve geliştirme durur. Sıra:

| Faz | Kapsam | Ayar | Bitirme koşulu |
| --- | --- | --- | --- |
| 1 | `src/vehicle_speed/geometry.py` | `strict = true` | sıfır hata |
| 2 | `+ tracking.py`, `calibration.py` | `strict = true` | sıfır hata |
| 3 | `+ telemetry/*` | `strict = true` | sıfır hata |
| 4 | `+ detector.py` | `strict = true` | sıfır hata |
| 5 | `+ visualization.py`, `cli.py` | `strict = true` | sıfır hata |
| 6 | `tests/`, `scripts/` | `strict = false`, `check_untyped_defs = true` | sıfır hata |

Faz 1-3'te `torch`/`ultralytics`/`cv2` hiç import edilmediği için mypy çok
hızlı çalışır (~1 saniye). Faz 4'te süre artar; bu,
`01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md` bölüm 4'teki import
sınırının pratik faydası.

Test dosyaları için `strict` zorlanmamalı: pytest fixture desenleri
(`@pytest.fixture` ile `Any` dönüşler) strict altında gereksiz sürtünme üretir.

Kademeli geçiş sırasında kapsam dışı dosyaları susturmak için:

```toml
[[tool.mypy.overrides]]
module = ["vehicle_speed.visualization", "vehicle_speed.cli"]
ignore_errors = true
```

Faz 5'te bu blok **silinmeli**. Silinmesi bir takip issue'su olarak açılmalı,
yoksa kalıcı hale gelir.

---

## 5. Yerel ve CI çalıştırma

```bash
# tüm kapsam
mypy

# tek modül
mypy src/vehicle_speed/geometry.py

# hata kodunu öğren
mypy --show-error-codes src/

# cache temizleyip yeniden
rm -rf .mypy_cache && mypy
```

CI'da cache saklanmalı, aksi halde her koşuda tam analiz yapılır:

```yaml
- name: Cache mypy
  uses: actions/cache@v4
  with:
    path: .mypy_cache
    key: mypy-${{ runner.os }}-${{ hashFiles('pyproject.toml') }}
```

`.mypy_cache` `.gitignore`'a eklenmeli.

---

## 6. `possibly-undefined` dışındaki yararlı hata kodları

```toml
[tool.mypy]
enable_error_code = [
    "possibly-undefined",   # A1 sınıfı hatalar
    "ignore-without-code",  # "# type: ignore" açıklamalı olmalı
    "redundant-expr",       # her zaman True/False olan koşullar
    "truthy-bool",          # if np.array: gibi yanlış doğruluk testleri
    "truthy-iterable",
    "unused-awaitable",
]
```

`truthy-bool` özellikle değerli: `if frame:` gibi bir kontrol numpy dizisinde
anlamsız ve çalışma zamanında `ValueError: truth value of an array is
ambiguous` üretir. Mevcut kodda bu hata yok (`frame is None or frame.size == 0`
kullanılıyor, doğru), ama `PLANLANIYOR` modüllerde kolayca yapılabilir.

---

## 7. Doğrulama

Bu raporun iddialarını test etmek için, `mypy` kurulduktan sonra:

```bash
uv pip install mypy
cat > /tmp/mypy_probe.py <<'PY'
def f(n: int) -> float:
    try:
        x = n / 1
        tid = int(x)
    except ZeroDivisionError as e:
        print(f"id {tid}: {e}")
        return 0.0
    return float(tid)
PY
mypy --strict --enable-error-code possibly-undefined /tmp/mypy_probe.py
```

Beklenen çıktı:

```console
error: Name "tid" may be undefined  [possibly-undefined]
```

Bu, A1 hatasının mypy ile otomatik yakalandığının kanıtıdır. Kod tabanına
uygulandığında aynı hata `kadir.py` ve `1_donem_kadir.py` içinde raporlanır.

---

## 8. Açık noktalar

1. **`torch` tipleri sürüme bağlı.** `torch` 2.x sürümleri arasında stub
   kapsamı değişiyor. Sürüm sabitlenmeden mypy sonuçları tekrarlanamaz.
   `uv.lock` bu yüzden zorunlu.
2. **`numpy` 2.5.1 kurulu, `scipy` yok.** NumPy 2.x stub'ları mypy 1.11+
   gerektirir. Eski mypy ile `numpy` üzerinde sahte hatalar çıkabilir.
3. **`disallow_any_expr` bilinçli olarak kapalı.** Açılırsa ultralytics
   sarmalayıcısı yazılamaz hale gelir. Bunun yerine `_to_bbox` gibi sınır
   fonksiyonlarıyla `Any` daraltılır. Bu bir ödünleşimdir, unutulmamalı.
