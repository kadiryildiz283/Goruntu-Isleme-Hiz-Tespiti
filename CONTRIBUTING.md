# Katkı Rehberi (Contributing Guide)

Bu depo, monoküler kamera görüntüsü ve ESP32 vidalı mil telemetrisi ile taşıt hızı tespiti üzerine geliştirilen akademik ve kurumsal bir açık kaynak araştırma platformudur.

Katkıda bulunmadan önce lütfen bağlayıcı mühendislik sözleşmemiz olan [.coderules](.coderules) ve [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) belgelerini inceleyiniz.

---

## 1. Geliştirme Ortamı Kurulumu

Projeyi klonladıktan sonra tek komutla izole ortamı kurabilirsiniz:

```bash
# Kurulum ve doğrulama betiği
./scripts/bootstrap.sh
```

Manuel kurulum için:
```bash
# uv ile Python 3.12 ortamı ve geliştirici paketleri
uv sync --extra dev

# Pre-commit kancalarını etkinleştirin
uv run pre-commit install
```

---

## 2. Kalite Kapıları ve Sözleşmeler

Her commit veya PR öncesinde aşağıdaki kontrollerin yerel ortamınızda yeşil olması şarttır:

1. **Linter & Formatter:**
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   ```
2. **Statik Tip Denetimi:**
   ```bash
   uv run mypy
   ```
3. **C++ Biçimlendirme (Firmware):**
   ```bash
   clang-format --dry-run --Werror firmware/src/main.cpp
   ```
4. **Birim & Entegrasyon Testleri:**
   ```bash
   uv run pytest -v --cov=vehicle_speed --cov-report=term-missing
   ```
   *Asgari test kapsamı (coverage) eşiği **%85**'tir.*

---

## 3. Kod ve Modül Kuralları (.coderules)

* **Tek Sorumluluk Prensibi (SRP):** Her modül tek bir amaca hizmet eder. Matematiksel hesaplamalar (`geometry.py`), nesne takibi (`tracking.py`) ve telemetri protokolü (`protocol.py`) bağımsız olmalıdır.
* **Görselleştirme İzolasyonu:** `cv2.imshow` gibi grafik ve pencere çağrıları kesinlikle `src/vehicle_speed/visualization.py` haricindeki hiçbir dosyada bulunamaz.
* **Tip Notasyonları:** Tüm fonksiyon ve metotlar dönüş tipi ve argüman tipleriyle birlikte (`mypy strict`) tanımlanmalı, Google stili docstring içermelidir.

---

## 4. Commit ve Dal (Branch) Protokolü

* Doğrudan `main` dalına commit atılamaz. Geliştirmeler özellik dalında (`feat/...` veya `fix/...`) yapılır.
* Commit mesajları **Conventional Commits** standardına uygun olmalıdır:
  * `feat:` Yeni bir özellik veya modül
  * `fix:` Hata düzeltmesi
  * `refactor:` Yeniden yapılandırma
  * `style:` Kod biçimlendirme
  * `test:` Test ekleme / güncelleme
  * `chore:` Altyapı, bağımlılık ve CI güncellemeleri
