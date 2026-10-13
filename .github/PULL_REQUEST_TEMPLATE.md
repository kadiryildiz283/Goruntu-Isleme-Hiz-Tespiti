## Ne Değişti?

<!-- 2-4 cümle. Hangi dosya/fonksiyon, hangi davranış iyileştirildi veya eklendi? -->

## Neden?

<!-- Hangi issue'yu çözüyor, hangi gözlem bu değişikliği gerektirdi? -->

## Değişiklik Tipi

- [ ] `fix` — Hata düzeltme
- [ ] `feat` — Yeni özellik / modül
- [ ] `refactor` — Davranış değişmeden yapısal iyileştirme
- [ ] `style` — Kod biçimlendirme
- [ ] `test` — Test ekleme / güncelleme
- [ ] `docs` — Dokümantasyon
- [ ] `chore` / `ci` — Altyapı, bağımlılık, CI/CD

## Doğrulama ve Kanıt Tablosu (.coderules Madde 0)

<!-- Çalıştırılan komutlar ve sonuçları -->

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -v --cov=vehicle_speed
```

- [ ] `ruff check .` sıfır hata (Passed)
- [ ] `ruff format --check .` sıfır fark (Passed)
- [ ] `mypy` strict tip kontrolü sıfır hata (Passed)
- [ ] `pytest` tüm testler yeşil (Passed)
- [ ] Asgari %85 test kapsamı (coverage) korundu

## İlgili Issue

Closes #
