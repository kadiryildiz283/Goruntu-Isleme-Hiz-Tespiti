# Pre-commit: Yerel Kalite Kapıları

**Araç durumu (doğrulandı):** `pre-commit` bu makinede **kurulu değil**.
**İlke:** CI'da yakalanan her hata, yerelde yakalanmalıydı. Pre-commit bu
farkı kapatır; CI yerine geçmez.

---

## 1. Kurulum

```bash
uv pip install pre-commit
pre-commit install                  # .git/hooks/pre-commit yazar
pre-commit install --hook-type commit-msg   # Conventional Commits denetimi
pre-commit install --hook-type pre-push     # ağır kontroller push'ta
```

**Ön koşul:** çalışan bir `.git` dizini. Depo şu an bozuk
(`00_durum_analizi/01_mevcut_envanter.md` bölüm 1), dolayısıyla `pre-commit
install` bugün başarısız olur:

```console
$ pre-commit install
fatal: not a git repository
```

Bu yüzden pre-commit, uygulama yol haritasında **depo onarımından sonra** gelir.

---

## 2. Yapılandırma

`.pre-commit-config.yaml`:

```yaml
default_language_version:
  python: python3.12

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
        args: ["--unsafe"]   # GitHub Actions !input tarzı etiketler için
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: check-case-conflict
      - id: check-added-large-files
        args: ["--maxkb=1024"]
      - id: end-of-file-fixer
        exclude: ^(docs/paper/figures/|archive/)
      - id: trailing-whitespace
        args: ["--markdown-linebreaks"]
      - id: mixed-line-ending
        args: ["--fix=lf"]
      - id: detect-private-key
      - id: check-symlinks
      - id: destroyed-symlinks
      - id: no-commit-to-branch
        args: ["--branch", "main"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.16.7            # pyproject.toml'daki ruff sürümüyle AYNI olmalı
    hooks:
      - id: ruff
        args: ["--fix", "--exit-non-zero-on-fix"]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2            # pyproject.toml'daki mypy sürümüyle AYNI olmalı
    hooks:
      - id: mypy
        additional_dependencies:
          - "numpy==2.5.1"
          - "types-PyYAML"
        args: ["--config-file=pyproject.toml"]
        files: ^(src|scripts)/
        pass_filenames: false

  - repo: https://github.com/codespell-project/codespell
    rev: v2.3.0
    hooks:
      - id: codespell
        additional_dependencies: ["tomli"]
        args: ["--toml", "pyproject.toml"]
        exclude: ^(archive/|docs/paper/)

  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.29.0
    hooks:
      - id: commitizen
        stages: [commit-msg]

  - repo: https://github.com/python-jsonschema/check-jsonschema
    rev: 0.29.4
    hooks:
      - id: check-github-workflows
      - id: check-github-actions

  - repo: https://github.com/pre-commit/mirrors-clang-format
    rev: v19.1.3
    hooks:
      - id: clang-format
        files: ^firmware/(src|include)/.*\.(c|cpp|h|hpp)$
        types_or: []
```

`pyproject.toml` içine codespell ayarı:

```toml
[tool.codespell]
skip = ".git,archive,docs/paper,*.pdf,*.png,*.mp4,uv.lock"
ignore-words-list = "nd,ot,te"
```

---

## 3. Kanca seçimlerinin gerekçesi

### `check-added-large-files --maxkb=1024`

Depoda 273 MB'lık `yolov8x.onnx` ve 137 MB'lık `yolov8x.pt` var
(`00_durum_analizi/01_mevcut_envanter.md` bölüm 3). `.gitignore` yazılsa bile
`git add -f` ile zorla eklenebilirler. Bu kanca 1 MB üzerindeki her yeni
dosyayı commit aşamasında durdurur. GitHub'ın 100 MB sert sınırından çok önce
uyarı verir.

### `no-commit-to-branch --branch main`

`05_yonetisim/04_branch_koruma_inceleme.md` içindeki branch protection
kuralının yerel karşılığı. GitHub tarafında kural koymadan önce yerelde
alışkanlık oluşturur.

### `check-yaml --unsafe`

GitHub Actions workflow dosyaları standart YAML olmayan `${{ }}` ifadeleri ve
bazı durumlarda `!input` etiketleri içerir. `--unsafe` bayrağı olmadan kanca
bu dosyaları reddeder.

### `check-github-workflows`

Yapıştırılan plandaki `uses: actions/checkout @v4` hatası (`@` öncesi boşluk)
bu kanca tarafından yakalanır. YAML geçerli ama action referansı geçersiz
olduğu için `check-yaml` tek başına yetmez; `check-jsonschema` şema düzeyinde
denetler.

### `mirrors-mypy` + `pass_filenames: false`

`pass_filenames: false` olmadan mypy yalnızca değişen dosyaları görür ve
modüller arası tutarlılığı denetleyemez. `files: ^(src|scripts)/` ile kapsam
sınırlandırılır, `tests/` dışarıda kalır
(`02_kod_kalitesi/02_mypy_strict.md` bölüm 4, Faz 6 kararı).

`additional_dependencies` içinde `numpy` gerekli: mypy hook'u izole bir
ortamda çalışır ve `numpy` stub'larını göremezse sahte `import-untyped`
hataları üretir.

### `clang-format` firmware için

`firmware/src/main.cpp` bugün elle biçimlendirilmiş ve `esp32.c` ile 10 satır
farklı — farkın tamamı yorum biçimi. İki kopyanın ayrışmasının sebebi tam
olarak otomatik biçimlendirme olmaması. `clang-format` bu ayrışmayı bitirir.

Sürüm notu: sistemde `clang-format 23.1.1` kurulu, ama pre-commit
`mirrors-clang-format v19.1.3` çeker. Sürüm farkı farklı çıktı üretebilir.
İki seçenek:

1. `repo: local` ile sistemdeki clang-format kullanılır (tutarlı, ama katkıcı
   makinesinde kurulu olmasını gerektirir)
2. pre-commit sürümüne uyulur ve yerel kurulum atlanır (taşınabilir)

**Öneri: seçenek 2.** Açık kaynak depoda katkıcı makinesi kontrol edilemez.
`.clang-format` dosyası sürümden bağımsız olacak kadar belirleyici yazılmalı:

```text
# .clang-format
BasedOnStyle: Google
IndentWidth: 2
ColumnLimit: 100
PointerAlignment: Left
AllowShortFunctionsOnASingleLine: Inline
SortIncludes: CaseSensitive
```

---

## 4. Conventional Commits

`commitizen` `commit-msg` aşamasında çalışır. `pyproject.toml`:

```toml
[tool.commitizen]
name = "cz_conventional_commits"
version = "0.1.0"
tag_format = "v$version"
update_changelog_on_bump = true
major_version_zero = true
commit_parser = "conventional"
changelog_file = "CHANGELOG.md"
bump_files = [
    { pattern = "pyproject.toml", pattern_replacement = 'version = "$new_version"' },
    { pattern = "src/vehicle_speed/__init__.py", pattern_replacement = '__version__ = "$new_version"' },
]
```

İzin verilen tipler:

| Tip | Kullanım |
| --- | --- |
| `feat` | yeni işlev |
| `fix` | hata düzeltme |
| `docs` | yalnızca doküman |
| `style` | davranış değiştirmeyen biçimlendirme |
| `refactor` | davranış değiştirmeyen yeniden yapılandırma |
| `perf` | performans iyileştirmesi |
| `test` | test ekleme/düzeltme |
| `build` | bağımlılık, paketleme |
| `ci` | workflow değişiklikleri |
| `chore` | diğer bakım işleri |
| `revert` | geri alma |

Kapsam (scope) bu proje için `01_label_taksonomisi.md` içindeki `component/*`
adlarıyla aynı olmalı:

```text
feat(telemetry): add pyserial-based reader with configurable port
fix(vision): guard against unbound track_id in detection loop
ci(workflows): split lint and test jobs, enable uv cache
docs(paper): document HFOV calibration procedure
chore(firmware): remove duplicate esp32.c
```

İki ayrı adlandırma şeması (label `component/telemetry`, commit scope
`telemetry`) tutarlı tutulmalı; aksi halde otomatik eşleme yapılamaz.

### Sürüm üretimi

```bash
cz bump --dry-run     # hangi sürümün geleceğini göster
cz bump               # tag + CHANGELOG + dosya güncelleme
cz changelog          # CHANGELOG.md yeniden üret
```

`major_version_zero = true` önemli: proje `0.x` sürümünde, bu ayar altında
`feat!` bile 1.0.0'a sıçratmaz.

---

## 5. Aşama (stage) dağılımı

Her kontrolü `pre-commit` aşamasına koymak commit'i yavaşlatır. Dağılım:

| Aşama | Kancalar | Süre hedefi |
| --- | --- | --- |
| `pre-commit` | ruff, ruff-format, yaml/json/toml, large-files, private-key, codespell | < 5 sn |
| `commit-msg` | commitizen | < 1 sn |
| `pre-push` | mypy, pytest (birim), clang-format | < 60 sn |

`pre-push` için ayrı kanca tanımı:

```yaml
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest (unit only)
        entry: uv run pytest tests/unit -q --no-cov
        language: system
        pass_filenames: false
        stages: [pre-push]
      - id: mypy-full
        name: mypy (full)
        entry: uv run mypy
        language: system
        pass_filenames: false
        stages: [pre-push]
```

`stages: [pre-push]` yazılmazsa kanca her commit'te çalışır ve geliştirme
hızını düşürür.

---

## 6. Kaçış yolları ve ne zaman meşru

```bash
git commit --no-verify          # tüm kancaları atlar
SKIP=mypy git commit -m "..."   # tek kancayı atlar
pre-commit run --all-files      # elle tam koşu
pre-commit run ruff --files src/vehicle_speed/geometry.py
```

`--no-verify` **kullanılmamalı**. Bir kanca engel oluyorsa ya kod düzeltilir ya
da kanca yapılandırması güncellenir. Kanca atlamak, sorunu CI'a taşır ve orada
yakalanma maliyeti daha yüksektir.

Meşru tek istisna: büyük bir yeniden yapılandırma sırasında ara commit'ler.
Bu durumda `SKIP=mypy` tercih edilir, `--no-verify` değil — biçimlendirme ve
büyük dosya kontrolleri açık kalır.

---

## 7. Sürüm eşitleme denetimi

Üç yerde sürüm bilgisi var ve ayrışmaları sessiz arıza üretir:

| Yer | Alan | Değer |
| --- | --- | --- |
| `pyproject.toml` | `[project.optional-dependencies] dev` | `ruff==0.16.7` |
| `.pre-commit-config.yaml` | `rev` | `v0.16.7` |
| CI workflow | `uv sync --extra dev` | `pyproject.toml`'dan okur |

Pre-commit `rev` ile `pyproject.toml` sürümü ayrışırsa, yerelde geçen kod
CI'da kalır (veya tersi). Bu ayrışmayı yakalayan bir CI adımı:

```yaml
- name: Verify pre-commit revisions match locked dev deps
  run: |
    set -euo pipefail
    python - <<'PY'
    import re, sys, tomllib, pathlib
    cfg = pathlib.Path(".pre-commit-config.yaml").read_text()
    py = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    dev = dict(
        re.match(r"([a-z0-9_-]+)\s*==\s*([^\s;,]+)", d).groups()
        for d in py["project"]["optional-dependencies"]["dev"]
        if re.match(r"([a-z0-9_-]+)\s*==", d)
    )
    checks = {"ruff": "ruff-pre-commit", "mypy": "mirrors-mypy"}
    revs = dict(re.findall(r"rev:\s*v?([0-9.]+).*?\n", cfg))
    bad = []
    for pkg, repo_frag in checks.items():
        if pkg in dev:
            block = re.search(rf"{repo_frag}.*?rev:\s*v?([0-9.]+)", cfg, re.S)
            if block and block.group(1) != dev[pkg]:
                bad.append(f"{pkg}: pyproject={dev[pkg]} pre-commit={block.group(1)}")
    if bad:
        print("SÜRÜM AYRIŞMASI:\n" + "\n".join(bad))
        sys.exit(1)
    print("Sürümler eşleşik.")
    PY
```

Bu adım `03_ci_cd/02_ci_workflow_adimlari.md` içindeki lint job'ına eklendi.

---

## 8. Doğrulama

```bash
# tüm kancaları tüm dosyalarda çalıştır
pre-commit run --all-files

# tek kanca
pre-commit run ruff --all-files

# hangi kancalar kurulu
pre-commit run --help
ls .git/hooks/    # pre-commit, commit-msg, pre-push görünmeli
```

`pre-commit run --all-files` çıktısının ilk koşuda kırmızı olması normaldir:
141 lint hatası ve biçimlendirme farkları raporlanır. Beklenen davranış,
`--fix` sayesinde bir kısmının otomatik düzeltilip dosyaların değiştirilmiş
olarak işaretlenmesidir. Bu durumda commit yeniden yapılmalıdır.
