#!/usr/bin/env bash
set -euo pipefail

# 1. uv kontrolü
command -v uv >/dev/null || {
    echo "HATA: uv bulunamadı. Kurulum için: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}

# 2. Python 3.12 pinlenmesi
echo "[1/4] Python 3.12 ortamı hazırlanıyor..."
uv python install 3.12

# 3. Sanal ortam ve geliştirme bağımlılıklarının senkronizasyonu
echo "[2/4] Paketler ve dev bağımlılıkları yükleniyor..."
uv sync --extra dev

# 4. Pre-commit kancalarının git deposuna kurulması
echo "[3/4] Git pre-commit kancaları kuruluyor..."
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg 2>/dev/null || true

# 5. Yerel doğrulama
echo "[4/4] Temel sağlık kontrolleri çalıştırılıyor..."
uv run python -c "import vehicle_speed; print('✓ Paket sürümlendi:', vehicle_speed.__version__)"
uv run ruff check .
uv run mypy
uv run pytest -q tests/unit

echo "=============================================="
echo "✓ Geliştirme ortamı başarıyla kuruldu ve doğrulandı!"
echo "=============================================="
