"""Shared pytest fixtures, hardware flag registration, and mocks."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Komut satırına donanım testi çalıştırma seçeneği ekler."""
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Bağlı fiziksel donanım gerektiren testleri çalıştırır",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """--run-hardware verilmediğinde donanım testlerini otomatik atlar."""
    if config.getoption("--run-hardware"):
        return
    skip_hardware = pytest.mark.skip(
        reason="Fiziksel donanım testi (çalıştırmak için: --run-hardware)",
    )
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)
