"""Fixtures compartidas: todo corre offline, sin claves y en carpetas temporales."""

from pathlib import Path

import pytest

from agents.memory import EmbeddingHash, MemoriaBusquedas
from agents.research_tools import BuscadorMock


@pytest.fixture
def memoria(tmp_path: Path) -> MemoriaBusquedas:
    """ChromaDB nueva en una carpeta temporal, con el embedding offline."""
    return MemoriaBusquedas(ruta=tmp_path / "chroma", embedding=EmbeddingHash())


@pytest.fixture
def buscador() -> BuscadorMock:
    """Tavily simulado (lee data/mock_busquedas.json)."""
    return BuscadorMock()
