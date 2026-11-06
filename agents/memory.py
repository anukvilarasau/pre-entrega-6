"""
memory.py — Memoria local de búsquedas con ChromaDB.

Todo lo que el investigador trae de la web (Tavily) se guarda acá, con
metadatos: destino, tema, URL, título y FECHA de búsqueda. La próxima vez que
alguien pregunte algo parecido, se busca primero acá: es gratis, instantáneo
y no gasta créditos de Tavily.

¿Cuándo un dato guardado "sirve"? Tiene que pasar TRES filtros:
  1. Mismo destino (y mismo tema, si se pide uno)  → filtro por metadatos.
  2. No más viejo que MEMORIA_MAX_DIAS (7 días)    → filtro por metadatos
     sobre `fecha_ts` (la fecha como número, porque Chroma solo compara números).
  3. Parecido a la consulta (distancia coseno ≤ MEMORIA_MAX_DISTANCIA)
     → filtro por embeddings (búsqueda semántica).

¿Qué se compara en el filtro 3? La consulta nueva contra "la consulta que
originó la búsqueda + el título del resultado" (no contra la página entera:
un texto largo diluye el tema y todo parece poco parecido). El contenido de
la página se guarda aparte, en los metadatos, y se devuelve tal cual.

Si ninguno pasa los tres, el investigador tiene que ir a la web.

Embeddings:
- Modo real: el modelo por defecto de ChromaDB (all-MiniLM-L6-v2 en ONNX,
  SIN torch; se descarga una sola vez, ~80 MB).
- Modo mock y tests: `EmbeddingHash`, un embedding simple por palabras que no
  descarga nada, para que todo corra offline.
"""

import hashlib
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction, register_embedding_function

import config
from state import Tema

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes: 'Bariloche ', 'BARILOCHE' y 'bariloché' son lo mismo."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", sin_tildes.lower()).strip()


def id_memoria(url: str, destino: str, tema: str) -> str:
    """ID estable: la misma URL + destino + tema siempre da el mismo ID.

    Así, si se vuelve a buscar lo mismo en la web, el registro se ACTUALIZA
    (con la fecha nueva) en lugar de duplicarse.
    """
    clave = f"{url}|{normalizar(destino)}|{tema}"
    return "mem-" + hashlib.sha1(clave.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Embedding offline para mock y tests
# ---------------------------------------------------------------------------
@register_embedding_function
class EmbeddingHash(EmbeddingFunction[Documents]):
    """Embedding simple "bolsa de palabras con hashing" (sin descargas, determinístico).

    Cada palabra (y su raíz de 5 letras, para que 'pasaje' y 'pasajes' se
    parezcan) suma 1 en una posición del vector elegida por un hash. No entiende
    sinónimos como un modelo real, pero alcanza para el modo mock y los tests.
    """

    DIM: int = 256

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 (nombre exigido por Chroma)
        import numpy as np

        vectores: Embeddings = []
        for texto in input:
            v = np.zeros(self.DIM, dtype=np.float32)
            for palabra in re.findall(r"[a-z0-9]+", normalizar(texto)):
                for token in {palabra, palabra[:5]}:
                    v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norma = math.sqrt(float((v * v).sum())) or 1.0
            vectores.append(v / norma)
        return vectores

    @staticmethod
    def name() -> str:
        return "embedding_hash_pe6"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "EmbeddingHash":  # noqa: ARG004
        return EmbeddingHash()

    def default_space(self) -> str:  # type: ignore[override]
        return "cosine"

    def supported_spaces(self) -> list[str]:  # type: ignore[override]
        return ["cosine"]


# ---------------------------------------------------------------------------
# Datos que entran y salen de la memoria
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultadoWeb:
    """Un resultado de búsqueda web (lo que devuelve Tavily, ya limpio)."""

    url: str
    titulo: str
    contenido: str


@dataclass(frozen=True)
class RegistroMemoria:
    """Un dato recuperado de la memoria, con su "ficha" completa."""

    id: str
    texto: str
    url: str
    titulo: str
    destino: str
    tema: str
    fecha: datetime
    distancia: float

    def antiguedad_dias(self, ahora: datetime) -> float:
        return round((ahora - self.fecha).total_seconds() / 86400, 1)


@dataclass
class ResultadoBusquedaMemoria:
    """Lo que devuelve una búsqueda en memoria: lo que sirve y por qué se descartó el resto."""

    utiles: list[RegistroMemoria] = field(default_factory=list)
    descartados_por_antiguedad: int = 0   # del destino/tema, pero más viejos que el límite
    descartados_por_relevancia: int = 0   # recientes, pero poco parecidos a la consulta

    @property
    def hay_utiles(self) -> bool:
        return bool(self.utiles)


# ---------------------------------------------------------------------------
# La memoria
# ---------------------------------------------------------------------------
class MemoriaBusquedas:
    """Envoltorio de una colección de ChromaDB persistente en disco."""

    def __init__(
        self,
        ruta: Path = config.CHROMA_DIR,
        embedding: EmbeddingFunction[Documents] | None = None,
        coleccion: str = config.CHROMA_COLECCION,
        max_dias: int = config.MEMORIA_MAX_DIAS,
        max_distancia: float = config.MEMORIA_MAX_DISTANCIA,
    ) -> None:
        ruta.mkdir(parents=True, exist_ok=True)
        self.ruta = ruta
        self.max_dias = max_dias
        self.max_distancia = max_distancia
        self._cliente = chromadb.PersistentClient(path=str(ruta))
        self._coleccion = self._cliente.get_or_create_collection(
            name=coleccion,
            embedding_function=embedding or DefaultEmbeddingFunction(),  # type: ignore[arg-type]
            configuration={"hnsw": {"space": "cosine"}},  # distancia coseno: 0 = idéntico
        )

    # ------------------------------------------------------------------
    def guardar(
        self,
        resultados: list[ResultadoWeb],
        destino: str,
        tema: Tema,
        consulta: str,
        fecha: datetime | None = None,
    ) -> list[str]:
        """Guarda (o actualiza) resultados web con sus metadatos. Devuelve los IDs."""
        if not resultados:
            return []
        fecha = fecha or datetime.now().astimezone()
        ids: list[str] = []
        documentos: list[str] = []
        metadatos: list[dict[str, Any]] = []
        for r in resultados:
            ids.append(id_memoria(r.url, destino, tema))
            # Lo que se convierte en embedding: la pregunta original + el título.
            documentos.append(f"{consulta} | {r.titulo}".strip())
            metadatos.append({
                "destino": normalizar(destino),
                "tema": tema,
                "url": r.url,
                "titulo": r.titulo[:300],
                "contenido": r.contenido[:3000],
                "consulta": consulta[:300],
                "fecha_ts": int(fecha.timestamp()),   # para filtrar por antigüedad
                "fecha_iso": fecha.isoformat(timespec="seconds"),
            })
        # upsert = insertar o, si el ID ya existe, actualizar (no duplica)
        self._coleccion.upsert(ids=ids, documents=documentos, metadatas=metadatos)  # type: ignore[arg-type]
        log.info("Memoria: resultados guardados: %d [destino=%s, tema=%s]", len(ids), normalizar(destino), tema)
        return ids

    # ------------------------------------------------------------------
    def buscar(
        self,
        consulta: str,
        destino: str,
        tema: Tema | None = None,
        k: int = 4,
        ahora: datetime | None = None,
    ) -> ResultadoBusquedaMemoria:
        """Busca datos útiles: mismo destino/tema, recientes y parecidos a la consulta."""
        ahora = ahora or datetime.now().astimezone()
        limite_ts = int((ahora - timedelta(days=self.max_dias)).timestamp())

        filtro_base: list[dict[str, Any]] = [{"destino": normalizar(destino)}]
        if tema is not None:
            filtro_base.append({"tema": tema})

        # Filtros 1 + 2 (metadatos): destino/tema y antigüedad.
        filtro_recientes = _y(filtro_base + [{"fecha_ts": {"$gte": limite_ts}}])
        filtro_viejos = _y(filtro_base + [{"fecha_ts": {"$lt": limite_ts}}])
        viejos = len(self._coleccion.get(where=filtro_viejos, include=[])["ids"])

        resultado = ResultadoBusquedaMemoria(descartados_por_antiguedad=viejos)
        if self._coleccion.count() == 0:
            return resultado

        respuesta = self._coleccion.query(
            query_texts=[consulta],
            n_results=k,
            where=filtro_recientes,
            include=["metadatas", "distances"],
        )
        ids = (respuesta["ids"] or [[]])[0]
        metas = (respuesta["metadatas"] or [[]])[0]
        dists = (respuesta["distances"] or [[]])[0]

        # Filtro 3 (embeddings): parecido suficiente a la consulta.
        for id_, meta, dist in zip(ids, metas, dists, strict=True):
            if dist > self.max_distancia:
                resultado.descartados_por_relevancia += 1
                continue
            resultado.utiles.append(RegistroMemoria(
                id=id_,
                texto=str(meta.get("contenido", "")),
                url=str(meta["url"]),
                titulo=str(meta.get("titulo", "")),
                destino=str(meta["destino"]),
                tema=str(meta["tema"]),
                fecha=datetime.fromisoformat(str(meta["fecha_iso"])),
                distancia=round(float(dist), 3),
            ))
        log.info(
            "Memoria: '%s' → útiles: %d, vencidos: %d, poco relevantes: %d",
            consulta, len(resultado.utiles), resultado.descartados_por_antiguedad,
            resultado.descartados_por_relevancia,
        )
        return resultado

    def cantidad(self) -> int:
        return self._coleccion.count()


def _y(condiciones: list[dict[str, Any]]) -> dict[str, Any]:
    """Chroma exige $and solo si hay 2 o más condiciones."""
    return condiciones[0] if len(condiciones) == 1 else {"$and": condiciones}
