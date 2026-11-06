"""Tests de agents/memory.py: guardar, filtrar por destino/tema/antigüedad/relevancia, persistir."""

from datetime import datetime, timedelta
from pathlib import Path

from agents.memory import EmbeddingHash, MemoriaBusquedas, ResultadoWeb, id_memoria, normalizar

AHORA = datetime(2026, 11, 5, 12, 0).astimezone()
VUELOS = ResultadoWeb("https://a.example/vuelos", "Vuelos Mendoza Bariloche", "Pasaje ida y vuelta $480.000 por persona")
HOSTEL = ResultadoWeb("https://a.example/hostel", "Hostel en Bariloche", "Habitación doble $60.000 por noche")


def test_guarda_y_encuentra_lo_reciente(memoria: MemoriaBusquedas) -> None:
    memoria.guardar([VUELOS], "Bariloche", "pasajes", "vuelos mendoza bariloche", fecha=AHORA)
    r = memoria.buscar("precio pasaje vuelo Mendoza Bariloche", "Bariloche", "pasajes", ahora=AHORA)
    assert r.hay_utiles
    reg = r.utiles[0]
    assert reg.id == id_memoria(VUELOS.url, "Bariloche", "pasajes")
    assert reg.url == VUELOS.url and reg.tema == "pasajes" and reg.distancia <= memoria.max_distancia


def test_descarta_datos_vencidos(memoria: MemoriaBusquedas) -> None:
    memoria.guardar([VUELOS], "Bariloche", "pasajes", "vuelos", fecha=AHORA - timedelta(days=10))
    r = memoria.buscar("pasaje vuelo Mendoza Bariloche", "Bariloche", "pasajes", ahora=AHORA)
    assert not r.hay_utiles
    assert r.descartados_por_antiguedad == 1


def test_limite_exacto_de_7_dias_todavia_sirve(memoria: MemoriaBusquedas) -> None:
    memoria.guardar([VUELOS], "Bariloche", "pasajes", "vuelos", fecha=AHORA - timedelta(days=7))
    assert memoria.buscar("pasaje vuelo Mendoza Bariloche", "Bariloche", "pasajes", ahora=AHORA).hay_utiles


def test_filtra_por_destino_y_tema(memoria: MemoriaBusquedas) -> None:
    memoria.guardar([VUELOS], "Bariloche", "pasajes", "vuelos", fecha=AHORA)
    memoria.guardar([HOSTEL], "Bariloche", "alojamiento", "hostel", fecha=AHORA)
    assert not memoria.buscar("pasaje vuelo Mendoza", "Ushuaia", "pasajes", ahora=AHORA).hay_utiles
    r = memoria.buscar("hostel habitación doble noche Bariloche", "Bariloche", "alojamiento", ahora=AHORA)
    assert [u.url for u in r.utiles] == [HOSTEL.url]


def test_descarta_lo_poco_relevante(memoria: MemoriaBusquedas) -> None:
    memoria.guardar([VUELOS], "Bariloche", "pasajes", "vuelos", fecha=AHORA)
    r = memoria.buscar("museos de arte contemporáneo nocturnos", "Bariloche", None, ahora=AHORA)
    assert not r.hay_utiles and r.descartados_por_relevancia == 1


def test_destino_normalizado(memoria: MemoriaBusquedas) -> None:
    assert normalizar("  BARILOCHÉ ") == "bariloche"
    memoria.guardar([VUELOS], "BARILOCHÉ", "pasajes", "vuelos", fecha=AHORA)
    assert memoria.buscar("pasaje vuelo Mendoza Bariloche", "bariloche", "pasajes", ahora=AHORA).hay_utiles


def test_misma_url_se_actualiza_sin_duplicar(memoria: MemoriaBusquedas) -> None:
    memoria.guardar([VUELOS], "Bariloche", "pasajes", "vuelos", fecha=AHORA - timedelta(days=10))
    memoria.guardar([VUELOS], "Bariloche", "pasajes", "vuelos", fecha=AHORA)  # se volvió a buscar
    assert memoria.cantidad() == 1
    assert memoria.buscar("pasaje vuelo Mendoza Bariloche", "Bariloche", "pasajes", ahora=AHORA).hay_utiles


def test_es_persistente_en_disco(tmp_path: Path) -> None:
    ruta = tmp_path / "chroma"
    MemoriaBusquedas(ruta=ruta, embedding=EmbeddingHash()).guardar([VUELOS], "Bariloche", "pasajes", "v", fecha=AHORA)
    otra = MemoriaBusquedas(ruta=ruta, embedding=EmbeddingHash())  # como si se reiniciara el programa
    assert otra.cantidad() == 1


def test_memoria_vacia_no_rompe(memoria: MemoriaBusquedas) -> None:
    r = memoria.buscar("lo que sea", "Bariloche", "clima", ahora=AHORA)
    assert not r.hay_utiles and r.descartados_por_antiguedad == 0
