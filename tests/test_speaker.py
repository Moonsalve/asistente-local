"""Tests de la verificacion de locutor.

El modelo (26 MB) no se descarga en CI, asi que aqui se prueba la LOGICA DE
DECISION con un extractor de mentira. Lo que de verdad discrimina —el modelo—
se valido aparte con cuatro voces reales generadas por `say` de macOS, y el
resultado esta en el docstring de `audio/speaker.py`: separa voces limpias, y se
degrada con ruido de fondo.

Lo que se fija aqui es el comportamiento ante lo que puede salir mal, porque el
fallo caro de este modulo no es dejar pasar a un extrano: es dejar sordo a su
dueno sin decir por que.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asistente.audio.speaker import SpeakerError, SpeakerGate, build_profile

SAMPLE_RATE = 16_000


class FakeEmbedder:
    """Devuelve vectores prefijados. Si `raises` esta puesto, simula que el
    extractor de features revienta."""

    def __init__(self, vectors: list[np.ndarray] | None = None, raises: bool = False) -> None:
        self._vectors = list(vectors or [])
        self._raises = raises
        self.calls = 0

    def encode(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        self.calls += 1
        if self._raises:
            raise SpeakerError("fbank roto")
        vector = self._vectors.pop(0) if self._vectors else np.array([1.0, 0.0], dtype=np.float32)
        return vector / np.linalg.norm(vector)


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_acepta_la_voz_enrolada() -> None:
    centroide = np.array([1.0, 0.0], dtype=np.float32)
    gate = SpeakerGate(FakeEmbedder([np.array([0.95, 0.05])]), centroide, threshold=0.45)

    veredicto = gate.check(_audio(2.0), SAMPLE_RATE)
    assert veredicto.accepted is True
    assert veredicto.score > 0.9


def test_rechaza_otra_voz() -> None:
    centroide = np.array([1.0, 0.0], dtype=np.float32)
    gate = SpeakerGate(FakeEmbedder([np.array([0.1, 1.0])]), centroide, threshold=0.45)

    veredicto = gate.check(_audio(2.0), SAMPLE_RATE)
    assert veredicto.accepted is False
    assert "otra voz" in veredicto.reason


def test_las_frases_cortas_se_aceptan_sin_verificar() -> None:
    """Con menos de medio segundo el embedding dice mas del fonema que de quien
    habla. Rechazar por un vector poco fiable seria peor que dejar que decida la
    palabra clave."""
    embedder = FakeEmbedder([np.array([0.1, 1.0])])  # se rechazaria si se usara
    gate = SpeakerGate(embedder, np.array([1.0, 0.0], dtype=np.float32), threshold=0.45)

    veredicto = gate.check(_audio(0.3), SAMPLE_RATE)
    assert veredicto.accepted is True
    assert embedder.calls == 0, "ni siquiera deberia haber calculado el embedding"


def test_ante_un_fallo_del_extractor_se_oye() -> None:
    """La regla que gobierna todo el modulo: ante la duda, oir. Quedarse sordo
    por un fallo interno es el peor resultado posible, y ademas silencioso."""
    gate = SpeakerGate(
        FakeEmbedder(raises=True), np.array([1.0, 0.0], dtype=np.float32), threshold=0.45
    )
    assert gate.check(_audio(2.0), SAMPLE_RATE).accepted is True


def test_el_score_se_devuelve_tambien_al_aceptar() -> None:
    """Es el unico dato con el que ajustar el umbral con hechos en vez de a ojo,
    y por eso el pipeline lo registra siempre."""
    gate = SpeakerGate(
        FakeEmbedder([np.array([0.8, 0.6])]), np.array([1.0, 0.0], dtype=np.float32), 0.45
    )
    veredicto = gate.check(_audio(2.0), SAMPLE_RATE)
    assert veredicto.accepted is True
    assert veredicto.score == pytest.approx(0.8, abs=0.01)


def test_el_perfil_promedia_varias_frases() -> None:
    """Un solo embedding arrastra la entonacion y el contenido de esa frase
    concreta; el centroide de varias es lo que hace el umbral estable."""
    embedder = FakeEmbedder([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    centroide = build_profile(embedder, [_audio(2.0), _audio(2.0)], SAMPLE_RATE)

    assert np.linalg.norm(centroide) == pytest.approx(1.0, abs=1e-5)
    assert centroide[0] == pytest.approx(centroide[1], abs=1e-5)


def test_sin_muestras_no_hay_perfil() -> None:
    with pytest.raises(SpeakerError):
        build_profile(FakeEmbedder(), [], SAMPLE_RATE)


def test_el_perfil_se_lee_del_disco(tmp_path: Path) -> None:
    perfil = tmp_path / "voz.json"
    perfil.write_text(json.dumps({"centroid": [1.0, 0.0], "samples": 5}), encoding="utf-8")

    gate = SpeakerGate.from_profile(FakeEmbedder([np.array([0.9, 0.1])]), perfil, 0.45)
    assert gate.check(_audio(2.0), SAMPLE_RATE).accepted is True
