"""Tests de la supresion de ruido.

Que "suena mejor" no se puede afirmar en un test unitario; lo que si se puede
fijar son las propiedades que hacen que el modulo no haga dano, que es donde
estuvo el bug real: la reconstruccion no era exacta y estaba metiendo distorsion
en el audio ANTES de que Whisper lo viera.
"""

from __future__ import annotations

import numpy as np
import pytest

from asistente.audio.denoise import N_FFT, _istft, _stft, denoise, estimate_noise, snr_db

SAMPLE_RATE = 16_000


def _voz_sintetica(seconds: float = 1.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Tono con armonicos y envolvente: no es voz, pero comparte con ella lo que
    aqui importa — energia concentrada en pocas bandas y amplitud variable."""
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False, dtype=np.float32)
    señal = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((220, 440, 880)))
    envolvente = (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
    return (señal * envolvente * 0.3).astype(np.float32)


def test_stft_reconstruye_exactamente() -> None:
    """La propiedad que estaba rota. Con la Hann simetrica de `np.hanning` la
    suma de ventanas solapadas no es constante y el error maximo era 4.6e-2:
    distorsion audible metida por el propio "limpiador"."""
    audio = _voz_sintetica()
    spectrum, length = _stft(audio)
    assert np.max(np.abs(_istft(spectrum, length) - audio)) < 1e-5


def test_la_longitud_se_conserva() -> None:
    """Recortar o alargar la frase desalinearia el preroll y se comeria la
    primera silaba, que es justo lo que el preroll existe para evitar."""
    for muestras in (N_FFT, 5000, 16_000, 35_049):
        audio = np.random.default_rng(0).normal(0, 0.1, muestras).astype(np.float32)
        assert denoise(audio).size == muestras


def test_audio_muy_corto_pasa_sin_tocar() -> None:
    """Por debajo de una ventana no hay STFT posible. Devolver el audio tal cual
    es preferible a devolver ceros."""
    corto = np.ones(N_FFT - 1, dtype=np.float32)
    assert np.array_equal(denoise(corto), corto)


def test_no_estropea_audio_limpio() -> None:
    """Sin ruido de fondo la ganancia es ~1 y la senal sale casi como entro.

    El margen es del 5% y no de una millonesima porque esta senal es el PEOR
    caso posible para el algoritmo: un tono sostenido es, por definicion,
    estacionario, que es exactamente lo que el estimador llama ruido. Se lo come
    un poco y hace bien.

    Con voz de verdad no pasa: medido sobre habla real, el denoise cambia el RMS
    en 2e-8, o sea nada. La voz no es estacionaria y por eso sobrevive.
    """
    audio = _voz_sintetica()
    cambio = np.sqrt(np.mean((denoise(audio) - audio) ** 2))
    assert cambio / np.sqrt(np.mean(audio**2)) < 0.05


def test_atenua_el_ruido_estacionario() -> None:
    """Lo que se le pide: con un siseo constante encima, la salida se parece mas
    al original limpio que la entrada sucia."""
    limpio = _voz_sintetica()
    rng = np.random.default_rng(0)
    sucio = limpio + rng.normal(0, 0.05, len(limpio)).astype(np.float32)

    error_antes = np.sqrt(np.mean((sucio - limpio) ** 2))
    error_despues = np.sqrt(np.mean((denoise(sucio) - limpio) ** 2))
    assert error_despues < error_antes


def test_el_suelo_impide_silencios_absolutos() -> None:
    """`floor` no es un ajuste fino: los huecos totales son lo que dispara las
    alucinaciones de Whisper. Con solo ruido, algo tiene que sobrevivir."""
    rng = np.random.default_rng(0)
    solo_ruido = rng.normal(0, 0.05, 16_000).astype(np.float32)
    salida = denoise(solo_ruido, floor=0.15)
    assert np.max(np.abs(salida)) > 0.0
    # Y aun asi tiene que haber atenuado de verdad.
    assert np.sqrt(np.mean(salida**2)) < np.sqrt(np.mean(solo_ruido**2))


def test_estimate_noise_toma_el_suelo_no_el_pico() -> None:
    """La estimacion se basa en que la voz es intermitente y el ruido no: en
    cada banda, los instantes mas flojos de la frase son ruido puro."""
    magnitudes = np.array([[1.0, 10.0], [1.0, 0.5], [1.0, 0.6], [1.0, 20.0]])
    ruido = estimate_noise(magnitudes, percentile=15.0)
    assert ruido[0] == pytest.approx(1.0)      # banda constante: todo es ruido
    assert ruido[1] < 1.0                      # banda con picos: se queda con el suelo


def test_snr_distingue_una_frase_de_puro_ruido() -> None:
    """Es la medida que decide si la grabacion merece el coste del STT."""
    limpio = _voz_sintetica()
    rng = np.random.default_rng(0)
    assert snr_db(limpio) > snr_db(rng.normal(0, 0.05, len(limpio)).astype(np.float32))
