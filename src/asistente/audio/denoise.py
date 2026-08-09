"""Supresion de ruido estacionario antes de transcribir.

QUE ATACA
---------
El ruido que no cambia: ventilador, aire acondicionado, zumbido de la fuente,
siseo del microfono. Es exactamente el que mas dano hace, porque esta SIEMPRE y
porque Whisper, entrenado con audio limpio, empieza a inventar texto cuando solo
oye eso.

No ataca la musica ni las voces de fondo: eso cambia todo el rato y no se puede
resumir en un espectro fijo. Para eso esta la verificacion de locutor.

COMO
----
Resta espectral con mascara suave:

    1. STFT de la frase (ventana de Hann, solape del 75%).
    2. Se estima el ruido como un percentil bajo de cada banda a lo largo del
       tiempo. La idea: en una banda concreta, el minimo que se ve durante la
       frase es ruido, porque la voz es intermitente y el ventilador no.
    3. Se calcula una ganancia por celda tiempo-frecuencia y se aplica a la
       magnitud, conservando la fase.
    4. ISTFT por solapamiento y suma.

POR QUE UNA MASCARA SUAVE Y NO UNA RESTA DURA
---------------------------------------------
Restar magnitudes y recortar en cero produce "ruido musical": trocitos de
espectro que sobreviven aislados y suenan como campanitas. A Whisper le sientan
peor que el ruido original, porque no se parecen a nada que haya oido. La
ganancia continua tipo Wiener con un suelo deja pasar algo de ruido a cambio de
no crear artefactos.

EL SUELO NO ES NEGOCIABLE
-------------------------
`floor` impide que una banda se lleve a cero. Suprimir del todo suena mas limpio
al oido y transcribe PEOR: los huecos absolutos son la senal que dispara las
alucinaciones. Se busca audio con menos ruido, no audio sin ruido.

Solo numpy: para una frase de dos segundos son unos pocos milisegundos, muy por
debajo de lo que cuesta el STT.
"""

from __future__ import annotations

import numpy as np

#: Ventana de la STFT. 512 muestras = 32 ms a 16 kHz: suficiente resolucion en
#: frecuencia para separar bandas de voz del zumbido, y suficientemente corta
#: para no emborronar las consonantes.
N_FFT = 512
#: Solape del 75%. Con Hann, la suma de ventanas es constante y la
#: reconstruccion es exacta cuando la ganancia es 1.
HOP = N_FFT // 4


def _window() -> np.ndarray:
    """Hann PERIODICA, no la simetrica de `np.hanning`.

    Con la simetrica la suma de ventanas solapadas no es constante y la
    reconstruccion no es exacta: medido, un error maximo de 4.6e-2 frente a
    ~1e-7. Ese error es distorsion anadida a la senal *antes* de que Whisper la
    vea, o sea justo lo contrario de lo que este modulo pretende.
    """
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def _stft(audio: np.ndarray) -> tuple[np.ndarray, int]:
    # Se rellena por delante y por detras para que TODA muestra real quede
    # cubierta por el mismo numero de ventanas; si no, los bordes salen
    # atenuados y se come el principio de la primera palabra.
    padded = np.concatenate(
        [np.zeros(N_FFT, dtype=np.float32), audio, np.zeros(2 * N_FFT, dtype=np.float32)]
    )
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, N_FFT),
        strides=(padded.strides[0] * HOP, padded.strides[0]),
        writeable=False,
    )
    return np.fft.rfft(frames * _window(), axis=1), len(audio)


def _istft(spectrum: np.ndarray, length: int) -> np.ndarray:
    window = _window()
    frames = np.fft.irfft(spectrum, n=N_FFT, axis=1).astype(np.float32) * window

    out = np.zeros((len(frames) - 1) * HOP + N_FFT, dtype=np.float32)
    norm = np.zeros_like(out)
    for i, frame in enumerate(frames):
        out[i * HOP : i * HOP + N_FFT] += frame
        norm[i * HOP : i * HOP + N_FFT] += window**2
    # Donde la suma de ventanas es ~0 no hay informacion; dividir ahi amplifica
    # ruido numerico hasta hacerlo audible.
    np.divide(out, norm, out=out, where=norm > 1e-6)
    return out[N_FFT : N_FFT + length]


def estimate_noise(magnitude: np.ndarray, percentile: float = 15.0) -> np.ndarray:
    """Espectro del ruido: percentil bajo de cada banda a lo largo del tiempo.

    Funciona porque la voz es intermitente y el ruido de fondo no. En una banda
    dada, los instantes mas silenciosos de la frase son ruido puro, y el preroll
    garantiza que siempre hay algunos: son los 300 ms anteriores a que el VAD
    detectara habla.
    """
    return np.percentile(magnitude, percentile, axis=0)


def denoise(
    audio: np.ndarray,
    strength: float = 1.5,
    floor: float = 0.15,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Atenua el ruido estacionario de una frase ya grabada.

    `strength` sobre-resta: por encima de 1 se quita mas de lo estimado, para
    compensar que el percentil se queda corto. Pasarse produce voz con agujeros.
    `floor` es la atenuacion maxima por banda (0.15 = -16 dB), y existe para no
    dejar silencios absolutos, que es lo que hace alucinar a Whisper.
    """
    if audio.size < N_FFT:
        return audio

    spectrum, length = _stft(audio.astype(np.float32))
    magnitude = np.abs(spectrum)

    noise_mag = estimate_noise(magnitude) if noise is None else noise
    # Ganancia tipo Wiener sobre potencias. El +1e-10 evita dividir por cero en
    # bandas completamente vacias (audio digital sin nada de senal).
    power, noise_power = magnitude**2, (strength * noise_mag) ** 2
    gain = np.sqrt(np.clip((power - noise_power) / (power + 1e-10), 0.0, 1.0))
    gain = np.maximum(gain, floor)

    return _istft(spectrum * gain, length)


def snr_db(audio: np.ndarray) -> float:
    """SNR aproximada de la frase, en dB.

    Compara la energia de los instantes mas fuertes con la de los mas flojos.
    No es una medida rigurosa -haria falta saber que trozo es voz- pero es
    monotona con la calidad real y basta para decidir si merece la pena gastar
    el STT en esta grabacion.
    """
    if audio.size < N_FFT:
        return 0.0
    magnitude = np.abs(_stft(audio.astype(np.float32))[0])
    energy = magnitude.sum(axis=1)
    if energy.size == 0:
        return 0.0
    speech = float(np.percentile(energy, 90))
    noise = float(np.percentile(energy, 10))
    if noise <= 1e-9 or speech <= 1e-9:
        return 0.0
    return float(20.0 * np.log10(speech / noise))
