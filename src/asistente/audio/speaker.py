"""Verificacion de locutor: responder solo a la voz enrolada.

QUE PROBLEMA RESUELVE Y CUAL NO
-------------------------------
Resuelve el ruido que *habla*: la television, la musica cantada, otra persona en
la habitacion. Contra esos, ni el VAD ni el denoise sirven de nada -son voz de
verdad- y la palabra clave solo protege a medias, porque si alguien dice "Apolo"
el asistente obedece.

NO resuelve el ventilador ni el zumbido. Para eso esta `denoise.py`, que es mas
barato y va antes.

COMO FUNCIONA
-------------
Un modelo de embeddings de locutor (ResNet34 de WeSpeaker, 256 dimensiones,
entrenado con VoxCeleb2) convierte cada frase en un vector que depende de QUIEN
habla y no de QUE dice. Se compara por coseno con el centroide de las frases
enroladas.

El modelo NO come audio crudo: espera fbank de Kaldi de 80 bandas. Esas features
se sacan con `kaldi-native-fbank`, que es la implementacion en C++ del propio
Kaldi empaquetada como wheel. Se usa esa y no una version casera en numpy a
proposito: un fbank *casi* correcto no falla, produce embeddings plausibles pero
que no discriminan, y el sintoma seria "a veces no me reconoce" — imposible de
depurar.

Detalles que el modelo da por supuestos y que hay que respetar:
  - muestras en escala de int16 (-32768..32767), no en -1..1
  - `dither = 0` para que dos pasadas del mismo audio den lo mismo
  - normalizacion de media por dimension (CMN) sobre la frase entera

EL FALLO MAS CARO ES EL FALSO NEGATIVO
--------------------------------------
Si esto rechaza a su dueno, el asistente se queda sordo y no hay forma obvia de
darse cuenta de por que. Por eso:
  - viene desactivado hasta que hay una voz enrolada,
  - el umbral por defecto es permisivo,
  - las frases demasiado cortas para ser fiables se ACEPTAN sin verificar,
  - cada rechazo se registra con su coseno, para poder ajustar el umbral con
    datos en vez de a ojo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Repo de HuggingFace con el ONNX exportado por el propio proyecto WeSpeaker.
DEFAULT_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
DEFAULT_FILE = "voxceleb_resnet34_LM.onnx"

#: Por debajo de esto no hay material suficiente para un embedding fiable. Es
#: una limitacion del modelo, no un ajuste: con media palabra el vector depende
#: mas del fonema que del locutor.
MIN_SECONDS = 0.6


class SpeakerError(RuntimeError):
    pass


def _fbank(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Fbank de Kaldi de 80 bandas, tal y como lo espera WeSpeaker."""
    try:
        import kaldi_native_fbank as knf
    except ImportError as exc:  # pragma: no cover - depende de la instalacion
        raise SpeakerError(
            "falta kaldi-native-fbank; instala con: pip install -e ."
        ) from exc

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sample_rate)
    # Sin dither: es ruido aleatorio que Kaldi anade para evitar log(0). Aqui
    # haria que el mismo audio diera embeddings distintos en cada pasada.
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80

    fbank = knf.OnlineFbank(opts)
    # Escala de int16: el modelo se entreno sobre wavs sin normalizar.
    fbank.accept_waveform(float(sample_rate), (audio * 32768.0).tolist())
    fbank.input_finished()

    frames = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
    if not frames:
        raise SpeakerError("audio demasiado corto para extraer features")
    feats = np.asarray(frames, dtype=np.float32)
    # CMN: resta la media de cada banda a lo largo de la frase. Cancela el color
    # del microfono y de la sala, que si no acabarian dentro del "quien habla".
    return feats - feats.mean(axis=0, keepdims=True)


class SpeakerEmbedder:
    """Frase de audio -> vector de 256 dimensiones normalizado."""

    def __init__(self, model_path: Path | str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )

    def encode(self, audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        feats = _fbank(audio.astype(np.float32), sample_rate)[None, :, :]
        (embedding,) = self._session.run(None, {"feats": feats})
        vector = embedding[0].astype(np.float32)
        # Normalizado para que el producto punto SEA el coseno.
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else vector


def download_model(cache_dir: Path | str | None = None) -> Path:
    """Descarga el ONNX (~26 MB) la primera vez y devuelve su ruta."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(DEFAULT_REPO, DEFAULT_FILE, cache_dir=str(cache_dir) if cache_dir else None)
    )


@dataclass(frozen=True, slots=True)
class Verdict:
    """Resultado de verificar una frase. `score` se registra siempre, tambien
    cuando se acepta: es lo unico que permite ajustar el umbral con datos."""

    accepted: bool
    score: float
    reason: str


class SpeakerGate:
    """Acepta o rechaza una frase comparandola con la voz enrolada."""

    def __init__(
        self,
        embedder: SpeakerEmbedder,
        centroid: np.ndarray,
        threshold: float,
        min_seconds: float = MIN_SECONDS,
    ) -> None:
        self._embedder = embedder
        self._centroid = centroid
        self._threshold = threshold
        self._min_seconds = min_seconds

    @classmethod
    def from_profile(
        cls, embedder: SpeakerEmbedder, profile_path: Path | str, threshold: float
    ) -> SpeakerGate:
        data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        centroid = np.asarray(data["centroid"], dtype=np.float32)
        return cls(embedder, centroid, threshold)

    def check(self, audio: np.ndarray, sample_rate: int = 16_000) -> Verdict:
        duration = len(audio) / sample_rate
        if duration < self._min_seconds:
            # Se acepta a proposito: con menos de medio segundo el embedding
            # dice mas del fonema que del locutor, y rechazar por un vector poco
            # fiable es peor que dejar pasar la frase a la palabra clave.
            return Verdict(True, 0.0, f"frase corta ({duration:.2f}s), sin verificar")

        try:
            score = float(np.dot(self._embedder.encode(audio, sample_rate), self._centroid))
        except SpeakerError as exc:
            # Ante la duda, oir. Quedarse sordo por un fallo del extractor seria
            # el peor resultado posible.
            log.warning("no se pudo verificar el locutor (%s); se acepta", exc)
            return Verdict(True, 0.0, "fallo al extraer el embedding")

        if score < self._threshold:
            return Verdict(False, score, f"otra voz ({score:.3f} < {self._threshold:.2f})")
        return Verdict(True, score, f"voz reconocida ({score:.3f})")


def build_profile(
    embedder: SpeakerEmbedder, samples: list[np.ndarray], sample_rate: int = 16_000
) -> np.ndarray:
    """Centroide normalizado de varias frases de la misma persona.

    Promediar varias frases y no quedarse con una es lo que hace el umbral
    estable: un solo embedding arrastra la entonacion y el contenido de esa
    frase concreta.
    """
    vectors = [embedder.encode(s, sample_rate) for s in samples]
    if not vectors:
        raise SpeakerError("no hay muestras para construir el perfil")
    centroid = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-9:
        raise SpeakerError("las muestras no produjeron un centroide utilizable")
    return (centroid / norm).astype(np.float32)
