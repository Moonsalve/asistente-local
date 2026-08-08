"""Encoder de frases para el matching semantico.

Se usa `onnxruntime` + `tokenizers` directamente en lugar de
`sentence-transformers` porque este ultimo arrastra PyTorch entero (~2.5 GB de
instalacion y ~1 s extra de import). Aqui solo necesitamos un forward pass y un
mean pooling: no hace falta el framework de entrenamiento.

Modelo por defecto: `intfloat/multilingual-e5-small` (384 dims, ~0.12 GB en
fp32). Es multilingue de verdad, entrenado con pares de parafrasis, que es
exactamente la tarea que tenemos. El prefijo "query: " no es opcional: e5 se
entreno con el, y omitirlo degrada la similitud de forma notable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# e5 espera este prefijo. Para tareas simetricas (frase vs frase, que es nuestro
# caso) la documentacion del modelo indica usar "query: " en ambos lados.
_E5_PREFIX = "query: "


@runtime_checkable
class Embedder(Protocol):
    """Contrato minimo del encoder, para poder inyectar un doble en los tests."""

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Devuelve una matriz (len(texts), dim) L2-normalizada por filas."""
        ...


class OnnxEmbedder:
    """Encoder e5 sobre ONNX Runtime."""

    def __init__(
        self,
        repo_id: str = "intfloat/multilingual-e5-small",
        onnx_file: str = "onnx/model.onnx",
        *,
        use_gpu: bool = False,
        max_length: int = 64,
        cache_dir: Path | None = None,
    ) -> None:
        # Imports perezosos: mantienen el arranque del proceso rapido cuando el
        # router se usa desde tests con un embedder falso.
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        cache = str(cache_dir) if cache_dir is not None else None
        model_path = hf_hub_download(repo_id, onnx_file, cache_dir=cache)
        tokenizer_path = hf_hub_download(repo_id, "tokenizer.json", cache_dir=cache)

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # Truncar a 64 tokens: los comandos de voz son cortos y el coste del
        # forward pass crece con la longitud de secuencia.
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, opts, providers=providers)
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._dim = int(self._session.get_outputs()[0].shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        encodings = self._tokenizer.encode_batch([_E5_PREFIX + t for t in texts])
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # XLM-R no usa token_type_ids, pero algunos exports los declaran igual.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        hidden = self._session.run(None, feeds)[0]  # (batch, seq, dim)
        return _mean_pool_and_normalize(hidden, attention_mask)

    def warmup(self) -> None:
        """Fuerza la primera inferencia, que es la cara (alocacion de buffers)."""
        self.encode(["calentando el modelo"])


def _mean_pool_and_normalize(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling ignorando el padding, seguido de normalizacion L2.

    Normalizar aqui permite que la similitud coseno en runtime sea un simple
    producto matricial `catalogo @ consulta`, sin dividir por normas.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = summed / counts
    norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), a_min=1e-9, a_max=None)
    return (pooled / norms).astype(np.float32)
