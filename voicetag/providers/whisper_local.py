"""Local Whisper model transcription provider."""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from voicetag.exceptions import TranscriptionError
from voicetag.transcriber import BaseTranscriber


class WhisperLocalTranscriber(BaseTranscriber):
    """Transcribe audio using a local OpenAI Whisper model.

    Requires: pip install voicetag[whisper]
    No API key needed — runs locally.
    """

    def __init__(
        self,
        model: str = "base",
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._model_name = model
        self._device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import whisper
        except ImportError:
            raise TranscriptionError(
                "OpenAI Whisper not installed. Run: pip install voicetag[whisper]"
            )
        logger.debug("Loading local Whisper model '{}'", self._model_name)
        self._model = whisper.load_model(self._model_name, device=self._device)
        logger.info("Whisper model '{}' loaded", self._model_name)

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        self._ensure_loaded()

        try:
            audio_float32 = audio.astype(np.float32)
            kwargs: dict = {"fp16": False}
            if language:
                kwargs["language"] = language
            result = self._model.transcribe(audio_float32, **kwargs)
            return result["text"].strip()
        except Exception as exc:
            raise TranscriptionError(f"Local Whisper transcription failed: {exc}") from exc
