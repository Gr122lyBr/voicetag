"""OpenAI Whisper API transcription provider."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from voicetag.exceptions import TranscriptionError
from voicetag.transcriber import BaseTranscriber


class OpenAITranscriber(BaseTranscriber):
    """Transcribe audio using OpenAI's Whisper API.

    Requires: pip install voicetag[openai]
    Env var: OPENAI_API_KEY
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise TranscriptionError(
                "OpenAI API key required. Set api_key parameter or "
                "OPENAI_API_KEY environment variable."
            )
        self._model = model

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise TranscriptionError("OpenAI SDK not installed. Run: pip install voicetag[openai]")

        client = OpenAI(api_key=self._api_key)
        wav_bytes = self._audio_to_wav_bytes(audio, sr)

        try:
            kwargs: dict = {
                "model": self._model,
                "file": ("audio.wav", wav_bytes, "audio/wav"),
            }
            if language:
                kwargs["language"] = language

            response = client.audio.transcriptions.create(**kwargs)
            return response.text.strip()
        except Exception as exc:
            raise TranscriptionError(f"OpenAI transcription failed: {exc}") from exc
