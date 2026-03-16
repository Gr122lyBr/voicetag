"""Fireworks AI transcription provider."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from voicetag.exceptions import TranscriptionError
from voicetag.transcriber import BaseTranscriber


class FireworksTranscriber(BaseTranscriber):
    """Transcribe audio using Fireworks AI's Whisper API.

    Requires: pip install voicetag[fireworks]
    Env var: FIREWORKS_API_KEY
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-v3",
    ) -> None:
        self._api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self._api_key:
            raise TranscriptionError(
                "Fireworks API key required. Set api_key parameter or "
                "FIREWORKS_API_KEY environment variable."
            )
        self._model = model

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        try:
            import httpx
        except ImportError:
            raise TranscriptionError("httpx not installed. Run: pip install voicetag[fireworks]")

        wav_bytes = self._audio_to_wav_bytes(audio, sr)

        try:
            url = (
                "https://audio-turbo.us-virginia-1.direct.fireworks.ai" "/v1/audio/transcriptions"
            )
            headers = {"Authorization": f"Bearer {self._api_key}"}
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            data: dict = {"model": self._model}
            if language:
                data["language"] = language

            response = httpx.post(url, headers=headers, files=files, data=data, timeout=60.0)
            response.raise_for_status()
            return response.json()["text"].strip()
        except Exception as exc:
            raise TranscriptionError(f"Fireworks transcription failed: {exc}") from exc
