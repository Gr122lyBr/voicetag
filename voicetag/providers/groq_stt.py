"""Groq Whisper API transcription provider."""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import numpy as np

from voicetag.exceptions import TranscriptionError
from voicetag.transcriber import BaseTranscriber


class GroqTranscriber(BaseTranscriber):
    """Transcribe audio using Groq's Whisper API.

    Requires: pip install voicetag[groq]
    Env var: GROQ_API_KEY
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3",
    ) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise TranscriptionError(
                "Groq API key required. Set api_key parameter or "
                "GROQ_API_KEY environment variable."
            )
        self._model = model

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        try:
            from groq import Groq
        except ImportError:
            raise TranscriptionError("Groq SDK not installed. Run: pip install voicetag[groq]")

        client = Groq(api_key=self._api_key)
        tmp_path = self._audio_to_temp_file(audio, sr)

        try:
            with open(tmp_path, "rb") as f:
                kwargs: dict = {
                    "model": self._model,
                    "file": ("audio.wav", f.read(), "audio/wav"),
                }
                if language:
                    kwargs["language"] = language
                response = client.audio.transcriptions.create(**kwargs)
            result: str = response.text.strip()
            return result
        except Exception as exc:
            raise TranscriptionError(f"Groq transcription failed: {exc}") from exc
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)
