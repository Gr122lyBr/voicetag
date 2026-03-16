"""Deepgram transcription provider."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from voicetag.exceptions import TranscriptionError
from voicetag.transcriber import BaseTranscriber


class DeepgramTranscriber(BaseTranscriber):
    """Transcribe audio using Deepgram's API.

    Requires: pip install voicetag[deepgram]
    Env var: DEEPGRAM_API_KEY
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-2",
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self._api_key:
            raise TranscriptionError(
                "Deepgram API key required. Set api_key parameter or "
                "DEEPGRAM_API_KEY environment variable."
            )
        self._model = model

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        try:
            from deepgram import DeepgramClient, FileSource, PrerecordedOptions
        except ImportError:
            raise TranscriptionError(
                "Deepgram SDK not installed. Run: pip install voicetag[deepgram]"
            )

        wav_bytes = self._audio_to_wav_bytes(audio, sr)

        try:
            client = DeepgramClient(self._api_key)
            payload: FileSource = {"buffer": wav_bytes}
            options = PrerecordedOptions(model=self._model, smart_format=True)
            if language:
                options.language = language

            response = client.listen.rest.v("1").transcribe_file(payload, options)
            return response.results.channels[0].alternatives[0].transcript.strip()
        except Exception as exc:
            raise TranscriptionError(f"Deepgram transcription failed: {exc}") from exc
