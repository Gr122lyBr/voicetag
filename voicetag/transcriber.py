"""Speech-to-text transcription with pluggable providers."""

from __future__ import annotations

import importlib
import io
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import soundfile as sf

from voicetag.exceptions import TranscriptionError


class BaseTranscriber(ABC):
    """Abstract base for STT providers."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: 1-D float32 audio waveform.
            sr: Sample rate.
            language: Optional ISO language code (e.g., "en", "he").

        Returns:
            Transcribed text string.
        """
        ...

    def _audio_to_wav_bytes(self, audio: np.ndarray, sr: int) -> bytes:
        """Convert numpy audio to WAV bytes for API upload."""
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)
        return buf.read()

    def _audio_to_temp_file(self, audio: np.ndarray, sr: int) -> str:
        """Write audio to a temporary WAV file, return path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, sr, format="WAV")
        return tmp.name


# Provider registry: name -> (module_path, class_name)
_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("voicetag.providers.openai_stt", "OpenAITranscriber"),
    "groq": ("voicetag.providers.groq_stt", "GroqTranscriber"),
    "fireworks": ("voicetag.providers.fireworks_stt", "FireworksTranscriber"),
    "whisper": ("voicetag.providers.whisper_local", "WhisperLocalTranscriber"),
    "deepgram": ("voicetag.providers.deepgram_stt", "DeepgramTranscriber"),
}


def get_transcriber(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> BaseTranscriber:
    """Get a transcriber instance by provider name.

    Args:
        provider: Provider name ("openai", "groq", "fireworks", "whisper", "deepgram").
        api_key: API key (falls back to provider-specific env var).
        model: Model name override.
        **kwargs: Additional provider-specific arguments.

    Returns:
        A BaseTranscriber instance.

    Raises:
        TranscriptionError: If provider is unknown or SDK not installed.
    """
    if provider not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise TranscriptionError(f"Unknown STT provider '{provider}'. Available: {available}")

    module_path, class_name = _PROVIDERS[provider]

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise TranscriptionError(
            f"Provider '{provider}' requires additional dependencies. "
            f"Install with: pip install voicetag[{provider}]"
        ) from exc

    cls = getattr(module, class_name)

    init_kwargs: dict = {}
    if api_key is not None:
        init_kwargs["api_key"] = api_key
    if model is not None:
        init_kwargs["model"] = model
    init_kwargs.update(kwargs)

    instance: BaseTranscriber = cls(**init_kwargs)
    return instance


def available_providers() -> list[str]:
    """Return list of registered provider names."""
    return sorted(_PROVIDERS.keys())
