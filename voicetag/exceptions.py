"""Custom exception hierarchy for the voicetag library.

All exceptions inherit from ``VoiceTagError`` so callers can catch broadly
(``except VoiceTagError``) or narrowly (``except AudioLoadError``).
"""

from __future__ import annotations


class VoiceTagError(Exception):
    """Base exception for all voicetag errors."""

    def __init__(self, message: str = "An error occurred in voicetag.") -> None:
        self.message = message
        super().__init__(self.message)


class VoiceTagConfigError(VoiceTagError):
    """Raised for invalid configuration values or missing authentication.

    Examples:
        - Missing HuggingFace token when pyannote access is required.
        - Invalid device string.
        - Out-of-range threshold values.

    Hint:
        If you need a HuggingFace token, set ``hf_token`` in your
        ``VoiceTagConfig``, export the ``HF_TOKEN`` environment variable,
        or visit https://huggingface.co/settings/tokens to create one.
    """

    def __init__(
        self,
        message: str = (
            "Invalid voicetag configuration. "
            "If a HuggingFace token is required, set hf_token in config, "
            "the HF_TOKEN env var, or see https://huggingface.co/settings/tokens"
        ),
    ) -> None:
        super().__init__(message)


class EnrollmentError(VoiceTagError):
    """Raised when speaker enrollment fails.

    Common causes:
        - No valid audio files provided for enrollment.
        - Speaker name already exists (when duplicates are disallowed).
        - Audio files are too short to produce a reliable embedding.
    """

    def __init__(
        self,
        message: str = "Speaker enrollment failed.",
    ) -> None:
        super().__init__(message)


class DiarizationError(VoiceTagError):
    """Raised when pyannote diarization fails.

    Common causes:
        - Authentication failure (HTTP 401) — check your HuggingFace token.
        - Model download failure — ensure internet access and that you have
          accepted the pyannote model license on HuggingFace.
        - Processing error on the audio file.

    Hint:
        Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
        and ensure your token has the appropriate permissions.
    """

    def __init__(
        self,
        message: str = "Diarization failed.",
    ) -> None:
        super().__init__(message)


class AudioLoadError(VoiceTagError):
    """Raised when an audio file cannot be loaded.

    Common causes:
        - File not found at the specified path.
        - Unsupported audio format (supported: wav, mp3, flac, ogg, m4a).
        - Corrupted or unreadable audio file.

    Hint:
        Ensure the file exists, is in a supported format, and is not
        corrupted. Supported formats: .wav, .mp3, .flac, .ogg, .m4a
    """

    def __init__(
        self,
        message: str = (
            "Failed to load audio file. " "Supported formats: wav, mp3, flac, ogg, m4a"
        ),
    ) -> None:
        super().__init__(message)


class TranscriptionError(VoiceTagError):
    """Raised when speech-to-text transcription fails."""

    def __init__(
        self,
        message: str = "Transcription failed.",
    ) -> None:
        super().__init__(message)
