"""voicetag — Speaker identification powered by pyannote and resemblyzer."""

from __future__ import annotations

from loguru import logger

from voicetag.exceptions import (  # noqa: E402
    AudioLoadError,
    DiarizationError,
    EnrollmentError,
    TranscriptionError,
    VoiceTagConfigError,
    VoiceTagError,
)
from voicetag.models import (  # noqa: E402
    DiarizationResult,
    OverlapSegment,
    SpeakerProfile,
    SpeakerSegment,
    TranscriptResult,
    TranscriptSegment,
    VoiceTagConfig,
)
from voicetag.pipeline import Pipeline as VoiceTag  # noqa: E402
from voicetag.transcriber import available_providers  # noqa: E402

logger.disable("voicetag")

__version__ = "0.1.1"

__all__ = [
    "VoiceTag",
    "VoiceTagConfig",
    "SpeakerSegment",
    "OverlapSegment",
    "SpeakerProfile",
    "DiarizationResult",
    "TranscriptSegment",
    "TranscriptResult",
    "VoiceTagError",
    "VoiceTagConfigError",
    "EnrollmentError",
    "DiarizationError",
    "AudioLoadError",
    "TranscriptionError",
    "available_providers",
    "__version__",
]
