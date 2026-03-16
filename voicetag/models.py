"""Pydantic v2 data models used throughout the voicetag library.

All models use strict validation. Segment models are frozen (immutable)
to prevent accidental mutation of pipeline results.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VoiceTagConfig(BaseModel):
    """Configuration for the voicetag pipeline.

    Attributes:
        hf_token: HuggingFace API token for pyannote model access.
            Falls back to the ``HF_TOKEN`` environment variable if not set.
        similarity_threshold: Minimum cosine similarity to consider a
            speaker match. Segments below this are labelled ``"UNKNOWN"``.
        overlap_threshold: Minimum overlap ratio to flag a region as
            overlapping speech.
        max_workers: Number of threads for parallel embedding computation.
        min_segment_duration: Segments shorter than this (seconds) are
            discarded as unreliable.
        device: Torch device for model inference (``"cpu"``, ``"cuda"``,
            or ``"mps"``).
    """

    model_config = ConfigDict(frozen=True)

    hf_token: Optional[str] = Field(default=None)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_workers: int = Field(default=4, ge=1)
    min_segment_duration: float = Field(default=0.5, ge=0.0)
    device: str = Field(default="cpu")

    @model_validator(mode="before")
    @classmethod
    def _resolve_hf_token(cls, values: dict) -> dict:
        """Resolve hf_token from environment if not explicitly provided."""
        if isinstance(values, dict):
            token = values.get("hf_token")
            if token is None:
                token = os.environ.get("HF_TOKEN")
            if token is not None:
                values["hf_token"] = token
        return values


class SpeakerSegment(BaseModel):
    """A single speaker segment with timing and identification info.

    Attributes:
        speaker: Name of the identified speaker, or ``"UNKNOWN"``.
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        confidence: Cosine similarity score for the speaker match.
    """

    model_config = ConfigDict(frozen=True)

    speaker: str
    start: float
    end: float
    confidence: float = Field(default=0.0)

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        """Validate that end time is strictly after start time."""
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"end ({v}) must be greater than start ({start})")
        return v

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start


class OverlapSegment(BaseModel):
    """A region where multiple speakers talk simultaneously.

    Attributes:
        speakers: List of speaker names involved in the overlap.
        start: Overlap start time in seconds.
        end: Overlap end time in seconds.
        speaker: Literal label, always ``"OVERLAP"``.
    """

    model_config = ConfigDict(frozen=True)

    speakers: list[str]
    start: float
    end: float
    speaker: Literal["OVERLAP"] = "OVERLAP"

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        """Validate that end time is strictly after start time."""
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"end ({v}) must be greater than start ({start})")
        return v

    @property
    def duration(self) -> float:
        """Duration of the overlap region in seconds."""
        return self.end - self.start


class SpeakerProfile(BaseModel):
    """An enrolled speaker's embedding profile.

    Attributes:
        name: Speaker name.
        embedding: Mean embedding vector (256-dimensional for resemblyzer).
        num_samples: Number of audio files used for enrollment.
        created_at: Timestamp when the profile was created.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    embedding: list[float]
    num_samples: int = Field(default=1, ge=1)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class TranscriptSegment(BaseModel):
    """A speaker segment with transcribed text."""

    model_config = ConfigDict(frozen=True)

    speaker: str
    start: float
    end: float
    text: str
    confidence: float = 0.0

    @model_validator(mode="after")
    def _validate_times(self) -> TranscriptSegment:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start})")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class TranscriptResult(BaseModel):
    """Full transcription result with speaker-attributed text."""

    model_config = ConfigDict(frozen=True)

    segments: list[TranscriptSegment]
    audio_duration: float
    num_speakers: int
    processing_time: float = 0.0

    @property
    def full_transcript(self) -> str:
        """Return the full transcript as formatted text."""
        lines = []
        for seg in self.segments:
            lines.append(f"[{seg.speaker}] {seg.text}")
        return "\n".join(lines)

    @property
    def by_speaker(self) -> dict[str, list[TranscriptSegment]]:
        """Group segments by speaker."""
        result: dict[str, list[TranscriptSegment]] = {}
        for seg in self.segments:
            result.setdefault(seg.speaker, []).append(seg)
        return result


class DiarizationResult(BaseModel):
    """Complete result of a speaker identification pipeline run.

    Attributes:
        segments: Ordered list of speaker and overlap segments.
        audio_duration: Total duration of the input audio in seconds.
        num_speakers: Number of distinct speakers detected.
        processing_time: Wall-clock time for the pipeline run in seconds.
    """

    model_config = ConfigDict(frozen=True)

    segments: list[Union[SpeakerSegment, OverlapSegment]]
    audio_duration: float
    num_speakers: int
    processing_time: float = Field(default=0.0)
