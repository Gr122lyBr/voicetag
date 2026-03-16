"""Tests for voicetag.models — Pydantic v2 data models."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from voicetag.models import (
    DiarizationResult,
    OverlapSegment,
    SpeakerProfile,
    SpeakerSegment,
    TranscriptResult,
    TranscriptSegment,
    VoiceTagConfig,
)

# ---------------------------------------------------------------------------
# VoiceTagConfig
# ---------------------------------------------------------------------------


class TestVoiceTagConfig:
    def test_defaults(self):
        cfg = VoiceTagConfig()
        assert cfg.similarity_threshold == 0.75
        assert cfg.overlap_threshold == 0.5
        assert cfg.max_workers == 4
        assert cfg.min_segment_duration == 0.5
        assert cfg.device == "cpu"
        assert cfg.hf_token is None

    def test_env_var_resolution(self):
        with patch.dict("os.environ", {"HF_TOKEN": "env-test-token"}):
            cfg = VoiceTagConfig()
        assert cfg.hf_token == "env-test-token"

    def test_explicit_token_overrides_env(self):
        with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
            cfg = VoiceTagConfig(hf_token="explicit-token")
        assert cfg.hf_token == "explicit-token"

    def test_threshold_lower_bound(self):
        with pytest.raises(ValidationError):
            VoiceTagConfig(similarity_threshold=-0.1)

    def test_threshold_upper_bound(self):
        with pytest.raises(ValidationError):
            VoiceTagConfig(similarity_threshold=1.1)

    def test_overlap_threshold_bounds(self):
        with pytest.raises(ValidationError):
            VoiceTagConfig(overlap_threshold=-0.01)
        with pytest.raises(ValidationError):
            VoiceTagConfig(overlap_threshold=1.5)

    def test_max_workers_must_be_positive(self):
        with pytest.raises(ValidationError):
            VoiceTagConfig(max_workers=0)

    def test_frozen(self):
        cfg = VoiceTagConfig()
        with pytest.raises(ValidationError):
            cfg.device = "cuda"


# ---------------------------------------------------------------------------
# SpeakerSegment
# ---------------------------------------------------------------------------


class TestSpeakerSegment:
    def test_valid_segment(self):
        seg = SpeakerSegment(speaker="alice", start=1.0, end=3.0, confidence=0.85)
        assert seg.speaker == "alice"
        assert seg.start == 1.0
        assert seg.end == 3.0
        assert seg.confidence == 0.85

    def test_end_must_be_after_start(self):
        with pytest.raises(ValidationError, match="must be greater than start"):
            SpeakerSegment(speaker="alice", start=5.0, end=3.0)

    def test_equal_start_end_raises(self):
        with pytest.raises(ValidationError, match="must be greater than start"):
            SpeakerSegment(speaker="alice", start=2.0, end=2.0)

    def test_duration_property(self):
        seg = SpeakerSegment(speaker="bob", start=1.5, end=4.5)
        assert seg.duration == pytest.approx(3.0)

    def test_default_confidence_is_zero(self):
        seg = SpeakerSegment(speaker="x", start=0.0, end=1.0)
        assert seg.confidence == 0.0

    def test_frozen(self):
        seg = SpeakerSegment(speaker="x", start=0.0, end=1.0)
        with pytest.raises(ValidationError):
            seg.speaker = "y"


# ---------------------------------------------------------------------------
# OverlapSegment
# ---------------------------------------------------------------------------


class TestOverlapSegment:
    def test_speaker_field_is_overlap(self):
        seg = OverlapSegment(speakers=["alice", "bob"], start=2.0, end=3.5)
        assert seg.speaker == "OVERLAP"

    def test_end_after_start_validation(self):
        with pytest.raises(ValidationError, match="must be greater than start"):
            OverlapSegment(speakers=["a", "b"], start=5.0, end=3.0)

    def test_duration_property(self):
        seg = OverlapSegment(speakers=["a", "b"], start=1.0, end=2.5)
        assert seg.duration == pytest.approx(1.5)

    def test_frozen(self):
        seg = OverlapSegment(speakers=["a", "b"], start=0.0, end=1.0)
        with pytest.raises(ValidationError):
            seg.start = 0.5


# ---------------------------------------------------------------------------
# SpeakerProfile
# ---------------------------------------------------------------------------


class TestSpeakerProfile:
    def test_basic_creation(self):
        profile = SpeakerProfile(name="alice", embedding=[0.1] * 256, num_samples=2)
        assert profile.name == "alice"
        assert len(profile.embedding) == 256
        assert profile.num_samples == 2
        assert isinstance(profile.created_at, datetime)

    def test_serialization_roundtrip(self):
        profile = SpeakerProfile(
            name="bob",
            embedding=[0.5] * 256,
            num_samples=3,
            created_at=datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        )
        data = profile.model_dump(mode="json")
        restored = SpeakerProfile(**data)
        assert restored.name == profile.name
        assert restored.embedding == profile.embedding
        assert restored.num_samples == profile.num_samples

    def test_num_samples_min_one(self):
        with pytest.raises(ValidationError):
            SpeakerProfile(name="x", embedding=[0.0], num_samples=0)


# ---------------------------------------------------------------------------
# DiarizationResult
# ---------------------------------------------------------------------------


class TestDiarizationResult:
    def test_mixed_segment_types(self):
        segments = [
            SpeakerSegment(speaker="alice", start=0.0, end=3.0, confidence=0.9),
            OverlapSegment(speakers=["alice", "bob"], start=2.5, end=3.5),
            SpeakerSegment(speaker="bob", start=3.0, end=6.0, confidence=0.8),
        ]
        result = DiarizationResult(
            segments=segments,
            audio_duration=6.0,
            num_speakers=2,
            processing_time=1.23,
        )
        assert len(result.segments) == 3
        assert result.audio_duration == 6.0
        assert result.num_speakers == 2
        assert result.processing_time == pytest.approx(1.23)

    def test_empty_segments(self):
        result = DiarizationResult(segments=[], audio_duration=10.0, num_speakers=0)
        assert result.segments == []
        assert result.processing_time == 0.0


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------


class TestTranscriptSegment:
    def test_valid_creation(self):
        seg = TranscriptSegment(
            speaker="alice", start=1.0, end=3.0, text="Hello world", confidence=0.9
        )
        assert seg.speaker == "alice"
        assert seg.start == 1.0
        assert seg.end == 3.0
        assert seg.text == "Hello world"
        assert seg.confidence == 0.9

    def test_end_must_be_after_start(self):
        with pytest.raises(ValidationError, match="must be after start"):
            TranscriptSegment(speaker="alice", start=5.0, end=3.0, text="hi")

    def test_duration_property(self):
        seg = TranscriptSegment(speaker="bob", start=1.5, end=4.5, text="test")
        assert seg.duration == pytest.approx(3.0)

    def test_frozen(self):
        seg = TranscriptSegment(speaker="x", start=0.0, end=1.0, text="test")
        with pytest.raises(ValidationError):
            seg.speaker = "y"


# ---------------------------------------------------------------------------
# TranscriptResult
# ---------------------------------------------------------------------------


class TestTranscriptResult:
    def test_mixed_segments(self):
        segments = [
            TranscriptSegment(speaker="alice", start=0.0, end=3.0, text="Hi", confidence=0.9),
            TranscriptSegment(speaker="bob", start=3.0, end=6.0, text="Hey", confidence=0.8),
        ]
        result = TranscriptResult(
            segments=segments, audio_duration=6.0, num_speakers=2, processing_time=1.0
        )
        assert len(result.segments) == 2
        assert result.audio_duration == 6.0
        assert result.num_speakers == 2

    def test_full_transcript_property(self):
        segments = [
            TranscriptSegment(speaker="alice", start=0.0, end=2.0, text="Hello"),
            TranscriptSegment(speaker="bob", start=2.0, end=4.0, text="World"),
        ]
        result = TranscriptResult(segments=segments, audio_duration=4.0, num_speakers=2)
        assert result.full_transcript == "[alice] Hello\n[bob] World"

    def test_by_speaker_property(self):
        segments = [
            TranscriptSegment(speaker="alice", start=0.0, end=2.0, text="Hi"),
            TranscriptSegment(speaker="bob", start=2.0, end=4.0, text="Hey"),
            TranscriptSegment(speaker="alice", start=4.0, end=6.0, text="Bye"),
        ]
        result = TranscriptResult(segments=segments, audio_duration=6.0, num_speakers=2)
        by_spk = result.by_speaker
        assert len(by_spk["alice"]) == 2
        assert len(by_spk["bob"]) == 1

    def test_empty_segments(self):
        result = TranscriptResult(segments=[], audio_duration=10.0, num_speakers=0)
        assert result.segments == []
        assert result.full_transcript == ""
        assert result.by_speaker == {}
