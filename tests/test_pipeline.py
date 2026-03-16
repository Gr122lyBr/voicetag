"""Tests for voicetag.pipeline — Pipeline orchestration with mocked backends."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from voicetag.models import (
    DiarizationResult,
    OverlapSegment,
    SpeakerSegment,
    TranscriptResult,
    VoiceTagConfig,
)
from voicetag.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(256).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture()
def pipeline(config: VoiceTagConfig) -> Pipeline:
    """Return a Pipeline with both Diarizer and SpeakerEncoder mocked."""
    with (
        patch("voicetag.pipeline.Diarizer") as MockDiarizer,
        patch("voicetag.pipeline.SpeakerEncoder") as MockEncoder,
    ):
        mock_diarizer = MockDiarizer.return_value
        mock_encoder = MockEncoder.return_value

        # Defaults — tests can override
        mock_diarizer.diarize.return_value = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
            {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.0},
        ]
        mock_encoder.get_embedding.return_value = _fake_embedding(0)
        mock_encoder.compare.return_value = ("alice", 0.9)
        mock_encoder.enrolled_speakers = ["alice", "bob"]
        mock_encoder._profiles = {}

        p = Pipeline(config=config)
        # Store mocks for test access
        p._test_mock_diarizer = mock_diarizer
        p._test_mock_encoder = mock_encoder
    return p


# ---------------------------------------------------------------------------
# identify
# ---------------------------------------------------------------------------


class TestIdentify:
    @patch("voicetag.pipeline.load_audio")
    @patch("voicetag.pipeline.validate_audio_path")
    def test_returns_diarization_result(
        self, mock_validate, mock_load, pipeline: Pipeline, sample_audio_file: Path
    ):
        mock_validate.return_value = sample_audio_file
        mock_load.return_value = (np.zeros(80_000, dtype=np.float32), 16_000)

        result = pipeline.identify(str(sample_audio_file))
        assert isinstance(result, DiarizationResult)
        assert result.audio_duration == pytest.approx(5.0)
        assert len(result.segments) > 0

    @patch("voicetag.pipeline.load_audio")
    @patch("voicetag.pipeline.validate_audio_path")
    def test_no_enrolled_speakers_marks_unknown(
        self, mock_validate, mock_load, pipeline: Pipeline, sample_audio_file: Path
    ):
        mock_validate.return_value = sample_audio_file
        mock_load.return_value = (np.zeros(80_000, dtype=np.float32), 16_000)
        pipeline._encoder.compare.return_value = ("UNKNOWN", 0.0)

        result = pipeline.identify(str(sample_audio_file))
        for seg in result.segments:
            if isinstance(seg, SpeakerSegment):
                assert seg.speaker == "UNKNOWN"

    @patch("voicetag.pipeline.load_audio")
    @patch("voicetag.pipeline.validate_audio_path")
    def test_handles_overlapping_segments(
        self, mock_validate, mock_load, pipeline: Pipeline, sample_audio_file: Path
    ):
        mock_validate.return_value = sample_audio_file
        mock_load.return_value = (np.zeros(160_000, dtype=np.float32), 16_000)
        pipeline._diarizer.diarize.return_value = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
            {"speaker": "SPEAKER_01", "start": 3.0, "end": 8.0},
        ]

        result = pipeline.identify(str(sample_audio_file))
        assert isinstance(result, DiarizationResult)
        overlap_segs = [s for s in result.segments if isinstance(s, OverlapSegment)]
        # With default overlap_threshold=0.5, the 2s overlap should be detected
        assert len(overlap_segs) >= 1

    @patch("voicetag.pipeline.load_audio")
    @patch("voicetag.pipeline.validate_audio_path")
    def test_respects_min_segment_duration(
        self, mock_validate, mock_load, sample_audio_file: Path, config: VoiceTagConfig
    ):
        """Short segments below min_segment_duration should be filtered."""
        with (
            patch("voicetag.pipeline.Diarizer") as MockDiarizer,
            patch("voicetag.pipeline.SpeakerEncoder") as MockEncoder,
        ):
            mock_diarizer = MockDiarizer.return_value
            mock_encoder = MockEncoder.return_value
            mock_diarizer.diarize.return_value = [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.3},  # too short (0.3 < 0.5)
                {"speaker": "SPEAKER_01", "start": 1.0, "end": 4.0},  # long enough
            ]
            mock_encoder.get_embedding.return_value = _fake_embedding(0)
            mock_encoder.compare.return_value = ("alice", 0.9)
            mock_encoder._profiles = {}

            p = Pipeline(config=config)

        mock_validate.return_value = sample_audio_file
        mock_load.return_value = (np.zeros(80_000, dtype=np.float32), 16_000)

        result = p.identify(str(sample_audio_file))
        # The 0.3s segment should be filtered by merge_segments
        durations = [s.end - s.start for s in result.segments]
        assert all(d >= config.min_segment_duration for d in durations)


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


class TestTranscribe:
    @patch("voicetag.pipeline.get_transcriber")
    @patch("voicetag.pipeline.load_audio")
    @patch("voicetag.pipeline.validate_audio_path")
    def test_returns_transcript_result(
        self,
        mock_validate,
        mock_load,
        mock_get_transcriber,
        pipeline: Pipeline,
        sample_audio_file: Path,
    ):
        mock_validate.return_value = sample_audio_file
        mock_load.return_value = (np.zeros(80_000, dtype=np.float32), 16_000)

        mock_transcriber = mock_get_transcriber.return_value
        mock_transcriber.transcribe.return_value = "Hello world"

        result = pipeline.transcribe(str(sample_audio_file), provider="openai", api_key="fake-key")
        assert isinstance(result, TranscriptResult)
        assert len(result.segments) > 0
        assert result.segments[0].text == "Hello world"
        mock_get_transcriber.assert_called_once_with("openai", api_key="fake-key", model=None)


class TestPipelineCreation:
    @patch("voicetag.pipeline.Diarizer")
    @patch("voicetag.pipeline.SpeakerEncoder")
    def test_creation_with_config(self, MockEncoder, MockDiarizer, config: VoiceTagConfig):
        p = Pipeline(config=config)
        assert p._config == config
        MockDiarizer.assert_called_once_with(hf_token=config.hf_token, device=config.device)
        MockEncoder.assert_called_once_with(device=config.device)

    @patch("voicetag.pipeline.Diarizer")
    @patch("voicetag.pipeline.SpeakerEncoder")
    def test_creation_with_defaults(self, MockEncoder, MockDiarizer):
        p = Pipeline()
        assert p._config.device == "cpu"
