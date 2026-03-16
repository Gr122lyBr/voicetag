"""Tests for voicetag.cli — Typer CLI commands."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from voicetag.cli import app
from voicetag.exceptions import EnrollmentError
from voicetag.models import (
    DiarizationResult,
    SpeakerProfile,
    SpeakerSegment,
    TranscriptResult,
    TranscriptSegment,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedding(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(256).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


def _write_profiles(path: Path, names: list[str]) -> None:
    """Write a minimal profiles JSON file."""
    data = {}
    for i, name in enumerate(names):
        profile = SpeakerProfile(
            name=name,
            embedding=_fake_embedding(i),
            num_samples=2,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        data[name] = profile.model_dump(mode="json")
    path.write_text(json.dumps(data, default=str))


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


class TestVersionCommand:
    def test_version_output(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "voicetag" in result.output


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------


class TestEnrollCommand:
    @patch("voicetag.encoder.SpeakerEncoder.save_profiles")
    @patch("voicetag.encoder.SpeakerEncoder.enroll")
    def test_enroll_success(
        self, mock_enroll, mock_save, sample_audio_files: list[Path], tmp_path: Path
    ):
        mock_enroll.return_value = SpeakerProfile(
            name="alice", embedding=_fake_embedding(0), num_samples=3
        )

        profiles_path = tmp_path / "profiles.json"
        result = runner.invoke(
            app,
            [
                "enroll",
                "alice",
                str(sample_audio_files[0]),
                str(sample_audio_files[1]),
                "--profiles",
                str(profiles_path),
            ],
        )
        assert result.exit_code == 0
        assert "Enrollment complete" in result.output or "Enrolled" in result.output
        mock_enroll.assert_called_once()

    @patch("voicetag.encoder.SpeakerEncoder.save_profiles")
    @patch("voicetag.encoder.SpeakerEncoder.enroll")
    def test_enroll_error_shows_panel(
        self, mock_enroll, mock_save, sample_audio_files: list[Path], tmp_path: Path
    ):
        mock_enroll.side_effect = EnrollmentError("No valid audio")

        profiles_path = tmp_path / "profiles.json"
        result = runner.invoke(
            app,
            [
                "enroll",
                "alice",
                str(sample_audio_files[0]),
                "--profiles",
                str(profiles_path),
            ],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# profiles list
# ---------------------------------------------------------------------------


class TestProfilesListCommand:
    def test_list_no_profiles_file(self, tmp_path: Path):
        profiles_path = tmp_path / "nonexistent.json"
        result = runner.invoke(app, ["profiles", "list", "--profiles", str(profiles_path)])
        assert result.exit_code == 0
        assert "No Profiles" in result.output or "No profiles" in result.output


class TestProfilesListWithData:
    def test_list_shows_speakers(self, tmp_path: Path):
        """Integration-style: write real profiles and let the CLI load them."""
        profiles_path = tmp_path / "profiles.json"
        _write_profiles(profiles_path, ["alice", "bob"])

        # The CLI creates a real SpeakerEncoder (no resemblyzer needed for load).
        # load_profiles does not need the ML model.
        result = runner.invoke(app, ["profiles", "list", "--profiles", str(profiles_path)])
        assert result.exit_code == 0
        assert "alice" in result.output
        assert "bob" in result.output


# ---------------------------------------------------------------------------
# profiles remove
# ---------------------------------------------------------------------------


class TestProfilesRemoveCommand:
    def test_remove_success(self, tmp_path: Path):
        """Integration-style: write profiles, remove one, verify."""
        profiles_path = tmp_path / "profiles.json"
        _write_profiles(profiles_path, ["alice", "bob"])

        result = runner.invoke(
            app,
            ["profiles", "remove", "alice", "--profiles", str(profiles_path)],
        )
        assert result.exit_code == 0
        assert "Removed" in result.output or "remove" in result.output.lower()

        # Verify alice is actually removed from the file.
        with open(profiles_path) as f:
            data = json.load(f)
        assert "alice" not in data
        assert "bob" in data

    def test_remove_no_profiles_file(self, tmp_path: Path):
        profiles_path = tmp_path / "nonexistent.json"
        result = runner.invoke(
            app, ["profiles", "remove", "ghost", "--profiles", str(profiles_path)]
        )
        assert result.exit_code == 1

    def test_remove_nonexistent_speaker(self, tmp_path: Path):
        profiles_path = tmp_path / "profiles.json"
        _write_profiles(profiles_path, ["alice"])
        result = runner.invoke(
            app, ["profiles", "remove", "ghost", "--profiles", str(profiles_path)]
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# identify
# ---------------------------------------------------------------------------


class TestIdentifyCommand:
    @patch("voicetag.pipeline.Pipeline.identify")
    @patch("voicetag.pipeline.Diarizer")
    @patch("voicetag.pipeline.SpeakerEncoder")
    def test_identify_basic(
        self,
        MockEncoder,
        MockDiarizer,
        mock_identify,
        sample_audio_file: Path,
        tmp_path: Path,
    ):
        mock_result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker="alice", start=0.0, end=3.0, confidence=0.9),
                SpeakerSegment(speaker="bob", start=3.5, end=6.0, confidence=0.8),
            ],
            audio_duration=6.0,
            num_speakers=2,
            processing_time=1.5,
        )
        mock_identify.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "identify",
                str(sample_audio_file),
                "--profiles",
                str(tmp_path / "profiles.json"),
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0
        assert "Summary" in result.output


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


class TestTranscribeCommand:
    @patch("voicetag.pipeline.Pipeline.transcribe")
    @patch("voicetag.pipeline.Diarizer")
    @patch("voicetag.pipeline.SpeakerEncoder")
    def test_transcribe_basic(
        self,
        MockEncoder,
        MockDiarizer,
        mock_transcribe,
        sample_audio_file: Path,
        tmp_path: Path,
    ):
        mock_result = TranscriptResult(
            segments=[
                TranscriptSegment(
                    speaker="alice", start=0.0, end=3.0, text="Hello there", confidence=0.9
                ),
                TranscriptSegment(
                    speaker="bob", start=3.5, end=6.0, text="Hi back", confidence=0.8
                ),
            ],
            audio_duration=6.0,
            num_speakers=2,
            processing_time=2.0,
        )
        mock_transcribe.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "transcribe",
                str(sample_audio_file),
                "--provider",
                "openai",
                "--profiles",
                str(tmp_path / "profiles.json"),
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0
        assert "Summary" in result.output
        mock_transcribe.assert_called_once()

    @patch("voicetag.pipeline.Pipeline.transcribe")
    @patch("voicetag.pipeline.Diarizer")
    @patch("voicetag.pipeline.SpeakerEncoder")
    def test_transcribe_saves_json(
        self,
        MockEncoder,
        MockDiarizer,
        mock_transcribe,
        sample_audio_file: Path,
        tmp_path: Path,
    ):
        mock_result = TranscriptResult(
            segments=[
                TranscriptSegment(
                    speaker="alice", start=0.0, end=3.0, text="Hello", confidence=0.9
                ),
            ],
            audio_duration=3.0,
            num_speakers=1,
            processing_time=1.0,
        )
        mock_transcribe.return_value = mock_result

        output_path = tmp_path / "output.json"
        result = runner.invoke(
            app,
            [
                "transcribe",
                str(sample_audio_file),
                "--profiles",
                str(tmp_path / "profiles.json"),
                "--output",
                str(output_path),
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "segments" in data


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


class TestProvidersCommand:
    def test_providers_lists_providers(self):
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert "openai" in result.output
        assert "groq" in result.output
        assert "whisper" in result.output


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------


class TestErrorDisplay:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "usage" in result.output.lower()
