"""Tests for voicetag.transcriber — STT provider registry and base transcriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetag.exceptions import TranscriptionError
from voicetag.transcriber import BaseTranscriber, available_providers, get_transcriber

# ---------------------------------------------------------------------------
# available_providers
# ---------------------------------------------------------------------------


class TestAvailableProviders:
    def test_returns_expected_list(self):
        providers = available_providers()
        assert isinstance(providers, list)
        assert "openai" in providers
        assert "groq" in providers
        assert "fireworks" in providers
        assert "whisper" in providers
        assert "deepgram" in providers
        assert providers == sorted(providers)


# ---------------------------------------------------------------------------
# get_transcriber
# ---------------------------------------------------------------------------


class TestGetTranscriber:
    def test_unknown_provider_raises(self):
        with pytest.raises(TranscriptionError, match="Unknown STT provider"):
            get_transcriber("nonexistent_provider")

    def test_missing_sdk_raises(self):
        with patch("voicetag.transcriber.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'openai'")
            with pytest.raises(TranscriptionError, match="requires additional dependencies"):
                get_transcriber("openai", api_key="test-key")

    def test_get_openai_transcriber_returns_correct_type(self):
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.OpenAITranscriber = mock_class
        mock_class.return_value = MagicMock(spec=BaseTranscriber)

        with patch("voicetag.transcriber.importlib.import_module", return_value=mock_module):
            result = get_transcriber("openai", api_key="test-key")
            mock_class.assert_called_once_with(api_key="test-key")
            assert result == mock_class.return_value

    def test_get_groq_transcriber_returns_correct_type(self):
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.GroqTranscriber = mock_class
        mock_class.return_value = MagicMock(spec=BaseTranscriber)

        with patch("voicetag.transcriber.importlib.import_module", return_value=mock_module):
            result = get_transcriber("groq", api_key="test-key", model="whisper-large-v3")
            mock_class.assert_called_once_with(api_key="test-key", model="whisper-large-v3")
            assert result == mock_class.return_value

    def test_get_whisper_transcriber_returns_correct_type(self):
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.WhisperLocalTranscriber = mock_class
        mock_class.return_value = MagicMock(spec=BaseTranscriber)

        with patch("voicetag.transcriber.importlib.import_module", return_value=mock_module):
            result = get_transcriber("whisper", model="base")
            mock_class.assert_called_once_with(model="base")
            assert result == mock_class.return_value


# ---------------------------------------------------------------------------
# BaseTranscriber helpers
# ---------------------------------------------------------------------------


class TestBaseTranscriberHelpers:
    """Test the helper methods on BaseTranscriber via a concrete subclass."""

    def _make_concrete(self):
        """Create a minimal concrete subclass of BaseTranscriber."""

        class ConcreteTranscriber(BaseTranscriber):
            def transcribe(self, audio, sr=16000, language=None):
                return "hello"

        return ConcreteTranscriber()

    def test_audio_to_wav_bytes_returns_bytes(self):
        t = self._make_concrete()
        audio = np.zeros(16000, dtype=np.float32)
        result = t._audio_to_wav_bytes(audio, 16000)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_audio_to_temp_file_creates_file(self):
        t = self._make_concrete()
        audio = np.zeros(16000, dtype=np.float32)
        path = t._audio_to_temp_file(audio, 16000)
        assert Path(path).exists()
        assert path.endswith(".wav")
        # Cleanup
        Path(path).unlink(missing_ok=True)
