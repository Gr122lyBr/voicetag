"""Core pipeline orchestration for the voicetag library.

Contains the ``Pipeline`` class that coordinates diarization, embedding
computation, speaker matching, and overlap detection into a single
``identify()`` call.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger

from voicetag.diarizer import Diarizer
from voicetag.encoder import SpeakerEncoder
from voicetag.models import (
    DiarizationResult,
    OverlapSegment,
    SpeakerProfile,
    SpeakerSegment,
    TranscriptResult,
    TranscriptSegment,
    VoiceTagConfig,
)
from voicetag.overlap import detect_overlaps, merge_segments
from voicetag.transcriber import get_transcriber
from voicetag.utils import load_audio, validate_audio_path


class Pipeline:
    """Speaker identification pipeline.

    Orchestrates diarization, embedding computation, speaker matching,
    and overlap detection.

    Args:
        config: Pipeline configuration. If ``None``, uses defaults.
    """

    def __init__(self, config: Optional[VoiceTagConfig] = None) -> None:
        self._config: VoiceTagConfig = config or VoiceTagConfig()
        self._diarizer: Diarizer = Diarizer(
            hf_token=self._config.hf_token,
            device=self._config.device,
        )
        self._encoder: SpeakerEncoder = SpeakerEncoder(
            device=self._config.device,
        )
        logger.debug(
            "Pipeline initialised (device={}, max_workers={})",
            self._config.device,
            self._config.max_workers,
        )

    def enroll(self, name: str, audio_paths: list[str | Path]) -> SpeakerProfile:
        """Enroll a speaker from one or more audio samples.

        Args:
            name: Speaker name.
            audio_paths: Paths to audio files for this speaker.

        Returns:
            The created ``SpeakerProfile``.
        """
        return self._encoder.enroll(name, audio_paths)

    def save(self, path: str | Path) -> None:
        """Save enrolled speaker profiles to disk.

        Args:
            path: File path to save profiles to (JSON).
        """
        self._encoder.save_profiles(path)

    def load(self, path: str | Path) -> None:
        """Load speaker profiles from disk.

        Args:
            path: File path to load profiles from (JSON).
        """
        self._encoder.load_profiles(path)

    def remove_speaker(self, name: str) -> None:
        """Remove an enrolled speaker by name.

        Args:
            name: Speaker name to remove.
        """
        self._encoder.remove_speaker(name)

    @property
    def enrolled_speakers(self) -> list[str]:
        """List of currently enrolled speaker names."""
        return self._encoder.enrolled_speakers

    def identify(
        self,
        audio_path: str | Path,
        profiles: Optional[dict[str, SpeakerProfile]] = None,
    ) -> DiarizationResult:
        """Run the full speaker identification pipeline.

        Steps:
            1. Load and validate the audio file.
            2. Run pyannote diarization.
            3. Extract audio segments for each diarized region.
            4. Compute embeddings in parallel.
            5. Match embeddings against enrolled profiles.
            6. Detect overlapping speech regions.
            7. Merge and sort all segments.
            8. Return a ``DiarizationResult``.

        Args:
            audio_path: Path to the audio file to process.
            profiles: Optional external speaker profiles dict. If ``None``,
                uses the internal enrollment store from the encoder.

        Returns:
            ``DiarizationResult`` with identified segments, overlap regions,
            audio duration, and speaker count.
        """
        t_start = time.monotonic()

        validated_path = validate_audio_path(audio_path)
        audio, sr = load_audio(validated_path)
        audio_duration = len(audio) / sr
        logger.info(
            "Processing {}: {:.1f}s of audio",
            validated_path.name,
            audio_duration,
        )

        raw_segments = self._diarizer.diarize(validated_path)

        if not raw_segments:
            logger.warning("No speech segments detected")
            return DiarizationResult(
                segments=[],
                audio_duration=audio_duration,
                num_speakers=0,
                processing_time=time.monotonic() - t_start,
            )

        segment_data: list[tuple[dict, np.ndarray]] = []

        def _process_segment(seg: dict) -> Optional[tuple[dict, np.ndarray]]:
            segment_audio = self._extract_segment(audio, sr, seg["start"], seg["end"])
            if len(segment_audio) < int(sr * 0.1):
                return None
            try:
                embedding = self._encoder.get_embedding(segment_audio, sr)
                return (seg, embedding)
            except Exception as exc:
                logger.warning(
                    "Embedding computation failed for segment {:.1f}-{:.1f}s: {}",
                    seg["start"],
                    seg["end"],
                    exc,
                )
                return None

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as pool:
            results = list(pool.map(_process_segment, raw_segments))

        segment_data = [r for r in results if r is not None]

        speaker_segments: list[SpeakerSegment] = []
        detected_speakers: set[str] = set()

        for seg, embedding in segment_data:
            speaker_name, confidence = self._match_speaker(
                embedding,
                self._encoder,
                self._config.similarity_threshold,
                profiles=profiles,
            )
            detected_speakers.add(speaker_name)
            speaker_segments.append(
                SpeakerSegment(
                    speaker=speaker_name,
                    start=seg["start"],
                    end=seg["end"],
                    confidence=confidence,
                )
            )

        raw_overlaps = detect_overlaps(raw_segments, threshold=self._config.overlap_threshold)
        overlap_segments: list[OverlapSegment] = [
            OverlapSegment(
                speakers=ovlp["speakers"],
                start=ovlp["start"],
                end=ovlp["end"],
            )
            for ovlp in raw_overlaps
        ]

        speaker_dicts = [
            {
                "speaker": s.speaker,
                "start": s.start,
                "end": s.end,
                "confidence": s.confidence,
            }
            for s in speaker_segments
        ]
        overlap_dicts = [
            {"speakers": o.speakers, "start": o.start, "end": o.end} for o in overlap_segments
        ]
        merged = merge_segments(
            speaker_dicts,
            overlap_dicts,
            min_duration=self._config.min_segment_duration,
        )

        final_segments: list[Union[SpeakerSegment, OverlapSegment]] = []
        for item in merged:
            if item.get("speaker") == "OVERLAP":
                final_segments.append(
                    OverlapSegment(
                        speakers=item["speakers"],
                        start=item["start"],
                        end=item["end"],
                    )
                )
            else:
                final_segments.append(
                    SpeakerSegment(
                        speaker=item["speaker"],
                        start=item["start"],
                        end=item["end"],
                        confidence=item.get("confidence", 0.0),
                    )
                )

        real_speakers = {
            s.speaker
            for s in final_segments
            if isinstance(s, SpeakerSegment) and s.speaker != "UNKNOWN"
        }

        processing_time = time.monotonic() - t_start
        logger.info(
            "Identification complete: {} segments, {} speakers, {:.2f}s",
            len(final_segments),
            len(real_speakers),
            processing_time,
        )

        return DiarizationResult(
            segments=final_segments,
            audio_duration=audio_duration,
            num_speakers=len(real_speakers),
            processing_time=processing_time,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **provider_kwargs,
    ) -> TranscriptResult:
        """Run speaker identification and transcribe each segment.

        Args:
            audio_path: Path to the audio file.
            provider: STT provider name ("openai", "groq", "fireworks", "whisper", "deepgram").
            api_key: API key for the STT provider.
            model: Model name override for the STT provider.
            language: ISO language code hint (e.g., "en", "he", "zh").
            **provider_kwargs: Additional provider-specific arguments.

        Returns:
            ``TranscriptResult`` with speaker-attributed transcript segments.
        """
        import time as _time

        t_start = _time.monotonic()

        # Get speaker identification first
        diarization = self.identify(audio_path)

        if not diarization.segments:
            return TranscriptResult(
                segments=[],
                audio_duration=diarization.audio_duration,
                num_speakers=diarization.num_speakers,
                processing_time=_time.monotonic() - t_start,
            )

        # Load audio for segment extraction
        audio, sr = load_audio(audio_path)

        # Create transcriber
        transcriber = get_transcriber(provider, api_key=api_key, model=model, **provider_kwargs)

        def _transcribe_segment(seg):
            seg_audio = self._extract_segment(audio, sr, seg.start, seg.end)
            if len(seg_audio) < int(sr * 0.1):
                return None
            text = transcriber.transcribe(seg_audio, sr=sr, language=language)
            speaker = seg.speaker if isinstance(seg, SpeakerSegment) else "OVERLAP"
            conf = seg.confidence if isinstance(seg, SpeakerSegment) else 0.0
            return TranscriptSegment(
                speaker=speaker,
                start=seg.start,
                end=seg.end,
                text=text,
                confidence=conf,
            )

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as pool:
            results = list(pool.map(_transcribe_segment, diarization.segments))

        transcript_segments = [r for r in results if r is not None]

        return TranscriptResult(
            segments=transcript_segments,
            audio_duration=diarization.audio_duration,
            num_speakers=diarization.num_speakers,
            processing_time=_time.monotonic() - t_start,
        )

    @staticmethod
    def _extract_segment(
        audio: np.ndarray,
        sr: int,
        start: float,
        end: float,
    ) -> np.ndarray:
        """Extract a time-bounded slice from an audio waveform.

        Args:
            audio: Full 1-D audio waveform.
            sr: Sample rate.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            1-D numpy array of the audio slice.
        """
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        return audio[start_sample:end_sample]

    @staticmethod
    def _match_speaker(
        embedding: np.ndarray,
        encoder: SpeakerEncoder,
        threshold: float,
        profiles: Optional[dict[str, SpeakerProfile]] = None,
    ) -> tuple[str, float]:
        """Match an embedding against enrolled profiles.

        Args:
            embedding: 256-dimensional embedding vector.
            encoder: The ``SpeakerEncoder`` instance.
            threshold: Minimum cosine similarity for a positive match.
            profiles: Optional external profiles dict.

        Returns:
            Tuple of ``(speaker_name, confidence)``. Returns
            ``("UNKNOWN", 0.0)`` if no match exceeds the threshold.
        """
        name, score = encoder.compare(embedding, profiles=profiles)

        if score >= threshold:
            return (name, score)
        return ("UNKNOWN", 0.0)
