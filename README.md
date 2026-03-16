<p align="center"><img src="https://raw.githubusercontent.com/Gr122lyBr/voicetag/main/docs/assets/logo.png" width="200" alt="voicetag"></p>

<h1 align="center">voicetag</h1>

<p align="center"><strong>Know who said what. Automatically.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/voicetag/"><img src="https://img.shields.io/pypi/v/voicetag?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/voicetag/"><img src="https://img.shields.io/pypi/pyversions/voicetag" alt="Python versions"></a>
  <a href="https://github.com/Gr122lyBr/voicetag/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/Gr122lyBr/voicetag/actions"><img src="https://img.shields.io/github/actions/workflow/status/voicetag/voicetag/ci.yml?label=CI" alt="CI status"></a>
  <a href="https://pypi.org/project/voicetag/"><img src="https://img.shields.io/pypi/dm/voicetag" alt="Downloads"></a>
</p>

---

## What is voicetag?

voicetag is a Python library for **speaker diarization and named speaker identification**. It combines [pyannote.audio](https://github.com/pyannote/pyannote-audio) for diarization with [resemblyzer](https://github.com/resemble-ai/Resemblyzer) for speaker embeddings, giving you a single interface to answer: *who is speaking, and when?*

Enroll speakers once with a few audio samples, then identify them in any recording -- meetings, podcasts, interviews, phone calls.

## Features

- :zap: **Dead-simple API** -- enroll speakers and identify them in three lines of code
- :globe_with_meridians: **Language agnostic** -- works with Hebrew, English, Mandarin, or any spoken language
- :busts_in_silhouette: **Built-in overlap detection** -- flags regions where multiple speakers talk simultaneously
- :rocket: **Fast parallel processing** -- concurrent embedding computation with configurable thread pools
- :keyboard: **CLI tool included** -- enroll, identify, and manage profiles from the terminal
- :floppy_disk: **Save/load speaker profiles** -- persist enrolled speakers to disk and reuse across sessions
- :white_check_mark: **Pydantic result models** -- fully typed, validated, immutable result objects

## Quick Start

```python
from voicetag import VoiceTag

vt = VoiceTag()
vt.enroll("Alice", ["alice1.wav", "alice2.wav"])
vt.enroll("Bob", ["bob1.wav"])

result = vt.identify("meeting.wav")
for segment in result.segments:
    print(f"{segment.speaker}: {segment.start:.1f}s - {segment.end:.1f}s")
```

Output:

```
Alice: 0.0s - 4.2s
Bob: 4.5s - 8.1s
Alice: 8.3s - 12.7s
UNKNOWN: 13.0s - 15.4s
```

## Installation

```bash
pip install voicetag
```

voicetag requires access to the [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization model, which is gated behind a HuggingFace license agreement.

### Prerequisites

1. **Accept the pyannote model licenses** at:
   - [hf.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [hf.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [hf.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
2. **Create a HuggingFace token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Set the token** via environment variable or config:

```bash
export HF_TOKEN="hf_your_token_here"
```

Or pass it directly:

```python
from voicetag import VoiceTag, VoiceTagConfig

vt = VoiceTag(config=VoiceTagConfig(hf_token="hf_your_token_here"))
```

### GPU Acceleration (optional)

For faster processing on CUDA or Apple Silicon:

```python
vt = VoiceTag(config=VoiceTagConfig(device="cuda"))  # NVIDIA GPU
vt = VoiceTag(config=VoiceTagConfig(device="mps"))    # Apple Silicon
```

## CLI Usage

voicetag ships with a full-featured command-line interface.

### Enroll a speaker

```bash
voicetag enroll "Alice" alice_sample1.wav alice_sample2.wav
```

```
Enrolled speaker Alice from 2 sample(s).
Profiles saved to voicetag_profiles.json
```

### Identify speakers in a recording

```bash
voicetag identify meeting.wav --threshold 0.8
```

```
Speaker Timeline -- meeting.wav
+---------+----------+----------+----------+------------+
| Speaker | Start    | End      | Duration | Confidence |
+---------+----------+----------+----------+------------+
| Alice   | 00:00.00 | 00:04.20 | 00:04.20 | 0.92       |
| Bob     | 00:04.50 | 00:08.10 | 00:03.60 | 0.87       |
| Alice   | 00:08.30 | 00:12.70 | 00:04.40 | 0.91       |
| UNKNOWN | 00:13.00 | 00:15.40 | 00:02.40 | --         |
+---------+----------+----------+----------+------------+

Summary
  Total duration:  00:15.400
  Speakers:        2
  Segments:        4
  Overlaps:        0
```

Save results to JSON:

```bash
voicetag identify meeting.wav --output results.json
```

### Manage profiles

```bash
voicetag profiles list
voicetag profiles remove "Alice"
```

### All CLI options

```bash
voicetag --help
voicetag identify --help
```

| Option | Description |
|---|---|
| `--profiles PATH` | Path to speaker profiles file (default: `voicetag_profiles.json`) |
| `--output, -o PATH` | Save results as JSON |
| `--threshold FLOAT` | Similarity threshold override (0.0-1.0) |
| `--hf-token TEXT` | HuggingFace API token |
| `--device TEXT` | Torch device: `cpu`, `cuda`, `mps` |
| `--unknown-only` | Skip speaker matching, just diarize |

## API Reference

### `VoiceTag`

The main entry point. Wraps the full diarization + identification pipeline.

```python
from voicetag import VoiceTag, VoiceTagConfig

vt = VoiceTag(config=VoiceTagConfig(...))
```

| Method | Returns | Description |
|---|---|---|
| `enroll(name, audio_paths)` | `SpeakerProfile` | Register a speaker from one or more audio files |
| `identify(audio_path)` | `DiarizationResult` | Run full identification pipeline on an audio file |
| `save(path)` | `None` | Save enrolled speaker profiles to disk |
| `load(path)` | `None` | Load speaker profiles from disk |
| `remove_speaker(name)` | `None` | Remove an enrolled speaker by name |
| `enrolled_speakers` | `list[str]` | Property: list of enrolled speaker names |

### `VoiceTagConfig`

Configuration model (Pydantic v2, frozen/immutable).

```python
config = VoiceTagConfig(
    hf_token="hf_...",          # HuggingFace token (or set HF_TOKEN env var)
    similarity_threshold=0.75,  # min cosine similarity for a match
    overlap_threshold=0.5,      # min overlap ratio to flag
    max_workers=4,              # parallel embedding threads
    min_segment_duration=0.5,   # discard segments shorter than this (seconds)
    device="cpu",               # "cpu", "cuda", or "mps"
)
```

### Result Models

**`DiarizationResult`** -- returned by `identify()`:

| Field | Type | Description |
|---|---|---|
| `segments` | `list[SpeakerSegment \| OverlapSegment]` | Ordered timeline of speaker segments |
| `audio_duration` | `float` | Total audio length in seconds |
| `num_speakers` | `int` | Number of distinct speakers detected |
| `processing_time` | `float` | Wall-clock pipeline time in seconds |

**`SpeakerSegment`**:

| Field | Type | Description |
|---|---|---|
| `speaker` | `str` | Identified speaker name or `"UNKNOWN"` |
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |
| `confidence` | `float` | Cosine similarity score (0.0-1.0) |
| `duration` | `float` | Property: `end - start` |

**`OverlapSegment`**:

| Field | Type | Description |
|---|---|---|
| `speakers` | `list[str]` | Names of overlapping speakers |
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |
| `speaker` | `Literal["OVERLAP"]` | Always `"OVERLAP"` |
| `duration` | `float` | Property: `end - start` |

**`SpeakerProfile`**:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Speaker name |
| `embedding` | `list[float]` | 256-dimensional mean embedding vector |
| `num_samples` | `int` | Number of audio files used for enrollment |
| `created_at` | `datetime` | UTC timestamp of enrollment |

### Error Handling

All exceptions inherit from `VoiceTagError`:

```python
from voicetag import VoiceTagError

try:
    result = vt.identify("audio.wav")
except VoiceTagError as e:
    print(f"Error: {e}")
```

| Exception | When |
|---|---|
| `VoiceTagConfigError` | Invalid config or missing HuggingFace token |
| `EnrollmentError` | Enrollment fails (no audio, bad format) |
| `DiarizationError` | Pyannote processing failure |
| `AudioLoadError` | Audio file not found or unsupported format |

## Real-World Use Cases

- **Podcasts** -- automatically label host vs. guest segments for transcription
- **Interviews** -- separate interviewer and interviewee speech for analysis
- **Meeting recordings** -- identify who said what in team meetings, generate per-speaker summaries
- **Court recordings** -- tag judge, attorney, and witness speech segments
- **Call centers** -- distinguish agent from customer in call recordings for QA
- **Media monitoring** -- track specific speakers across broadcast recordings

## How It Works

voicetag runs a three-stage pipeline:

```
Audio File
    |
    v
1. DIARIZE (pyannote.audio)
   "When does each speaker talk?"
   -> segments: [(0.0-4.2, SPEAKER_00), (4.5-8.1, SPEAKER_01), ...]
    |
    v
2. EMBED (resemblyzer)
   "What does each speaker sound like?"
   -> 256-dim embedding vector per segment (computed in parallel)
    |
    v
3. MATCH (cosine similarity)
   "Which enrolled speaker does this sound like?"
   -> Alice (0.92), Bob (0.87), UNKNOWN (below threshold)
    |
    v
DiarizationResult with named speaker timeline
```

1. **Diarize** -- pyannote.audio segments the audio into speaker turns with anonymous labels (`SPEAKER_00`, `SPEAKER_01`, etc.)
2. **Embed** -- resemblyzer computes a 256-dimensional voice embedding for each segment, running in parallel via a thread pool
3. **Match** -- each embedding is compared against enrolled speaker profiles using cosine similarity. Matches above the threshold get assigned the speaker's name; others are labeled `"UNKNOWN"`

Overlap detection runs in parallel with matching, identifying regions where two or more speakers talk simultaneously.

## Comparison

| Feature | voicetag | pyannote alone | WhisperX | Manual labeling |
|---|:---:|:---:|:---:|:---:|
| Speaker diarization | Yes | Yes | Yes | N/A |
| Named speaker identification | Yes | No | No | Yes |
| Overlap detection | Yes | Yes | No | Varies |
| CLI tool | Yes | No | Yes | N/A |
| Save/load speaker profiles | Yes | N/A | N/A | N/A |
| Language agnostic | Yes | Yes | Yes | Yes |
| Typed result models | Yes (Pydantic) | No | No | N/A |
| Lines of code to identify | 3 | ~30 | ~20 | N/A |

## Configuration

`VoiceTagConfig` controls all tunable parameters:

| Field | Type | Default | Description |
|---|---|---|---|
| `hf_token` | `Optional[str]` | `None` | HuggingFace token. Falls back to `HF_TOKEN` env var. |
| `similarity_threshold` | `float` | `0.75` | Minimum cosine similarity for a match. Range: (0.0, 1.0). |
| `overlap_threshold` | `float` | `0.5` | Minimum overlap ratio to flag as overlapping speech. |
| `max_workers` | `int` | `4` | Thread count for parallel embedding computation. |
| `min_segment_duration` | `float` | `0.5` | Segments shorter than this (seconds) are discarded. |
| `device` | `str` | `"cpu"` | Torch device: `"cpu"`, `"cuda"`, or `"mps"`. |

**Token resolution order:**
1. `config.hf_token` (explicit)
2. `HF_TOKEN` environment variable
3. Raise `VoiceTagConfigError` with a link to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up the development environment, running tests, and submitting pull requests.

## License

[MIT](LICENSE) -- Copyright (c) 2026 voicetag contributors
