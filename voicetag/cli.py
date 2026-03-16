"""Typer CLI for the voicetag speaker identification library.

Provides commands for enrolling speakers, identifying speakers in audio,
managing speaker profiles, and checking the version.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from voicetag import __version__
from voicetag.exceptions import VoiceTagError

DEFAULT_PROFILES_PATH = "voicetag_profiles.json"

SPEAKER_COLORS = [
    "cyan",
    "green",
    "magenta",
    "blue",
    "red",
    "bright_cyan",
    "bright_green",
    "bright_magenta",
    "bright_blue",
    "bright_red",
    "dark_orange",
    "purple",
]

console = Console()
err_console = Console(stderr=True)


def format_time(seconds: float) -> str:
    """Format a time value as MM:SS.mmm.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like ``"01:23.456"``.
    """
    minutes = int(seconds) // 60
    secs = seconds - (minutes * 60)
    return f"{minutes:02d}:{secs:06.3f}"


def _speaker_color(speaker: str, color_map: dict[str, str]) -> str:
    """Return a Rich color for a speaker, assigning one if needed."""
    if speaker == "OVERLAP":
        return "yellow"
    if speaker == "UNKNOWN":
        return "dim"
    if speaker not in color_map:
        idx = len(color_map) % len(SPEAKER_COLORS)
        color_map[speaker] = SPEAKER_COLORS[idx]
    return color_map[speaker]


app = typer.Typer(
    name="voicetag",
    help="Speaker identification from audio files.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

profiles_app = typer.Typer(
    name="profiles",
    help="Manage enrolled speaker profiles.",
    no_args_is_help=True,
)
app.add_typer(profiles_app, name="profiles")


@app.command()
def enroll(
    name: str = typer.Argument(..., help="Speaker name to enroll."),
    audio_files: list[Path] = typer.Argument(
        ..., help="One or more audio files of the speaker.", exists=True
    ),
    profiles: Path = typer.Option(
        DEFAULT_PROFILES_PATH,
        "--profiles",
        help="Path to save/load speaker profiles.",
    ),
) -> None:
    """Enroll a speaker from one or more audio files."""
    from voicetag.encoder import SpeakerEncoder

    encoder = SpeakerEncoder()

    if profiles.exists():
        try:
            encoder.load_profiles(profiles)
            console.print(f"[dim]Loaded existing profiles from {profiles}[/dim]")
        except VoiceTagError as exc:
            err_console.print(
                Panel(
                    str(exc),
                    title="[red]Error loading profiles[/red]",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Enrolling [bold]{name}[/bold]…",
                total=len(audio_files),
            )
            progress.update(task, completed=0)
            profile = encoder.enroll(name, [str(p) for p in audio_files])
            progress.update(task, completed=len(audio_files))

        encoder.save_profiles(profiles)

        console.print(
            Panel(
                f"[bold green]Enrolled speaker [cyan]{name}[/cyan] "
                f"from {profile.num_samples} sample(s).[/bold green]\n"
                f"Profiles saved to [dim]{profiles}[/dim]",
                title="[green]Enrollment complete[/green]",
                border_style="green",
            )
        )
    except VoiceTagError as exc:
        err_console.print(Panel(str(exc), title="[red]Enrollment Error[/red]", border_style="red"))
        raise typer.Exit(code=1)


@app.command()
def identify(
    audio_file: Path = typer.Argument(
        ..., help="Audio file to identify speakers in.", exists=True
    ),
    profiles: Path = typer.Option(
        DEFAULT_PROFILES_PATH,
        "--profiles",
        help="Path to enrolled speaker profiles.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save results as JSON to this path."
    ),
    unknown_only: bool = typer.Option(
        False, "--unknown-only", help="Skip speaker matching (just diarize)."
    ),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", help="Similarity threshold override (0.0-1.0)."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace API token."
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device (cpu, cuda, mps)."),
) -> None:
    """Identify speakers in an audio file."""
    from voicetag import VoiceTag, VoiceTagConfig

    config_kwargs: dict = {"device": device}
    if hf_token:
        config_kwargs["hf_token"] = hf_token
    if threshold is not None:
        config_kwargs["similarity_threshold"] = threshold

    try:
        config = VoiceTagConfig(**config_kwargs)
        vt = VoiceTag(config=config)

        if not unknown_only and profiles.exists():
            vt._encoder.load_profiles(profiles)
            console.print(f"[dim]Loaded profiles from {profiles}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Identifying speakers…", total=None)
            result = vt.identify(str(audio_file))

        color_map: dict[str, str] = {}

        table = Table(
            title=f"Speaker Timeline — [bold]{audio_file.name}[/bold]",
            show_lines=False,
        )
        table.add_column("Speaker", style="bold", min_width=12)
        table.add_column("Start", justify="right")
        table.add_column("End", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Confidence", justify="right")

        for seg in result.segments:
            speaker = seg.speaker
            color = _speaker_color(speaker, color_map)
            duration = seg.end - seg.start

            confidence_str = ""
            if hasattr(seg, "confidence"):
                conf = getattr(seg, "confidence", 0.0)
                if speaker == "OVERLAP":
                    confidence_str = "[dim]—[/dim]"
                elif conf > 0:
                    confidence_str = f"{conf:.2f}"
                else:
                    confidence_str = "[dim]—[/dim]"
            else:
                confidence_str = "[dim]—[/dim]"

            table.add_row(
                f"[{color}]{speaker}[/{color}]",
                format_time(seg.start),
                format_time(seg.end),
                format_time(duration),
                confidence_str,
            )

        console.print()
        console.print(table)
        console.print()

        total_dur = result.audio_duration
        n_segments = len(result.segments)
        n_speakers = result.num_speakers
        overlap_count = sum(1 for s in result.segments if s.speaker == "OVERLAP")

        summary_lines = [
            f"[bold]Total duration:[/bold]  {format_time(total_dur)}",
            f"[bold]Speakers:[/bold]        {n_speakers}",
            f"[bold]Segments:[/bold]        {n_segments}",
            f"[bold]Overlaps:[/bold]        {overlap_count}",
            f"[bold]Processing time:[/bold] {result.processing_time:.2f}s",
        ]
        console.print(
            Panel(
                "\n".join(summary_lines),
                title="[bold]Summary[/bold]",
                border_style="blue",
            )
        )

        if output is not None:
            output_data = result.model_dump(mode="json")
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            console.print(f"[dim]Results saved to {output}[/dim]")

    except VoiceTagError as exc:
        err_console.print(Panel(str(exc), title="[red]Error[/red]", border_style="red"))
        raise typer.Exit(code=1)


@profiles_app.command("list")
def profiles_list(
    profiles: Path = typer.Option(
        DEFAULT_PROFILES_PATH,
        "--profiles",
        help="Path to speaker profiles file.",
    ),
) -> None:
    """List enrolled speakers with sample counts."""
    from voicetag.encoder import SpeakerEncoder
    from voicetag.models import SpeakerProfile

    if not profiles.exists():
        console.print(
            Panel(
                f"No profiles file found at [bold]{profiles}[/bold].\n"
                "Enroll speakers first with [cyan]voicetag enroll[/cyan].",
                title="[yellow]No Profiles[/yellow]",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=0)

    try:
        encoder = SpeakerEncoder()
        encoder.load_profiles(profiles)
    except VoiceTagError as exc:
        err_console.print(Panel(str(exc), title="[red]Error[/red]", border_style="red"))
        raise typer.Exit(code=1)

    speakers = encoder.enrolled_speakers
    if not speakers:
        console.print("[yellow]No speakers enrolled.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Enrolled Speakers")
    table.add_column("Name", style="cyan bold")
    table.add_column("Samples", justify="right")
    table.add_column("Enrolled At")

    for name in sorted(speakers):
        profile: SpeakerProfile = encoder._profiles[name]
        table.add_row(
            name,
            str(profile.num_samples),
            profile.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

    console.print(table)
    console.print(f"\n[dim]{len(speakers)} speaker(s) enrolled.[/dim]")


@profiles_app.command("remove")
def profiles_remove(
    name: str = typer.Argument(..., help="Speaker name to remove."),
    profiles: Path = typer.Option(
        DEFAULT_PROFILES_PATH,
        "--profiles",
        help="Path to speaker profiles file.",
    ),
) -> None:
    """Remove a speaker from the profiles file."""
    from voicetag.encoder import SpeakerEncoder

    if not profiles.exists():
        err_console.print(
            Panel(
                f"Profiles file not found: [bold]{profiles}[/bold]",
                title="[red]Error[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    try:
        encoder = SpeakerEncoder()
        encoder.load_profiles(profiles)
        encoder.remove_speaker(name)
        encoder.save_profiles(profiles)

        console.print(
            Panel(
                f"Removed speaker [cyan]{name}[/cyan] from profiles.\n"
                f"Profiles saved to [dim]{profiles}[/dim]",
                title="[green]Speaker Removed[/green]",
                border_style="green",
            )
        )
    except VoiceTagError as exc:
        err_console.print(Panel(str(exc), title="[red]Error[/red]", border_style="red"))
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Audio file to transcribe.", exists=True),
    provider: str = typer.Option(
        "openai",
        "--provider",
        "-p",
        help="STT provider (openai, groq, fireworks, whisper, deepgram).",
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Language code hint (e.g. en, he)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model name override for the STT provider."
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the STT provider."),
    profiles: Path = typer.Option(
        DEFAULT_PROFILES_PATH,
        "--profiles",
        help="Path to enrolled speaker profiles.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save JSON output to this path."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace API token."
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device (cpu, cuda, mps)."),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", help="Similarity threshold override (0.0-1.0)."
    ),
) -> None:
    """Transcribe an audio file with speaker identification."""
    from voicetag.models import VoiceTagConfig
    from voicetag.pipeline import Pipeline

    config_kwargs: dict = {"device": device}
    if hf_token:
        config_kwargs["hf_token"] = hf_token
    if threshold is not None:
        config_kwargs["similarity_threshold"] = threshold

    try:
        config = VoiceTagConfig(**config_kwargs)
        pipe = Pipeline(config=config)

        if profiles.exists():
            pipe.load(profiles)
            console.print(f"[dim]Loaded profiles from {profiles}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Transcribing…", total=None)
            result = pipe.transcribe(
                audio_file,
                provider=provider,
                api_key=api_key,
                model=model,
                language=language,
            )

        color_map: dict[str, str] = {}

        table = Table(
            title=f"Transcript — [bold]{audio_file.name}[/bold]",
            show_lines=False,
        )
        table.add_column("Speaker", style="bold", min_width=12)
        table.add_column("Start", justify="right")
        table.add_column("End", justify="right")
        table.add_column("Text")

        for seg in result.segments:
            color = _speaker_color(seg.speaker, color_map)
            table.add_row(
                f"[{color}]{seg.speaker}[/{color}]",
                format_time(seg.start),
                format_time(seg.end),
                seg.text,
            )

        console.print()
        console.print(table)
        console.print()

        n_segments = len(result.segments)
        n_speakers = result.num_speakers

        summary_lines = [
            f"[bold]Total duration:[/bold]  {format_time(result.audio_duration)}",
            f"[bold]Speakers:[/bold]        {n_speakers}",
            f"[bold]Segments:[/bold]        {n_segments}",
            f"[bold]Provider:[/bold]        {provider}",
            f"[bold]Processing time:[/bold] {result.processing_time:.2f}s",
        ]
        console.print(
            Panel(
                "\n".join(summary_lines),
                title="[bold]Summary[/bold]",
                border_style="blue",
            )
        )

        if output is not None:
            output_data = result.model_dump(mode="json")
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            console.print(f"[dim]Results saved to {output}[/dim]")

    except VoiceTagError as exc:
        err_console.print(Panel(str(exc), title="[red]Error[/red]", border_style="red"))
        raise typer.Exit(code=1)


@app.command()
def providers() -> None:
    """List available STT providers."""
    from voicetag.transcriber import available_providers

    provider_list = available_providers()
    table = Table(title="Available STT Providers")
    table.add_column("Provider", style="cyan bold")

    for name in provider_list:
        table.add_row(name)

    console.print(table)
    console.print(f"\n[dim]{len(provider_list)} provider(s) available.[/dim]")


@app.command()
def version() -> None:
    """Show the voicetag version."""
    console.print(f"voicetag [bold cyan]{__version__}[/bold cyan]")


def main() -> None:
    """CLI entry point with top-level error handling."""
    try:
        app()
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
