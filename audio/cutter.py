from __future__ import annotations

import subprocess
from pathlib import Path

from pydub import AudioSegment


def cut_segments(
    wav_path: str,
    segments: list[dict],
    output_format: str = "mp3",
) -> list[str]:
    wav = Path(wav_path).resolve()
    clips_dir = wav.parent / f"{wav.stem}-clips"
    clips_dir.mkdir(exist_ok=True)

    audio = AudioSegment.from_wav(str(wav))
    output_paths: list[str] = []

    for i, seg in enumerate(segments, 1):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        clip = audio[start_ms:end_ms]

        timestamp = _format_time(seg["start"])
        filename = f"{i:02d}-{timestamp}.{output_format}"
        out_path = clips_dir / filename

        # If WAV format, export directly (no ffmpeg needed)
        if output_format == "wav":
            clip.export(str(out_path), format="wav")
        else:
            # For other formats, write WAV to clips dir (snap ffmpeg needs home or accessible paths)
            tmp_wav_path = clips_dir / f".tmp_{i:02d}.wav"
            try:
                clip.export(str(tmp_wav_path), format="wav")
                _encode_with_ffmpeg(str(tmp_wav_path), str(out_path), output_format)
            finally:
                if tmp_wav_path.exists():
                    tmp_wav_path.unlink()

        output_paths.append(str(out_path))

    return output_paths


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _encode_with_ffmpeg(input_wav: str, output_file: str, fmt: str = "mp3") -> None:
    """Encode WAV to target format using ffmpeg.

    Uses absolute paths to work around snap sandbox restrictions.
    """
    input_abs = Path(input_wav).resolve()
    output_abs = Path(output_file).resolve()

    cmd = [
        "ffmpeg",
        "-i", str(input_abs),
        "-y",  # overwrite output file
        "-b:a", "192k",
        "-f", fmt,
        str(output_abs),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
