from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa

from audio.analyzer import detect_segments
from output.html_generator import generate_player_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect high-energy segments in a stadium WAV recording."
    )
    parser.add_argument("wav", help="Path to WAV file")
    parser.add_argument(
        "--threshold", type=float, default=1.5,
        help="Energy multiplier over median (default: 1.5)",
    )
    parser.add_argument(
        "--merge-gap", type=float, default=10.0,
        help="Seconds of quiet to bridge between segments (default: 10)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=15.0,
        help="Minimum segment length in seconds (default: 15)",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Keep only top N segments by energy score",
    )
    parser.add_argument(
        "--cut", action="store_true",
        help="Also export segments as MP3 clips (requires ffmpeg)",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    if not wav_path.exists():
        print(f"Error: file not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analysiere {wav_path.name} ...")

    duration = librosa.get_duration(path=str(wav_path))

    segments = detect_segments(
        str(wav_path),
        threshold=args.threshold,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
        top=args.top,
    )

    meta = {
        "file": wav_path.name,
        "duration_total": round(duration, 2),
        "segments_found": len(segments),
        "parameters": {
            "threshold": args.threshold,
            "merge_gap": args.merge_gap,
            "min_duration": args.min_duration,
        },
    }

    stem = wav_path.stem
    out_dir = wav_path.parent
    json_path = out_dir / f"{stem}-segments.json"
    html_path = out_dir / f"{stem}-player.html"

    json_path.write_text(
        json.dumps({**meta, "segments": segments}, indent=2, ensure_ascii=False)
    )
    html_path.write_text(
        generate_player_html(wav_path.name, segments, meta),
        encoding="utf-8",
    )

    print(f"{len(segments)} Segmente gefunden")
    print(f"  -> {json_path.name}")
    print(f"  -> {html_path.name}")

    for i, seg in enumerate(segments, 1):
        h = int(seg["start"] // 3600)
        m = int((seg["start"] % 3600) // 60)
        s = int(seg["start"] % 60)
        print(f"  {i:02d}. {h:02d}h{m:02d}m{s:02d}s  Energie: {seg['energy_score']:.0%}")

    if args.cut and segments:
        from audio.cutter import cut_segments

        clip_paths = cut_segments(str(wav_path), segments)
        clips_dir = Path(clip_paths[0]).parent
        print(f"  -> {clips_dir.name}/ ({len(clip_paths)} Clips)")


if __name__ == "__main__":
    main()
