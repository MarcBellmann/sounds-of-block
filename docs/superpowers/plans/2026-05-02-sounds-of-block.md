# Sounds of Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that detects high-energy segments in 90-minute stadium WAV recordings and outputs a timestamped JSON file plus a clickable browser-based HTML player; Phase 2 adds MP3 clip export and a mobile-accessible Flask server.

**Architecture:** `audio/analyzer.py` loads the WAV with librosa and detects loud segments via RMS energy windowing; `output/html_generator.py` renders those segments into a self-contained HTML player; `analyze.py` wires both together as a CLI. Phase 2 adds `audio/cutter.py` (pydub-based MP3 export) and `serve.py` (Flask server for LAN mobile access), both activated via CLI flags.

**Tech Stack:** Python 3.9+, librosa, numpy, soundfile, pydub (Phase 2), Flask (Phase 2), pytest

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `audio/__init__.py`
- Create: `output/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/audio/__init__.py`
- Create: `tests/output/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
librosa>=0.10.0
numpy>=1.24.0
soundfile>=0.12.0
```

- [ ] **Step 2: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest>=7.4.0
pytest-cov>=4.1.0
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements-dev.txt`
Expected: All packages install without errors.

- [ ] **Step 4: Create package init files**

Create the following as empty files:
- `audio/__init__.py`
- `output/__init__.py`
- `tests/__init__.py`
- `tests/audio/__init__.py`
- `tests/output/__init__.py`

- [ ] **Step 5: Verify test runner works**

Run: `pytest --co -q`
Expected: `no tests ran` with no errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt audio/ output/ tests/
git commit -m "chore: project setup with dependencies and package structure"
```

---

### Task 2: Energy Detection — `audio/analyzer.py`

**Files:**
- Create: `audio/analyzer.py`
- Create: `tests/audio/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/audio/test_analyzer.py`:

```python
import numpy as np
import pytest
import soundfile as sf
from audio.analyzer import detect_segments


def _write_wav(path, audio, sr=22050):
    sf.write(str(path), audio, sr)


def test_detect_segments_finds_loud_section(tmp_path):
    sr = 22050
    audio = np.random.normal(0, 0.01, sr * 60).astype(np.float32)
    audio[sr * 20 : sr * 35] = np.random.normal(0, 0.8, sr * 15).astype(np.float32)
    wav = tmp_path / "test.wav"
    _write_wav(wav, audio, sr)

    segments = detect_segments(str(wav))

    assert len(segments) >= 1
    assert any(s["start"] <= 22 and s["end"] >= 33 for s in segments)


def test_detect_segments_returns_required_keys(tmp_path):
    sr = 22050
    audio = np.random.normal(0, 0.01, sr * 60).astype(np.float32)
    audio[sr * 10 : sr * 25] = np.random.normal(0, 0.8, sr * 15).astype(np.float32)
    wav = tmp_path / "test.wav"
    _write_wav(wav, audio, sr)

    segments = detect_segments(str(wav))

    assert len(segments) >= 1
    for seg in segments:
        assert "start" in seg
        assert "end" in seg
        assert "duration" in seg
        assert "energy_score" in seg
        assert 0.0 <= seg["energy_score"] <= 1.0
        assert seg["duration"] > 0
        assert seg["end"] > seg["start"]


def test_detect_segments_merges_nearby_sections(tmp_path):
    sr = 22050
    audio = np.random.normal(0, 0.01, sr * 90).astype(np.float32)
    # two loud sections 5 s apart — should merge with default merge_gap=10
    audio[sr * 20 : sr * 30] = np.random.normal(0, 0.8, sr * 10).astype(np.float32)
    audio[sr * 35 : sr * 45] = np.random.normal(0, 0.8, sr * 10).astype(np.float32)
    wav = tmp_path / "test.wav"
    _write_wav(wav, audio, sr)

    segments = detect_segments(str(wav), merge_gap=10.0)

    assert len(segments) == 1
    assert segments[0]["start"] <= 22
    assert segments[0]["end"] >= 43


def test_detect_segments_filters_short_segments(tmp_path):
    sr = 22050
    audio = np.random.normal(0, 0.01, sr * 60).astype(np.float32)
    # 5 s burst — shorter than default min_duration=15
    audio[sr * 10 : sr * 15] = np.random.normal(0, 0.8, sr * 5).astype(np.float32)
    wav = tmp_path / "test.wav"
    _write_wav(wav, audio, sr)

    segments = detect_segments(str(wav), min_duration=15.0)

    assert len(segments) == 0


def test_detect_segments_top_n(tmp_path):
    sr = 22050
    audio = np.random.normal(0, 0.01, sr * 120).astype(np.float32)
    audio[sr * 10 : sr * 30] = np.random.normal(0, 0.9, sr * 20).astype(np.float32)
    audio[sr * 50 : sr * 70] = np.random.normal(0, 0.7, sr * 20).astype(np.float32)
    audio[sr * 90 : sr * 110] = np.random.normal(0, 0.5, sr * 20).astype(np.float32)
    wav = tmp_path / "test.wav"
    _write_wav(wav, audio, sr)

    segments = detect_segments(str(wav), top=2)

    assert len(segments) == 2


def test_detect_segments_sorted_by_energy_descending(tmp_path):
    sr = 22050
    audio = np.random.normal(0, 0.01, sr * 120).astype(np.float32)
    audio[sr * 10 : sr * 30] = np.random.normal(0, 0.9, sr * 20).astype(np.float32)
    audio[sr * 60 : sr * 80] = np.random.normal(0, 0.4, sr * 20).astype(np.float32)
    wav = tmp_path / "test.wav"
    _write_wav(wav, audio, sr)

    segments = detect_segments(str(wav))

    scores = [s["energy_score"] for s in segments]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/audio/test_analyzer.py -v`
Expected: All 6 tests FAIL with `ModuleNotFoundError: No module named 'audio.analyzer'`

- [ ] **Step 3: Implement `audio/analyzer.py`**

```python
from __future__ import annotations

import numpy as np
import librosa


def detect_segments(
    filepath: str,
    threshold: float = 1.5,
    merge_gap: float = 10.0,
    min_duration: float = 15.0,
    top: int | None = None,
) -> list[dict]:
    y, sr = librosa.load(filepath, sr=None, mono=True)

    hop_length = int(sr)           # 1-second hops
    frame_length = int(sr * 5)     # 5-second windows

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    cutoff = float(np.median(rms)) * threshold
    loud_mask = rms >= cutoff

    segments = _mask_to_segments(loud_mask, times, rms)
    segments = _merge_segments(segments, merge_gap)
    segments = [s for s in segments if s["duration"] >= min_duration]

    if not segments:
        return []

    max_rms = max(s["_peak_rms"] for s in segments)
    for s in segments:
        s["energy_score"] = round(s["_peak_rms"] / max_rms, 2)
        del s["_peak_rms"]

    segments.sort(key=lambda s: s["energy_score"], reverse=True)

    if top is not None:
        segments = segments[:top]

    return segments


def _mask_to_segments(
    loud_mask: np.ndarray,
    times: np.ndarray,
    rms: np.ndarray,
) -> list[dict]:
    segments: list[dict] = []
    in_segment = False
    start = 0.0
    peak_rms = 0.0

    for i, (is_loud, t) in enumerate(zip(loud_mask, times)):
        if is_loud and not in_segment:
            in_segment = True
            start = float(t)
            peak_rms = float(rms[i])
        elif is_loud and in_segment:
            peak_rms = max(peak_rms, float(rms[i]))
        elif not is_loud and in_segment:
            end = float(t)
            segments.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(end - start, 2),
                    "_peak_rms": peak_rms,
                }
            )
            in_segment = False

    if in_segment:
        end = float(times[-1])
        segments.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "_peak_rms": peak_rms,
            }
        )

    return segments


def _merge_segments(segments: list[dict], merge_gap: float) -> list[dict]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s["start"])
    merged = [ordered[0].copy()]

    for seg in ordered[1:]:
        last = merged[-1]
        if seg["start"] - last["end"] <= merge_gap:
            last["end"] = max(last["end"], seg["end"])
            last["duration"] = round(last["end"] - last["start"], 2)
            last["_peak_rms"] = max(last["_peak_rms"], seg["_peak_rms"])
        else:
            merged.append(seg.copy())

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/audio/test_analyzer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add audio/analyzer.py tests/audio/test_analyzer.py
git commit -m "feat: add energy-based segment detection"
```

---

### Task 3: HTML Player — `output/html_generator.py`

**Files:**
- Create: `output/html_generator.py`
- Create: `tests/output/test_html_generator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/output/test_html_generator.py`:

```python
from output.html_generator import generate_player_html

SEGMENTS = [
    {"start": 262.0, "end": 315.5, "duration": 53.5, "energy_score": 0.92},
    {"start": 728.0, "end": 824.1, "duration": 96.1, "energy_score": 0.75},
]
META = {
    "file": "dortmund.wav",
    "duration_total": 5400.0,
    "segments_found": 2,
}


def test_generate_player_html_returns_string():
    html = generate_player_html("dortmund.wav", SEGMENTS, META)
    assert isinstance(html, str)
    assert len(html) > 100


def test_generate_player_html_contains_audio_element():
    html = generate_player_html("dortmund.wav", SEGMENTS, META)
    assert "<audio" in html
    assert "dortmund.wav" in html


def test_generate_player_html_contains_all_segment_timestamps():
    html = generate_player_html("dortmund.wav", SEGMENTS, META)
    assert "262.0" in html
    assert "315.5" in html
    assert "728.0" in html
    assert "824.1" in html


def test_generate_player_html_shows_energy_score():
    html = generate_player_html("dortmund.wav", SEGMENTS, META)
    assert "92%" in html or "0.92" in html


def test_generate_player_html_has_click_to_seek():
    html = generate_player_html("dortmund.wav", SEGMENTS, META)
    assert "currentTime" in html
    assert "play()" in html


def test_generate_player_html_shows_segment_count():
    html = generate_player_html("dortmund.wav", SEGMENTS, META)
    assert "2" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/output/test_html_generator.py -v`
Expected: All 6 tests FAIL with `ModuleNotFoundError: No module named 'output.html_generator'`

- [ ] **Step 3: Implement `output/html_generator.py`**

```python
from __future__ import annotations


def generate_player_html(
    wav_filename: str,
    segments: list[dict],
    meta: dict,
) -> str:
    segment_items = "\n".join(
        _render_segment(i + 1, s) for i, s in enumerate(segments)
    )
    total_str = _format_time(meta.get("duration_total", 0))

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sounds of Block — {meta['file']}</title>
  <style>
    body {{ font-family: monospace; background: #0f172a; color: #e2e8f0;
            padding: 24px; max-width: 800px; margin: 0 auto; }}
    h1 {{ color: #6366f1; font-size: 1.2rem; }}
    .meta {{ color: #64748b; font-size: 0.85rem; margin-bottom: 16px; }}
    audio {{ width: 100%; margin: 16px 0; }}
    .segments {{ list-style: none; padding: 0; }}
    .segment {{ padding: 10px 14px; border-left: 3px solid #6366f1;
                background: #1e293b; margin-bottom: 6px; cursor: pointer; }}
    .segment:hover {{ background: #293548; }}
    .time {{ color: #e2e8f0; font-weight: bold; }}
    .score {{ color: #22c55e; letter-spacing: 2px; }}
    .dur {{ color: #64748b; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>{meta['file']}</h1>
  <p class="meta">{meta['segments_found']} Segmente — Gesamtdauer {total_str}</p>
  <audio id="player" controls src="{wav_filename}"></audio>
  <ul class="segments">
{segment_items}
  </ul>
  <script>
    const player = document.getElementById('player');
    document.querySelectorAll('.segment').forEach(el => {{
      el.addEventListener('click', () => {{
        player.currentTime = parseFloat(el.dataset.start);
        player.play();
      }});
    }});
  </script>
</body>
</html>"""


def _render_segment(index: int, seg: dict) -> str:
    start_str = _format_time(seg["start"])
    end_str = _format_time(seg["end"])
    dur_str = _format_duration(seg["duration"])
    dots = _energy_dots(seg["energy_score"])

    return (
        f'    <li class="segment" data-start="{seg["start"]}">'
        f'<span class="time">{index:02d}. {start_str} — {end_str}</span>'
        f' &nbsp; <span class="score">{dots}</span>'
        f' &nbsp; <span class="dur">{dur_str} &nbsp; {seg["energy_score"]:.0%}</span>'
        f"</li>"
    )


def _energy_dots(score: float) -> str:
    filled = round(score * 5)
    return "●" * filled + "○" * (5 - filled)


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m{s:02d}s"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/output/test_html_generator.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add output/html_generator.py tests/output/test_html_generator.py
git commit -m "feat: add HTML player generator"
```

---

### Task 4: CLI Entry Point — `analyze.py`

**Files:**
- Create: `analyze.py`
- Create: `tests/test_analyze_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analyze_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _write_loud_wav(path: Path, sr: int = 22050) -> None:
    duration = 120
    audio = np.random.normal(0, 0.01, sr * duration).astype(np.float32)
    audio[sr * 20 : sr * 40] = np.random.normal(0, 0.8, sr * 20).astype(np.float32)
    sf.write(str(path), audio, sr)


def test_analyze_creates_json_and_html(tmp_path):
    wav = tmp_path / "test.wav"
    _write_loud_wav(wav)

    result = subprocess.run(
        [sys.executable, "analyze.py", str(wav)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test-segments.json").exists()
    assert (tmp_path / "test-player.html").exists()


def test_analyze_json_has_correct_structure(tmp_path):
    wav = tmp_path / "test.wav"
    _write_loud_wav(wav)

    subprocess.run([sys.executable, "analyze.py", str(wav)], capture_output=True)

    data = json.loads((tmp_path / "test-segments.json").read_text())
    assert "file" in data
    assert "duration_total" in data
    assert "segments_found" in data
    assert "parameters" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)


def test_analyze_html_contains_audio_tag(tmp_path):
    wav = tmp_path / "test.wav"
    _write_loud_wav(wav)

    subprocess.run([sys.executable, "analyze.py", str(wav)], capture_output=True)

    html = (tmp_path / "test-player.html").read_text()
    assert "<audio" in html


def test_analyze_threshold_flag(tmp_path):
    wav = tmp_path / "test.wav"
    _write_loud_wav(wav)

    subprocess.run(
        [sys.executable, "analyze.py", str(wav), "--threshold", "3.0"],
        capture_output=True,
    )

    data = json.loads((tmp_path / "test-segments.json").read_text())
    assert data["parameters"]["threshold"] == 3.0


def test_analyze_prints_summary(tmp_path):
    wav = tmp_path / "test.wav"
    _write_loud_wav(wav)

    result = subprocess.run(
        [sys.executable, "analyze.py", str(wav)],
        capture_output=True, text=True,
    )

    output = result.stdout.lower()
    assert "segment" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analyze_cli.py -v`
Expected: All 5 tests FAIL (`analyze.py` not found)

- [ ] **Step 3: Implement `analyze.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analyze_cli.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest -v --cov=audio --cov=output --cov-report=term-missing`
Expected: All tests pass, coverage > 80%

- [ ] **Step 6: Smoke test with real invocation**

Run: `python analyze.py --help`
Expected: Prints usage/help without errors.

- [ ] **Step 7: Commit**

```bash
git add analyze.py tests/test_analyze_cli.py
git commit -m "feat: add CLI entry point"
```

---

### Task 5: Clip Export — `audio/cutter.py` (Phase 2)

**Files:**
- Create: `audio/cutter.py`
- Create: `tests/audio/test_cutter.py`
- Modify: `requirements.txt`

**Prerequisite:** ffmpeg must be installed on the system.
- Ubuntu/Debian: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

- [ ] **Step 1: Add pydub to `requirements.txt`**

```
librosa>=0.10.0
numpy>=1.24.0
soundfile>=0.12.0
pydub>=0.25.0
```

Run: `pip install pydub`
Expected: Installs without errors.

- [ ] **Step 2: Write failing tests**

Create `tests/audio/test_cutter.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio.cutter import cut_segments


def _write_wav(path: Path, sr: int = 22050, duration: int = 60) -> None:
    audio = np.random.normal(0, 0.3, sr * duration).astype(np.float32)
    sf.write(str(path), audio, sr)


SEGMENTS = [
    {"start": 5.0, "end": 15.0, "duration": 10.0, "energy_score": 0.9},
    {"start": 30.0, "end": 45.0, "duration": 15.0, "energy_score": 0.7},
]


def test_cut_segments_creates_output_directory(tmp_path):
    wav = tmp_path / "test.wav"
    _write_wav(wav)

    cut_segments(str(wav), SEGMENTS)

    assert (tmp_path / "test-clips").exists()


def test_cut_segments_creates_one_file_per_segment(tmp_path):
    wav = tmp_path / "test.wav"
    _write_wav(wav)

    cut_segments(str(wav), SEGMENTS)

    mp3_files = list((tmp_path / "test-clips").glob("*.mp3"))
    assert len(mp3_files) == 2


def test_cut_segments_filenames_include_timestamp(tmp_path):
    wav = tmp_path / "test.wav"
    _write_wav(wav)

    cut_segments(str(wav), SEGMENTS)

    names = [f.name for f in (tmp_path / "test-clips").glob("*.mp3")]
    assert any("00h00m05s" in n for n in names)
    assert any("00h00m30s" in n for n in names)


def test_cut_segments_returns_output_paths(tmp_path):
    wav = tmp_path / "test.wav"
    _write_wav(wav)

    paths = cut_segments(str(wav), SEGMENTS)

    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/audio/test_cutter.py -v`
Expected: All 4 tests FAIL with `ModuleNotFoundError: No module named 'audio.cutter'`

- [ ] **Step 4: Implement `audio/cutter.py`**

```python
from __future__ import annotations

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

        clip.export(str(out_path), format=output_format)
        output_paths.append(str(out_path))

    return output_paths


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/audio/test_cutter.py -v`
Expected: All 4 tests PASS (requires ffmpeg installed)

- [ ] **Step 6: Commit**

```bash
git add audio/cutter.py tests/audio/test_cutter.py requirements.txt
git commit -m "feat: add MP3 clip export"
```

---

### Task 6: Mobile Flask Server — `serve.py` (Phase 2)

**Files:**
- Create: `serve.py`
- Create: `tests/test_serve.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add Flask to `requirements.txt`**

```
librosa>=0.10.0
numpy>=1.24.0
soundfile>=0.12.0
pydub>=0.25.0
flask>=3.0.0
```

Run: `pip install flask`
Expected: Installs without errors.

- [ ] **Step 2: Write failing tests**

Create `tests/test_serve.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from serve import create_app


def _write_wav(path: Path, sr: int = 22050, duration: int = 10) -> None:
    audio = np.random.normal(0, 0.3, sr * duration).astype(np.float32)
    sf.write(str(path), audio, sr)


def _write_segments_json(path: Path, wav_name: str) -> None:
    data = {
        "file": wav_name,
        "duration_total": 10.0,
        "segments_found": 1,
        "parameters": {"threshold": 1.5, "merge_gap": 10, "min_duration": 15},
        "segments": [
            {"start": 1.0, "end": 5.0, "duration": 4.0, "energy_score": 0.9}
        ],
    }
    path.write_text(json.dumps(data))


@pytest.fixture
def client(tmp_path):
    wav = tmp_path / "test.wav"
    _write_wav(wav)
    json_file = tmp_path / "test-segments.json"
    _write_segments_json(json_file, "test.wav")
    # generate a minimal player.html so the index route has something to serve
    html_file = tmp_path / "test-player.html"
    html_file.write_text('<audio id="player" controls src="test.wav"></audio>')

    app = create_app(str(wav), str(json_file))
    app.config["TESTING"] = True
    return app.test_client()


def test_serve_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_serve_index_contains_audio_tag(client):
    response = client.get("/")
    assert b"<audio" in response.data


def test_serve_audio_endpoint_returns_wav(client):
    response = client.get("/audio")
    assert response.status_code == 200
    assert "audio" in response.content_type


def test_serve_audio_by_filename_returns_wav(client):
    response = client.get("/test.wav")
    assert response.status_code == 200
    assert "audio" in response.content_type


def test_serve_segments_returns_json(client):
    response = client.get("/segments")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "segments" in data
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_serve.py -v`
Expected: All 5 tests FAIL with `ModuleNotFoundError: No module named 'serve'`

- [ ] **Step 4: Implement `serve.py`**

```python
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from flask import Flask, Response, send_file


def create_app(wav_path: str, json_path: str) -> Flask:
    app = Flask(__name__)
    _wav = Path(wav_path).resolve()
    _json = Path(json_path).resolve()
    _html = _wav.parent / f"{_wav.stem}-player.html"

    @app.route("/")
    def index() -> Response:
        if _html.exists():
            return _html.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
        return "Player HTML not found. Run analyze.py first.", 404

    @app.route("/audio")
    def audio() -> Response:
        return send_file(str(_wav), mimetype="audio/wav", conditional=True)

    @app.route("/segments")
    def segments() -> Response:
        return Response(_json.read_text(), mimetype="application/json")

    # also serve by original filename — player.html uses a relative src="recording.wav"
    def audio_named() -> Response:
        return send_file(str(_wav), mimetype="audio/wav", conditional=True)

    app.add_url_rule(f"/{_wav.name}", "audio_named", audio_named)

    return app


def _local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the player over the local network for mobile access."
    )
    parser.add_argument("wav", help="Path to WAV file (must already be analysed)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    json_path = wav_path.parent / f"{wav_path.stem}-segments.json"

    if not json_path.exists():
        print(
            f"Error: {json_path.name} not found. Run analyze.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    app = create_app(str(wav_path), str(json_path))

    ip = _local_ip()
    print(f"Server laeuft — oeffne auf dem Smartphone: http://{ip}:{args.port}")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_serve.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Run complete test suite**

Run: `pytest -v --cov=audio --cov=output --cov=serve --cov-report=term-missing`
Expected: All tests pass, coverage > 80%

- [ ] **Step 7: Commit**

```bash
git add serve.py tests/test_serve.py requirements.txt
git commit -m "feat: add Flask mobile server"
```

---

## Usage Summary

After completing all tasks:

```bash
# Analyse a recording
python analyze.py /pfad/zu/spiel.wav

# With custom settings
python analyze.py spiel.wav --threshold 2.0 --top 10

# Analyse + immediately cut clips
python analyze.py spiel.wav --cut

# Serve over local network (phone same WiFi)
python serve.py spiel.wav
# -> Open http://192.168.x.x:5000 on your phone
```
