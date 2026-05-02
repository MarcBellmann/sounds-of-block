# Sounds of Block — Design Spec

**Date:** 2026-05-02  
**Status:** Approved

## Overview

A personal Python CLI tool that analyses 90-minute stadium recording WAV files, detects the most energetic (loudest) segments, and outputs a timestamp list plus a browser-based HTML player for previewing those segments. Phase 2 adds clip export and mobile access via a local Flask server.

## Goals

- Reduce manual effort of finding the best moments in long stadium recordings
- Personal tool — simplicity over polish
- PC-first, mobile-later

## Out of Scope

- Cloud storage or sharing
- Automatic social media publishing
- Multi-file batch processing (one file at a time)

---

## Architecture

### Phase 1 — CLI + HTML Player

```
recording.wav
    │
    ▼
analyze.py          ← entry point: python analyze.py recording.wav
    │
    ├── audio/analyzer.py     ← energy detection (librosa)
    │
    └── output/html_generator.py  ← builds player.html
    │
    ├── segments.json         ← output: timestamps + energy scores
    └── player.html           ← output: browser player
```

### Phase 2 — Clip Export + Mobile

```
cutter.py           ← python cut.py recording.wav → exports MP3 clips
serve.py            ← Flask server, accessible from phone on same WiFi
```

### File Layout

```
sounds-of-block/
├── analyze.py
├── audio/
│   ├── __init__.py
│   ├── analyzer.py
│   └── cutter.py          (Phase 2)
├── output/
│   ├── __init__.py
│   └── html_generator.py
├── serve.py               (Phase 2)
└── requirements.txt
```

---

## Components

### `analyze.py` — Entry Point

CLI interface. Accepts a WAV file path, calls the analyzer, writes outputs next to the input file.

```
Usage: python analyze.py <path/to/recording.wav>

Outputs:
  recording-segments.json
  recording-player.html
```

### `audio/analyzer.py` — Energy Detection

Detects high-energy segments using RMS (root mean square) energy over sliding windows.

**Algorithm:**
1. Load WAV file with librosa
2. Compute RMS energy in fixed windows (5 seconds each)
3. Calculate median energy of the entire recording as baseline
4. Mark windows exceeding `threshold_multiplier × median` as loud (default: 1.5×)
5. Merge adjacent loud windows separated by gaps shorter than `merge_gap` (default: 10 seconds)
6. Drop segments shorter than `min_duration` (default: 15 seconds)
7. Sort by energy score descending

**Output per segment:**
```json
{
  "start": 262.0,
  "end": 315.5,
  "duration": 53.5,
  "energy_score": 0.92
}
```

**Tunable parameters** (CLI flags):
- `--threshold` — multiplier over median (default: 1.5)
- `--merge-gap` — seconds of quiet allowed within a segment (default: 10)
- `--min-duration` — minimum segment length in seconds (default: 15)
- `--top` — only keep top N segments (default: all)

### `output/html_generator.py` — HTML Player

Generates a self-contained HTML file with:
- Standard HTML5 `<audio>` element pointing to the WAV file (relative path)
- Sorted list of segments with start time, end time, duration, and energy score displayed as filled circles (●●●●○)
- Clicking a segment seeks the audio player to that timestamp and starts playback
- No external dependencies — pure HTML/JS

### `audio/cutter.py` — Clip Export (Phase 2)

Reads `segments.json`, cuts the WAV at each segment boundary using pydub, exports as MP3.

```
Usage: python cut.py recording.wav
Reads:  recording-segments.json
Output: recording-clips/
          01-00h04m22s.mp3
          02-00h12m08s.mp3
          ...
```

### `serve.py` — Mobile Flask Server (Phase 2)

Minimal Flask app that serves the player.html and streams the audio file over HTTP. Accessible from any device on the same local network.

```
Usage: python serve.py recording.wav
Opens: http://<local-ip>:5000
```

---

## Technology Stack

| Library | Purpose |
|---------|---------|
| librosa | Audio loading, RMS energy computation |
| numpy | Array operations for energy windowing |
| pydub | Audio cutting and MP3 export (Phase 2) |
| Flask | Mobile web server (Phase 2) |

Python 3.9+ required.

---

## Output Format

### `segments.json`

```json
{
  "file": "dortmund-vs-schalke.wav",
  "duration_total": 5530.4,
  "segments_found": 17,
  "parameters": {
    "threshold": 1.5,
    "merge_gap": 10,
    "min_duration": 15
  },
  "segments": [
    { "start": 262.0, "end": 315.5, "duration": 53.5, "energy_score": 0.92 },
    { "start": 728.0, "end": 824.1, "duration": 96.1, "energy_score": 0.81 }
  ]
}
```

### `player.html`

Self-contained single HTML file. No build step, no external CDN. Opens directly in any browser by double-clicking.

---

## Future Considerations

- Waveform visualisation (low priority, Phase 3)
- Adjustable threshold via slider in the HTML player itself
- Support for MP3 input in addition to WAV
