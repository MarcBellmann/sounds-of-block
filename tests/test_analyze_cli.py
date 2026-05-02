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
