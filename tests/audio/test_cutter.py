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
