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
