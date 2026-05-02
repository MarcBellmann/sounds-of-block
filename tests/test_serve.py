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
