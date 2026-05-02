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
