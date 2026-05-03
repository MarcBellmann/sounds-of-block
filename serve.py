from __future__ import annotations

import argparse
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
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


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
