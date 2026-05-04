from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

_REQUIRED_ENV = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_PUBLIC_URL",
)


def _r2_client():
    for key in _REQUIRED_ENV:
        if not os.environ.get(key):
            print(f"Error: {key} missing from environment / .env", file=sys.stderr)
            sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _public_url(key: str) -> str:
    base = os.environ["R2_PUBLIC_URL"].rstrip("/")
    return f"{base}/{key}"


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-i", str(wav_path.resolve()),
        "-y",
        "-b:a", "192k",
        "-f", "mp3",
        str(mp3_path.resolve()),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it is on PATH.") from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed: {exc.stderr.decode(errors='replace')}"
        ) from exc


def _upload(client, local_path: Path, key: str, content_type: str) -> None:
    bucket = os.environ["R2_BUCKET"]
    print(f"  Uploading {key} …")
    client.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def _upload_bytes(client, data: bytes, key: str, content_type: str) -> None:
    import io
    bucket = os.environ["R2_BUCKET"]
    print(f"  Uploading {key} …")
    client.upload_fileobj(
        io.BytesIO(data),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def _load_catalog(client) -> dict:
    bucket = os.environ["R2_BUCKET"]
    try:
        resp = client.get_object(Bucket=bucket, Key="catalog.json")
        return json.loads(resp["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"recordings": []}
        raise


def _save_catalog(client, catalog: dict) -> None:
    data = json.dumps(catalog, indent=2, ensure_ascii=False).encode()
    _upload_bytes(client, data, "catalog.json", "application/json")


def _player_deployed(client) -> bool:
    bucket = os.environ["R2_BUCKET"]
    try:
        client.head_object(Bucket=bucket, Key="player/index.html")
        return True
    except ClientError:
        return False


def _deploy_player(client) -> None:
    player_path = Path(__file__).parent / "player" / "index.html"
    if not player_path.exists():
        print("Warning: player/index.html not found — skipping player deploy.", file=sys.stderr)
        return
    _upload(client, player_path, "player/index.html", "text/html; charset=utf-8")
    print(f"  Player: {_public_url('player/index.html')}")


def publish(wav_path: Path, title: str | None) -> None:
    stem = wav_path.stem
    json_path = wav_path.parent / f"{stem}-segments.json"

    if not wav_path.exists():
        print(f"Error: {wav_path.name} not found.", file=sys.stderr)
        sys.exit(1)

    if not json_path.exists():
        print(
            f"Error: {json_path.name} not found. Run analyze.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    meta_raw = json.loads(json_path.read_text(encoding="utf-8"))

    client = _r2_client()

    # Convert WAV → MP3 in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = Path(tmp) / f"{stem}.mp3"
        print(f"Konvertiere {wav_path.name} → MP3 …")
        _wav_to_mp3(wav_path, mp3_path)

        mp3_key = f"recordings/{stem}.mp3"
        json_key = f"recordings/{stem}-segments.json"

        _upload(client, mp3_path, mp3_key, "audio/mpeg")
        _upload(client, json_path, json_key, "application/json")

    mp3_url = _public_url(mp3_key)
    json_url = _public_url(json_key)

    # Update catalog
    catalog = _load_catalog(client)
    existing_ids = {r["id"] for r in catalog["recordings"]}

    if stem not in existing_ids:
        entry = {
            "id": stem,
            "title": title or stem,
            "mp3_url": mp3_url,
            "segments_url": json_url,
            "duration_total": meta_raw.get("duration_total"),
            "segments_found": meta_raw.get("segments_found"),
        }
        catalog["recordings"].append(entry)
        _save_catalog(client, catalog)
        print(f"  Eintrag in catalog.json hinzugefügt: {stem}")
    else:
        print(f"  {stem} bereits in catalog.json — übersprungen.")

    # Deploy player on first publish
    if not _player_deployed(client):
        print("Deploye Player …")
        _deploy_player(client)

    print(f"\nFertig! Player: {_public_url('player/index.html')}")
    print(f"  Audio: {mp3_url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a WAV recording (and its segments JSON) to Cloudflare R2."
    )
    parser.add_argument("wav", help="Path to WAV file (must already be analysed)")
    parser.add_argument("--title", help="Human-readable title shown in the player")
    args = parser.parse_args()

    publish(Path(args.wav).resolve(), args.title)


if __name__ == "__main__":
    main()
