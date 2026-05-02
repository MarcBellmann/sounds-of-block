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
        f'    <li class="segment" data-start="{seg["start"]}" data-end="{seg["end"]}">'
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
