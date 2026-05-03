from __future__ import annotations

import html as _html

from output.format_helpers import format_time


def generate_player_html(
    wav_filename: str,
    segments: list[dict],
    meta: dict,
) -> str:
    segment_items = "\n".join(
        _render_segment(i + 1, s) for i, s in enumerate(segments)
    )
    total_str = format_time(meta.get("duration_total", 0))
    safe_file = _html.escape(meta['file'])
    safe_src = _html.escape(wav_filename, quote=True)
    safe_count = _html.escape(str(meta['segments_found']))

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sounds of Block — {safe_file}</title>
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
  <h1>{safe_file}</h1>
  <p class="meta">{safe_count} Segmente — Gesamtdauer {total_str}</p>
  <audio id="player" controls src="{safe_src}"></audio>
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
    start_str = format_time(seg["start"])
    end_str = format_time(seg["end"])
    dur_str = _format_duration(seg["duration"])
    dots = _energy_dots(seg["energy_score"])

    return (
        f'    <li class="segment" data-start="{float(seg["start"])}" data-end="{float(seg["end"])}">'
        f'<span class="time">{index:02d}. {start_str} — {end_str}</span>'
        f' &nbsp; <span class="score">{dots}</span>'
        f' &nbsp; <span class="dur">{dur_str} &nbsp; {seg["energy_score"]:.0%}</span>'
        f"</li>"
    )


def _energy_dots(score: float) -> str:
    filled = max(0, min(5, round(score * 5)))
    return "●" * filled + "○" * (5 - filled)


def _format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m{s:02d}s"
