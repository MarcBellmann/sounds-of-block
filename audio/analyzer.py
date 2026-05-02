from __future__ import annotations

import numpy as np
import librosa


def detect_segments(
    filepath: str,
    threshold: float = 1.5,
    merge_gap: float = 10.0,
    min_duration: float = 15.0,
    top: int | None = None,
) -> list[dict]:
    y, sr = librosa.load(filepath, sr=None, mono=True)

    hop_length = int(sr)           # 1-second hops
    frame_length = int(sr * 5)     # 5-second windows

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    cutoff = float(np.median(rms)) * threshold
    loud_mask = rms >= cutoff

    segments = _mask_to_segments(loud_mask, times, rms)
    segments = _merge_segments(segments, merge_gap)
    segments = [s for s in segments if s["duration"] >= min_duration]

    if not segments:
        return []

    max_rms = max(s["_peak_rms"] for s in segments)
    segments = [
        {
            "start": s["start"],
            "end": s["end"],
            "duration": s["duration"],
            "energy_score": round(s["_peak_rms"] / max_rms, 2),
        }
        for s in segments
    ]

    segments.sort(key=lambda s: s["energy_score"], reverse=True)

    if top is not None:
        segments = segments[:top]

    return segments


def _mask_to_segments(
    loud_mask: np.ndarray,
    times: np.ndarray,
    rms: np.ndarray,
) -> list[dict]:
    segments: list[dict] = []
    in_segment = False
    start = 0.0
    peak_rms = 0.0
    last_loud_t = 0.0

    for i, (is_loud, t) in enumerate(zip(loud_mask, times)):
        if is_loud and not in_segment:
            in_segment = True
            start = float(t)
            peak_rms = float(rms[i])
            last_loud_t = float(t)
        elif is_loud and in_segment:
            peak_rms = max(peak_rms, float(rms[i]))
            last_loud_t = float(t)
        elif not is_loud and in_segment:
            end = last_loud_t
            segments.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(end - start, 2),
                    "_peak_rms": peak_rms,
                }
            )
            in_segment = False

    if in_segment:
        end = last_loud_t
        segments.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "_peak_rms": peak_rms,
            }
        )

    return segments


def _merge_segments(segments: list[dict], merge_gap: float) -> list[dict]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s["start"])
    merged = [ordered[0].copy()]

    for seg in ordered[1:]:
        last = merged[-1]
        if seg["start"] - last["end"] <= merge_gap:
            new_end = max(last["end"], seg["end"])
            merged[-1] = {
                "start": last["start"],
                "end": new_end,
                "duration": round(new_end - last["start"], 2),
                "_peak_rms": max(last["_peak_rms"], seg["_peak_rms"]),
            }
        else:
            merged.append(seg.copy())

    return merged
