"""Pyannote-free speaker assignment helpers.

Speaker diarization itself is performed by Atik Voice's dedicated
`diarize==0.1.2` stage. This module only maps externally-produced
(start, end, speaker) rows onto aligned WhisperX transcript segments.
"""
from typing import Optional, Union

import pandas as pd

from whisperx.schema import TranscriptionResult, AlignedTranscriptionResult


class Segment:
    def __init__(self, start: float, end: float, speaker: Optional[str] = None):
        self.start = float(start)
        self.end = float(end)
        self.speaker = speaker


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
):
    """Assign speaker IDs by maximum temporal overlap.

    diarize_df must contain numeric start/end columns and a speaker column.
    No speaker identity merging is performed here.
    """
    if diarize_df is None or diarize_df.empty:
        return transcript_result

    required = {"start", "end", "speaker"}
    missing = required.difference(diarize_df.columns)
    if missing:
        raise ValueError("Diarization table missing columns: " + ", ".join(sorted(missing)))

    rows = [
        (float(row.start), float(row.end), str(row.speaker))
        for row in diarize_df.itertuples(index=False)
    ]

    def choose_speaker(start: float, end: float):
        overlaps = {}
        for d_start, d_end, speaker in rows:
            overlap = min(end, d_end) - max(start, d_start)
            if overlap > 0:
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
        if overlaps:
            return max(overlaps.items(), key=lambda item: item[1])[0]
        if fill_nearest and rows:
            midpoint = (start + end) / 2.0
            nearest = min(rows, key=lambda item: abs(((item[0] + item[1]) / 2.0) - midpoint))
            return nearest[2]
        return None

    for segment in transcript_result.get("segments", []):
        speaker = choose_speaker(float(segment.get("start", 0.0)), float(segment.get("end", 0.0)))
        if speaker is not None:
            segment["speaker"] = speaker
        for word in segment.get("words", []):
            if "start" not in word:
                continue
            word_start = float(word["start"])
            word_end = float(word.get("end", word_start))
            word_speaker = choose_speaker(word_start, word_end)
            if word_speaker is not None:
                word["speaker"] = word_speaker

    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings
    return transcript_result
