from typing import Optional


class Vad:
    def __init__(self, vad_onset):
        if not (0 < vad_onset < 1):
            raise ValueError("vad_onset is a decimal value between 0 and 1.")

    @staticmethod
    def preprocess_audio(audio):
        raise NotImplementedError

    @staticmethod
    def merge_chunks(segments, chunk_size, onset: float, offset: Optional[float]):
        """Merge WhisperX speech segments into inference chunks."""
        if not segments:
            return []

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        merged_segments = []
        curr_start = segments[0].start
        curr_end = segments[0].end
        segment_indexes = [(segments[0].start, segments[0].end)]

        for seg in segments[1:]:
            # Keep adjacent speech regions together until the configured chunk
            # duration is reached. The next region starts a new chunk.
            if seg.end - curr_start > chunk_size and curr_end > curr_start:
                merged_segments.append({
                    "start": curr_start,
                    "end": curr_end,
                    "segments": segment_indexes,
                })
                curr_start = seg.start
                segment_indexes = []

            curr_end = seg.end
            segment_indexes.append((seg.start, seg.end))

        merged_segments.append({
            "start": curr_start,
            "end": curr_end,
            "segments": segment_indexes,
        })
        return merged_segments
