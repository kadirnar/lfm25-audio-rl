import numpy as np


def timing_scores(timing, audio_seconds=None, words=None):
    result = {}
    if timing is not None:
        for key in [
            "first_text_seconds",
            "first_audio_token_seconds",
            "first_audio_seconds",
            "generation_seconds",
            "decode_seconds",
            "audio_ready_seconds",
            "peak_memory_bytes",
        ]:
            value = getattr(timing, key)
            if value is not None:
                result[key] = value
        elapsed = timing.audio_ready_seconds
        if elapsed is not None and elapsed > 0 and audio_seconds and audio_seconds > 0:
            result["real_time_factor"] = elapsed / audio_seconds
            result["audio_seconds_per_second"] = audio_seconds / elapsed
        if timing.chunks:
            arrival, duration = timing.chunks[0]
            end = arrival + duration
            stalls = []
            for arrival, duration in timing.chunks[1:]:
                stalls.append(max(0.0, arrival - end))
                end = max(end, arrival) + duration
            result.update(
                stream_stall_seconds=sum(stalls),
                stream_stall_count=sum(s > 0 for s in stalls),
                stream_max_stall_seconds=max(stalls, default=0.0),
                first_playable_chunk_seconds=timing.chunks[0][0],
                chunk_gap_p95_seconds=float(
                    np.quantile(np.diff([c[0] for c in timing.chunks]), 0.95)
                )
                if len(timing.chunks) > 1
                else 0.0,
            )
    if audio_seconds and audio_seconds > 0 and words is not None:
        result["speech_words_per_minute"] = 60 * words / audio_seconds
    return result
