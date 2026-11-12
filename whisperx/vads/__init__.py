"""Voice activity detection backends supported by this WhisperX fork.

Silero is the supported VAD backend.
"""
from whisperx.vads.silero import Silero as Silero
from whisperx.vads.vad import Vad as Vad
