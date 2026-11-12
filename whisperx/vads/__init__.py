"""Voice activity detection backends supported by this WhisperX fork.

Pyannote has intentionally been removed. Silero is the supported VAD backend.
"""
from whisperx.vads.silero import Silero as Silero
from whisperx.vads.vad import Vad as Vad
