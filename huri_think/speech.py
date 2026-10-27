"""Thinking aloud in the robot's own voice.

People who need a moment say so, more as the wait grows: "hmm...", then "let
me think", "wait, it's going into my brain", then "almost there...".
`ThinkingVoice` says such phrases with the Piper voice the TTS speaks the
answer with, so the filler and the answer sound like the same person.
Nothing is recorded: Piper synthesizes each phrase when it is needed (about
60 ms on a CPU), and its delivery (pace, intonation, rhythm) is drawn anew
every time.
"""

import functools
import os
from typing import Any, List, Optional, Sequence

import numpy as np

# What is said, by how long the answer has been awaited. People do not
# comment on a short pause; a slow answer gets a hum; only a long one gets
# words, and only a very long one gets "still...".
HUMS = (
    "Hmm.",
    "Hmm...",
    "Hmmm.",
    "Um...",
    "Uh...",
    "Erm...",
    "Ah, hmm...",
)
OPENERS = (
    "Hmm, let me think about that.",
    "Wait, it's going into my brain.",
    "Hold on, let me think.",
    "Let me see...",
    "Give me a second.",
    "Hmm, wait a second.",
)
STALLS = (
    "Still thinking...",
    "Almost there...",
    "It's coming, it's coming.",
    "My brain is working on it.",
    "Bear with me.",
    "Nearly got it.",
)

# The environment variable naming the TTS's Piper voice (see huri_tts.piper).
VOICE_ENV = "HURI_PIPER_VOICE"


@functools.lru_cache(maxsize=4)
def load_voice(path: str) -> Any:
    """The Piper voice at `path`, loaded once per process (sessions share it)."""

    from piper import PiperVoice

    if not os.path.isfile(path):
        raise ValueError(f"Piper voice not found: {path!r}")
    return PiperVoice.load(path)


def pick(phrases: Sequence[str], said: List[str], rng: np.random.Generator) -> str:
    """A phrase of `phrases` not said yet in this episode (any, once all were)."""

    fresh = [p for p in phrases if p not in said] or list(phrases)
    return fresh[int(rng.integers(len(fresh)))]


class ThinkingVoice:
    """Says thinking phrases with a Piper voice.

    :voice: the path of a Piper voice (.onnx), by default the TTS's
        (`HURI_PIPER_VOICE`).
    :rng: the source of every choice (seed it for reproducible choices).
    """

    def __init__(
        self, voice: Optional[str] = None, rng: Optional[np.random.Generator] = None
    ):
        path = voice or os.environ.get(VOICE_ENV)
        if not path:
            raise ValueError(
                f"no Piper voice: pass `voice` or set {VOICE_ENV} like the TTS"
            )
        self.voice = load_voice(path)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.rng = rng if rng is not None else np.random.default_rng()

    def say(self, text: str) -> np.ndarray:
        """`text` spoken with a fresh delivery, float32 samples."""

        from piper import SynthesisConfig

        config = SynthesisConfig()
        config.length_scale = float(self.rng.uniform(1.05, 1.35))  # slower: musing
        config.noise_scale = float(self.rng.uniform(0.6, 0.9))  # intonation
        config.noise_w_scale = float(self.rng.uniform(0.8, 1.1))  # rhythm
        chunks = [
            np.asarray(c.audio_float_array, dtype=np.float32)
            for c in self.voice.synthesize(text, syn_config=config)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
