import functools
import os
from typing import Any, List, Optional, Sequence

import numpy as np

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

VOICE_ENV = "HURI_PIPER_VOICE"


@functools.lru_cache(maxsize=4)
def load_voice(path: str) -> Any:
    from piper import PiperVoice

    if not os.path.isfile(path):
        raise ValueError(f"Piper voice not found: {path!r}")
    return PiperVoice.load(path)


def pick(phrases: Sequence[str], said: List[str], rng: np.random.Generator) -> str:
    fresh = [p for p in phrases if p not in said] or list(phrases)
    return fresh[int(rng.integers(len(fresh)))]


class ThinkingVoice:
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
        from piper import SynthesisConfig

        config = SynthesisConfig()
        config.length_scale = float(self.rng.uniform(1.05, 1.35))
        config.noise_scale = float(self.rng.uniform(0.6, 0.9))
        config.noise_w_scale = float(self.rng.uniform(0.8, 1.1))
        chunks = [
            np.asarray(c.audio_float_array, dtype=np.float32)
            for c in self.voice.synthesize(text, syn_config=config)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
