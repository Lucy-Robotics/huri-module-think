"""The thinking phrases, spoken by a real Piper voice when one is available."""

import os

import numpy as np
import pytest

from huri_think.speech import OPENERS, STALLS, VOICE_ENV, ThinkingVoice, pick


def test_pick_says_each_phrase_once_before_repeating():
    rng = np.random.default_rng(0)
    said = []
    for _ in range(len(OPENERS)):
        said.append(pick(OPENERS, said, rng))
    assert sorted(said) == sorted(OPENERS)
    assert pick(OPENERS, said, rng) in OPENERS


@pytest.mark.skipif(
    not os.path.isfile(os.environ.get(VOICE_ENV, "")),
    reason=f"set {VOICE_ENV} to a Piper voice to hear it",
)
def test_a_real_voice_says_every_phrase_differently_each_time():
    voice = ThinkingVoice(rng=np.random.default_rng(0))
    for text in OPENERS + STALLS:
        samples = voice.say(text)
        assert samples.dtype == np.float32 and np.isfinite(samples).all()
        assert 0.3 < samples.size / voice.sample_rate < 4.0
    a, b = voice.say("Hmm..."), voice.say("Hmm...")
    assert a.shape != b.shape or not np.allclose(a, b)
