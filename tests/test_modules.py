"""The modules: thinking starts with a question and ends with the answer.

Time is fake: sleeping advances the clock instantly.
"""

import asyncio
import itertools

import numpy as np
import pytest
from huri_core.core.compat import check_manifests, check_modules, check_session
from huri_core.core.plugins import find_manifest, load_events, load_modules
from huri_core.testing import ModuleHarness
from huri_events.audio import Audio
from huri_events.motion import Motion
from huri_events.question import RAGQuestion
from huri_events.text import Token

from huri_think import module as think_module
from huri_think import speech
from huri_think.module import SIGNALS, WRITTEN, Think, ThinkSay, ThinkStop, ThinkToken
from huri_think.motion import REST_POSE

NAMES = ["think", "think_say", "think_token", "think_stop"]
QUESTION = RAGQuestion.from_wire({"text": "what is the meaning of life?"})


def _voice(text="Well,"):
    """Audio of the answer, as the TTS sends it."""

    return Audio(np.full(len(text) * 100, 0.1, np.float32), 22050, False, 0.0)


_users = itertools.count()


class FakeTime:
    def __init__(self):
        self.now = 1000.0

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.now += max(seconds, 0.0)
        await asyncio.sleep(0)


@pytest.fixture
def fake_time(monkeypatch):
    t = FakeTime()
    monkeypatch.setattr(think_module._Timed, "_clock", staticmethod(t.clock))
    monkeypatch.setattr(think_module._Timed, "_sleep", staticmethod(t.sleep))
    return t


class FakeVoice:
    """Stands for a Piper voice: 0.1 s of a tone per character said."""

    class config:
        sample_rate = 22050

    def __init__(self):
        self.said = []

    def synthesize(self, text, syn_config=None):
        self.said.append(text)
        n = int(0.1 * 22050) * len(text)
        tone = np.sin(np.arange(n) * 0.05).astype(np.float32)
        yield type("Chunk", (), {"audio_float_array": tone})()


@pytest.fixture(autouse=True)
def fake_voice(monkeypatch):
    voice = FakeVoice()
    monkeypatch.setattr(speech, "load_voice", lambda path: voice)
    monkeypatch.setenv(speech.VOICE_ENV, "voices/fake.onnx")
    return voice


@pytest.fixture
def user():
    user_id = f"user-{next(_users)}"
    yield user_id
    SIGNALS.forget(user_id)


async def _collect(module, question, fake_time, answer_after=None, user=None):
    """Everything `module` emits for `question`, with the answer starting
    `answer_after` seconds later (never when None), and when each came."""

    start = fake_time.now

    async def answer():
        while fake_time.now - start < answer_after:
            await asyncio.sleep(0)  # the module's sleeps move the clock
        await ThinkStop(_user_id=user).process(_voice("Well,"))

    answering = asyncio.ensure_future(answer()) if answer_after is not None else None
    out = []
    async for item in module.process(question):
        out.append((fake_time.now - start, item))
    if answering is not None:
        answering.cancel()
    return out


def test_think_moves_until_the_answer_then_settles(fake_time, user):
    think = Think(_user_id=user, seed=0)
    out = asyncio.run(_collect(think, QUESTION, fake_time, 3.0, user))
    times, motions = zip(*out)
    assert all(isinstance(m, Motion) and m.fps == 30 for m in motions)
    # Starts after delay_s, continues in time, ends soon after the answer.
    assert times[0] == pytest.approx(0.3, abs=0.06)
    pts = [m.pts for m in motions]
    assert pts[0] == 0.0 and pts == sorted(pts)
    for a, b in zip(motions, motions[1:]):
        assert b.pts == pytest.approx(a.pts + a.poses.shape[0] / a.fps)
    assert 3.0 <= times[-1] <= 3.0 + 0.25
    np.testing.assert_allclose(motions[0].poses[0], REST_POSE, atol=1e-6)
    np.testing.assert_allclose(motions[-1].poses[-1], REST_POSE, atol=1e-6)
    assert motions[-1].poses.shape[0] >= 0.6 * 30  # the settle chunk


def test_think_is_sent_a_little_ahead_of_real_time(fake_time, user):
    think = Think(_user_id=user, seed=0, lead_s=0.15)
    out = asyncio.run(_collect(think, QUESTION, fake_time, 4.0, user))
    start = out[0][0]
    for sent, motion in out[:-1]:
        ahead = motion.pts - (sent - start)
        assert 0.0 <= ahead <= 0.15 + 0.06


def test_a_quick_answer_needs_no_thinking(fake_time, user):
    async def scenario():
        think = Think(_user_id=user, seed=0, delay_s=0.3)
        gen = think.process(QUESTION)
        first = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        fake_time.now += 0.1
        await ThinkStop(_user_id=user).process(_voice("Yes."))
        with pytest.raises(StopAsyncIteration):
            await first

    asyncio.run(scenario())


def test_an_earlier_answer_does_not_stop_a_new_question(fake_time, user):
    asyncio.run(ThinkStop(_user_id=user).process(_voice("Done.")))
    fake_time.now += 1.0
    think = Think(_user_id=user, seed=0, max_s=2.0)
    out = asyncio.run(_collect(think, QUESTION, fake_time))
    assert len(out) > 5


def test_another_user_answer_does_not_stop_thinking(fake_time, user):
    think = Think(_user_id=user, seed=0, max_s=2.0)

    async def scenario():
        out = []
        async for item in think.process(QUESTION):
            out.append(item)
            await ThinkStop(_user_id=user + "-other").process(_voice("Hi"))
        return out

    out = asyncio.run(scenario())
    assert sum(m.poses.shape[0] for m in out) >= 2.0 * 30


def test_think_gives_up_after_max_s(fake_time, user):
    think = Think(_user_id=user, seed=0, max_s=2.0)
    out = asyncio.run(_collect(think, QUESTION, fake_time))
    assert out[-1][0] == pytest.approx(2.0, abs=0.25)
    np.testing.assert_allclose(out[-1][1].poses[-1], REST_POSE, atol=1e-6)


def test_a_new_question_takes_over(fake_time, user):
    async def scenario():
        think = Think(_user_id=user, seed=0, max_s=5.0)
        first = []
        async for item in think.process(QUESTION):
            first.append(item)
            if len(first) == 3:
                second = think.process(QUESTION)  # the next question arrives
                next_one = asyncio.ensure_future(second.__anext__())
        await next_one
        await second.aclose()
        return first

    first = asyncio.run(scenario())
    assert len(first) == 4  # three chunks, then it settled at once
    np.testing.assert_allclose(first[-1].poses[-1], REST_POSE, atol=1e-6)


def test_think_say_hums_when_the_answer_is_slow(fake_time, user, fake_voice):
    say = ThinkSay(_user_id=user, seed=0, hum_s=1.5)
    out = asyncio.run(_collect(say, QUESTION, fake_time, 2.5, user))
    times, audios = zip(*out)
    assert all(isinstance(a, Audio) and a.sample_rate == 22050 for a in audios)
    assert times[0] == pytest.approx(1.5, abs=0.06)
    # One hum and its end marker, then the answer came.
    assert [a.end for a in audios] == [False, True]
    assert audios[0].data.size > 0 and audios[1].data.size == 0
    assert audios[1].pts == pytest.approx(audios[0].data.size / 22050)
    assert fake_voice.said[0] in speech.HUMS


def test_think_say_says_more_as_the_wait_grows(fake_time, user, fake_voice):
    say = ThinkSay(_user_id=user, seed=0, hum_s=1.5, think_s=4.0, stall_s=8.0)
    out = asyncio.run(_collect(say, QUESTION, fake_time))
    times = [t for t, a in out if not a.end]
    assert times == pytest.approx([1.5, 4.0, 12.0, 20.0], abs=0.06)
    hum, opener, *stalls = fake_voice.said
    assert hum in speech.HUMS and opener in speech.OPENERS
    assert all(p in speech.STALLS for p in stalls) and len(set(stalls)) == 2


def test_think_say_leaves_a_breath_between_phrases(fake_time, user, fake_voice):
    say = ThinkSay(_user_id=user, seed=0, hum_s=1.0, think_s=1.1, max_stalls=0)
    out = asyncio.run(_collect(say, QUESTION, fake_time))
    (t0, a0), (t1, _) = [(t, a) for t, a in out if not a.end]
    assert t1 - t0 == pytest.approx(a0.data.size / 22050 + 1.0, abs=0.06)


def test_think_say_is_silent_after_a_quick_answer(fake_time, user, fake_voice):
    out = asyncio.run(_collect(ThinkSay(_user_id=user), QUESTION, fake_time, 1.0, user))
    assert out == [] and fake_voice.said == []


def test_think_say_falls_silent_once_the_answer_is_written(fake_time, user, fake_voice):
    """Its voice comes soon: a phrase now would only delay it."""

    say = ThinkSay(_user_id=user, seed=0, hum_s=1.5)

    async def scenario():
        async def write():
            while fake_time.now < start + 1.0:
                await asyncio.sleep(0)
            await ThinkToken(_user_id=user).process(Token("The", False))

        start = fake_time.now
        writing = asyncio.ensure_future(write())
        out = [a async for a in say.process(QUESTION)]
        writing.cancel()
        return out

    assert asyncio.run(scenario()) == [] and fake_voice.said == []
    assert SIGNALS.since(user, 0.0, WRITTEN) and not SIGNALS.since(user, 0.0)


def test_think_keeps_moving_while_the_answer_is_written(fake_time, user):
    asyncio.run(ThinkToken(_user_id=user).process(Token("The", False)))
    fake_time.now += 1.0
    think = Think(_user_id=user, seed=0, max_s=2.0)

    async def scenario():
        out = []
        async for motion in think.process(QUESTION):
            out.append(motion)
            await ThinkToken(_user_id=user).process(Token(" sky", False))
        return out

    out = asyncio.run(scenario())
    assert sum(m.poses.shape[0] for m in out) >= 2.0 * 30


def test_think_say_does_not_stop_itself(fake_time, user):
    """Its phrases go to think_stop too: they are not the answer."""

    say = ThinkSay(_user_id=user, seed=0, hum_s=0.1, think_s=0.2, max_stalls=0)
    stop = ThinkStop(_user_id=user)

    async def scenario():
        out = []
        async for audio in say.process(QUESTION):
            out.append(audio)
            await stop.process(audio)
        return out

    assert len(asyncio.run(scenario())) == 4  # hum and words, each with its end


def test_only_sound_tells_the_answer_started(fake_time, user):
    asyncio.run(ThinkStop(_user_id=user).process(Audio(np.zeros(0), 22050, True)))
    assert not SIGNALS.since(user, 0.0)
    asyncio.run(ThinkStop(_user_id=user).process(_voice()))
    assert SIGNALS.since(user, 0.0)


def test_think_say_needs_a_voice(monkeypatch):
    monkeypatch.delenv(speech.VOICE_ENV)
    with pytest.raises(ValueError, match="HURI_PIPER_VOICE"):
        ThinkSay(_user_id="u")


def test_harness_checks_the_declared_events(fake_time):
    out = ModuleHarness(Think, seed=0, max_s=1.0).run(QUESTION)
    assert out and all(isinstance(m, Motion) for m in out)
    assert ModuleHarness(ThinkStop).run(_voice("a")) == []


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (Think, {"fps": 0}),
        (Think, {"max_s": 0}),
        (Think, {"delay_s": -1}),
        (ThinkSay, {"hum_s": 5.0, "think_s": 4.0}),
        (ThinkSay, {"stall_s": 0}),
        (ThinkSay, {"max_stalls": -1}),
    ],
)
def test_bad_args_are_refused(cls, kwargs):
    with pytest.raises(ValueError):
        cls(_user_id="u", **kwargs)


def test_installed_modules_pass_the_server_checks():
    """What the server verifies at startup: entry points, events, manifests."""

    modules = load_modules(NAMES)
    events = load_events()
    check_modules(modules, events)
    manifests = {name: find_manifest(name) for name in NAMES}
    assert None not in manifests.values()
    check_manifests(modules, manifests, events)


def test_a_session_with_rag_is_complete():
    """think_stop needs the voice of a TTS; think starts from questions."""

    class Rag:
        input_type, output_type = "huri.question", "huri.token"

    class Tts:
        input_type, output_type = "huri.token", "huri.audio.out"

    pipeline = [
        Rag(),
        Tts(),
        Think(_user_id="u"),
        ThinkSay(_user_id="u"),
        ThinkToken(_user_id="u"),
        ThinkStop(_user_id="u"),
    ]
    check_session(
        pipeline, sent=["huri.question"], subscribed=["huri.motion", "huri.audio.out"]
    )
