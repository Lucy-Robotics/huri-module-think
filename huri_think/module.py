"""Thinking while the answer is being prepared.

Between a question and the first word of its answer, the LLM (and the TTS
after it) take seconds during which the robot would freeze. These modules
fill that gap with generated behaviour, and stop as soon as the answer is
heard:

    huri.question ──> think      ──> huri.motion     (gaze aversion, head tilt,
                  │                                    thinking poses, fidgets)
                  └─> think_say  ──> huri.audio.out  ("hmm...", "let me think",
                                                       in the TTS's voice)
    huri.token    ──> think_token                    (the answer is written:
                                                       think_say falls silent)
    huri.audio.out ─> think_stop                     (the answer is heard)

`think_token` and `think_stop` tell the others how far the answer got; they
only share the session's user id. Without them, they think until their
`max_s`. The thinking ends with the answer's voice, not its first token: the
TTS waits for a whole sentence, seconds during which the robot would freeze
again.
"""

import asyncio
import time
import weakref
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
from huri_core.core.module import ModuleWithId
from huri_events.audio import Audio
from huri_events.motion import Motion
from huri_events.question import RAGQuestion
from huri_events.text import Token

from huri_think.motion import ThinkingMotion
from huri_think.speech import HUMS, OPENERS, STALLS, ThinkingVoice, pick

_EXPR_DIM = 100  # Motion's facial expression coefficients, left neutral
_POLL_S = 0.05  # how often a waiting module checks that the answer started
_BREATH_S = 1.0  # least silence between two thinking phrases


HEARD, WRITTEN = "heard", "written"


class AnswerSignals:
    """When each user's answer was last written (its tokens) and heard (its
    voice), shared by a session's modules.

    The modules of a session run in the same process (the HuRI deployment),
    and know nothing of each other but the session's user id.
    """

    def __init__(self) -> None:
        self._at: Dict[Tuple[str, str], float] = {}

    def mark(self, user_id: str, at: float, what: str = HEARD) -> None:
        self._at[(user_id, what)] = at

    def since(self, user_id: str, since: float, what: str = HEARD) -> bool:
        return self._at.get((user_id, what), float("-inf")) > since

    def forget(self, user_id: str) -> None:
        for what in (HEARD, WRITTEN):
            self._at.pop((user_id, what), None)


SIGNALS = AnswerSignals()

# The audio think_say sends, which think_stop must not take for the answer.
# The session's bus hands every module the very object a module yielded.
_SPOKEN: "weakref.WeakValueDictionary[int, Audio]" = weakref.WeakValueDictionary()


def _own(audio: Audio) -> Audio:
    _SPOKEN[id(audio)] = audio
    return audio


def _is_own(audio: Audio) -> bool:
    return _SPOKEN.get(id(audio)) is audio


class _Timed(ModuleWithId):
    """The clock the modules of this package share (a fake one in tests)."""

    _clock: Callable[[], float] = staticmethod(time.monotonic)
    _sleep: Callable[[float], Awaitable[None]] = staticmethod(asyncio.sleep)


class _Thinking(_Timed):
    """What `think` and `think_say` share: one episode per question, ended by
    the answer, by a newer question, or by `max_s`."""

    def __init__(self, _user_id: str, delay_s: float, max_s: float, seed):
        super().__init__(_user_id)
        if delay_s < 0 or max_s <= 0:
            raise ValueError("delay_s must be >= 0 and max_s > 0")
        self.delay_s = float(delay_s)
        self.max_s = float(max_s)
        self.rng = np.random.default_rng(seed)
        self._episode = 0

    def _begin(self) -> Tuple[int, float]:
        self._episode += 1
        return self._episode, self._clock()

    def _over(self, episode: int, since: float) -> bool:
        return (
            episode != self._episode
            or SIGNALS.since(self._user_id, since)
            or self._clock() - since >= self.max_s
        )

    async def _wait_until(self, deadline: float, episode: int, since: float) -> bool:
        """Sleep until `deadline`; False as soon as the episode is over."""

        while not self._over(episode, since):
            left = deadline - self._clock()
            if left <= 0:
                return True
            await self._sleep(min(left, _POLL_S))
        return False


class Think(_Thinking):
    """Moves like someone thinking until the answer starts.

    input: huri.question (RAGQuestion)
    output: huri.motion (Motion)

    Streams Motion chunks of generated SMPL-X poses in real time (a little
    ahead of it), pts counting from 0 like an utterance; once the answer
    starts, a last chunk eases back to the neutral pose (arms down).

    :fps: frame rate of the motion.
    :chunk_s: length of each Motion chunk (seconds).
    :lead_s: how far ahead of real time a chunk is sent, to absorb latency.
    :delay_s: nothing moves if the answer starts within this.
    :max_s: stop thinking after this long, answer or not.
    :settle_s: time to ease back to the neutral pose.
    :intensity: amplitude of the motion (0 still, 1 natural, 2 theatrical).
    :arms: allow the thinking poses (hand at the chin, arms crossed...).
    :seed: makes the motion reproducible (random by default).
    """

    input_type = "huri.question"
    output_type = "huri.motion"
    input_event = RAGQuestion
    output_event = Motion

    def __init__(
        self,
        _user_id: str,
        fps: int = 30,
        chunk_s: float = 0.2,
        lead_s: float = 0.15,
        delay_s: float = 0.3,
        max_s: float = 45.0,
        settle_s: float = 0.6,
        intensity: float = 1.0,
        arms: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(_user_id, delay_s, max_s, seed)
        if fps <= 0 or chunk_s <= 0 or lead_s < 0 or settle_s <= 0:
            raise ValueError("fps, chunk_s and settle_s must be > 0, lead_s >= 0")
        self.fps = int(fps)
        self.chunk = max(1, int(round(chunk_s * fps)))
        self.lead_s = float(lead_s)
        self.settle_s = float(settle_s)
        self.intensity = float(intensity)
        self.arms = bool(arms)

    def _motion(self, poses: np.ndarray, pts: float) -> Motion:
        n = poses.shape[0]
        return Motion(
            poses=poses,
            expressions=np.zeros((n, _EXPR_DIM), dtype=np.float32),
            trans=np.zeros((n, 3), dtype=np.float32),
            fps=self.fps,
            pts=pts,
        )

    async def process(self, data: RAGQuestion) -> AsyncGenerator[Motion, None]:
        episode, since = self._begin()
        if not await self._wait_until(since + self.delay_s, episode, since):
            return  # a quick answer needs no thinking
        motion = ThinkingMotion(self.fps, self.intensity, self.arms, self.rng)
        start = self._clock()
        pts = 0.0
        while True:
            yield self._motion(motion.next(self.chunk), pts)
            pts = motion.t
            if not await self._wait_until(start + pts - self.lead_s, episode, since):
                break
        # Even when a newer question took over: its motion starts from the
        # neutral pose, once this one is back there.
        yield self._motion(motion.settle(self.settle_s), pts)


class ThinkSay(_Thinking):
    """Thinks aloud in the voice of the TTS, as much as the wait calls for.

    input: huri.question (RAGQuestion)
    output: huri.audio.out (Audio)

    Nothing for a short pause; a hum ("hmm...") if no answer is written after
    `hum_s`; words ("let me think about that") if still none after
    `think_s`; then "still thinking..." every `stall_s`, `max_stalls` times.
    It falls silent once the answer is being written (think_token): its
    voice comes soon, and a thinking phrase then would delay it.

    Each phrase is synthesized when needed by the Piper voice the TTS speaks
    with (see huri_think.speech), then sent as one Audio chunk and an empty
    one with `end`, like a TTS utterance.

    :voice: path of the Piper voice (.onnx); by default the TTS's,
        `HURI_PIPER_VOICE`.
    :hum_s: wait before a hum.
    :think_s: wait before saying that it thinks.
    :stall_s: time between two "still thinking" afterwards.
    :max_stalls: how many of those at most.
    :max_s: stop after this long, answer or not.
    :seed: makes the choices reproducible (random by default).
    """

    input_type = "huri.question"
    output_type = "huri.audio.out"
    input_event = RAGQuestion
    output_event = Audio

    def __init__(
        self,
        _user_id: str,
        voice: Optional[str] = None,
        hum_s: float = 1.5,
        think_s: float = 4.0,
        stall_s: float = 8.0,
        max_stalls: int = 2,
        max_s: float = 45.0,
        seed: Optional[int] = None,
    ):
        super().__init__(_user_id, hum_s, max_s, seed)
        if not hum_s <= think_s or stall_s <= 0 or max_stalls < 0:
            raise ValueError("need hum_s <= think_s, stall_s > 0, max_stalls >= 0")
        self.voice = ThinkingVoice(voice, self.rng)
        self.plan = [(float(hum_s), HUMS), (float(think_s), OPENERS)] + [
            (think_s + k * stall_s, STALLS) for k in range(1, max_stalls + 1)
        ]

    def _over(self, episode: int, since: float) -> bool:
        return super()._over(episode, since) or SIGNALS.since(
            self._user_id, since, WRITTEN
        )

    async def process(self, data: RAGQuestion) -> AsyncGenerator[Audio, None]:
        episode, since = self._begin()
        sr = self.voice.sample_rate
        said: List[str] = []
        free = since  # when the last phrase ends
        for after, phrases in self.plan:
            due = max(since + after, free + _BREATH_S)
            if not await self._wait_until(due, episode, since):
                return
            text = pick(phrases, said, self.rng)
            said.append(text)
            # Piper is CPU-bound: keep the session's event loop free.
            samples = await asyncio.to_thread(self.voice.say, text)
            if self._over(episode, since):
                return
            yield _own(Audio(data=samples, sample_rate=sr, end=False, pts=0.0))
            yield _own(
                Audio(
                    data=np.zeros(0, dtype=np.float32),
                    sample_rate=sr,
                    end=True,
                    pts=samples.shape[0] / sr,
                )
            )
            free = self._clock() + samples.shape[0] / sr


class ThinkToken(_Timed):
    """Tells `think_say` that the answer is being written: no more phrases.

    input: huri.token (Token)
    output: none
    """

    input_type = "huri.token"
    output_type = None
    input_event = Token

    def __init__(self, _user_id: str):
        super().__init__(_user_id)

    async def process(self, data: Token) -> None:
        SIGNALS.mark(self._user_id, self._clock(), WRITTEN)
        return None


class ThinkStop(_Timed):
    """Tells `think` and `think_say` that the answer is heard.

    input: huri.audio.out (Audio)
    output: none

    Any sound on huri.audio.out counts (the TTS's), but think_say's phrases.
    """

    input_type = "huri.audio.out"
    output_type = None
    input_event = Audio

    def __init__(self, _user_id: str):
        super().__init__(_user_id)

    async def process(self, data: Audio) -> None:
        if data.data.size and not _is_own(data):
            SIGNALS.mark(self._user_id, self._clock(), HEARD)
        return None
