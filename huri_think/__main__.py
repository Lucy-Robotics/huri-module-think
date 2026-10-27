"""Try the generators without a server.

    python -m huri_think say thinking.wav --voice en_US-lessac-medium.onnx
    python -m huri_think motion thinking.npz [--seconds 8] [--no-arms]

`say` writes every thinking phrase spoken by a Piper voice (by default
$HURI_PIPER_VOICE, the TTS's), separated by silence, to a 16-bit WAV file.
`motion` writes an episode of generated thinking poses (SMPL-X axis-angle, the
Motion layout) as `poses` (frames x 165), `fps`, and the zero `expressions`
and `trans`, then prints the head's range.
"""

import argparse
import wave
from typing import List, Optional

import numpy as np

from huri_think.motion import HEAD, ThinkingMotion
from huri_think.speech import OPENERS, STALLS, ThinkingVoice


def write_phrases(path: str, voice: Optional[str], seed: Optional[int]) -> None:
    speaker = ThinkingVoice(voice, np.random.default_rng(seed))
    sr = speaker.sample_rate
    parts: List[np.ndarray] = []
    for text in OPENERS + STALLS:
        parts += [speaker.say(text), np.zeros(int(0.7 * sr), dtype=np.float32)]
    samples = np.clip(np.concatenate(parts), -1.0, 1.0)
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes((samples * 32767).astype("<i2").tobytes())
    print(f"{len(OPENERS + STALLS)} phrases, {samples.size / sr:.1f}s -> {path}")


def write_motion(
    path: str, seconds: float, fps: int, arms: bool, seed: Optional[int]
) -> None:
    motion = ThinkingMotion(fps, arms=arms, rng=np.random.default_rng(seed))
    poses = np.concatenate([motion.next(int(seconds * fps)), motion.settle()])
    n = poses.shape[0]
    np.savez(
        path,
        poses=poses,
        expressions=np.zeros((n, 100), np.float32),
        trans=np.zeros((n, 3), np.float32),
        fps=fps,
    )
    head = np.degrees(poses.reshape(n, 55, 3)[:, HEAD])
    print(f"{n} frames at {fps} fps -> {path}")
    for axis, name in enumerate(("pitch", "yaw", "roll")):
        lo, hi = head[:, axis].min(), head[:, axis].max()
        print(f"  head {name}: {lo:+.1f} .. {hi:+.1f} deg")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m huri_think",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="what", required=True)
    say = sub.add_parser("say", help="write every thinking phrase, spoken, to a WAV")
    say.add_argument("path")
    say.add_argument("--voice", help="Piper voice (.onnx), default $HURI_PIPER_VOICE")
    say.add_argument("--seed", type=int)
    mov = sub.add_parser("motion", help="write generated thinking poses to .npz")
    mov.add_argument("path")
    mov.add_argument("--seconds", type=float, default=8.0)
    mov.add_argument("--fps", type=int, default=30)
    mov.add_argument("--no-arms", dest="arms", action="store_false")
    mov.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.what == "say":
        write_phrases(args.path, args.voice, args.seed)
    else:
        write_motion(args.path, args.seconds, args.fps, args.arms, args.seed)


if __name__ == "__main__":
    main()
