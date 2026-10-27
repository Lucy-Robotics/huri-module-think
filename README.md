# huri-module-think

The robot thinks visibly while the answer is being prepared. Between a
question and the first word of its answer, the LLM (and the TTS after it)
need seconds during which a robot would freeze. These HuRI modules fill that
gap with **generated** behaviour and stop as soon as the answer is heard:

- **motion**: the robot looks away to think, as people do: up and to a side,
  sometimes down. Each glance lasts one to two seconds, with the eyes
  leading the head, the head tilted, and the chest turning and leaning with
  it. It keeps breathing, shifts its weight and never holds perfectly still.
  Within a second or so it usually takes a thinking pose, picked at random
  and changed every few seconds: a hand at the chin with the other arm
  across the belly or on the hip, or the arms crossed, each left- or
  right-handed. In a pose it fidgets: rubs its chin, taps a finger, nods to
  itself.
- **speech**: the robot thinks aloud, in the voice it answers with, as much
  as the wait calls for, like people: nothing for a short pause, a "hmm..."
  when the answer is slow to come, words ("let me think about that", "wait,
  it's going into my brain") when the LLM has still not started after a few
  seconds, then "almost there..." in a long wait. It falls silent as soon as
  the answer is being written, since its voice comes next.

Nothing is recorded. The poses are computed frame by frame from random glance
targets, minimum-jerk transitions, slow noise and breathing
(`huri_think/motion.py`). Each phrase is synthesized when it is needed by the
TTS's own Piper voice, with its pace, intonation and rhythm drawn anew
(`huri_think/speech.py`). No two episodes are the same.

```
huri.question ──> think      ──> huri.motion     ──> mov's consumers (inmoov, ...)
              └─> think_say  ──> huri.audio.out  ──> the speaker
huri.token    ──> think_token    (the answer is written: think_say falls silent)
huri.audio.out ─> think_stop     (the answer is heard: both stop)
```

| Module | Topics | What it does |
|---|---|---|
| `think` | `huri.question` → `huri.motion` | Streams generated SMPL-X thinking poses in real time; when the answer starts, eases back to the neutral pose (arms down) |
| `think_say` | `huri.question` → `huri.audio.out` | Thinks aloud in the TTS's voice: a hum if the answer is slow, words if it is very slow |
| `think_token` | `huri.token` → (nothing) | Tells `think_say` that the answer is being written (first token): no more phrases |
| `think_stop` | `huri.audio.out` → (nothing) | Tells `think` and `think_say` that the answer is heard (the TTS's first sound) |

No GPU. `think` needs numpy only; `think_say` uses the TTS's Piper voice
(ONNX, CPU), about 60 ms per phrase.

## Installing and testing it

In a virtualenv, install HuRI's core and curated events, then the module, and
test it:

```sh
python -m venv .venv && . .venv/bin/activate
python -m pip install \
  "huri-core[server] @ git+https://github.com/Lucy-Robotics/huri-core.git@<ref>" \
  "huri-events @ git+https://github.com/Lucy-Robotics/huri-modules.git@<ref>#subdirectory=packages/huri-events"
./install.sh                            # picks requirements-{cpu,nvidia,amd}.txt for this machine
pytest
```

`<ref>` is a git ref of each repository: `dev` for now, a release tag once
there is one; pick a compatible pair from the table in the
[huri-modules README](https://github.com/Lucy-Robotics/huri-modules#readme).
CI does the same through `HURI_CORE_REF` and `HURI_MODULES_REF`
(`.github/workflows/ci.yml`), and checks the style with `black`, `isort` and
`flake8` (installed by the `dev` extra).

The module introduces no event: it consumes and emits curated ones
(`huri-events>=0.1,<0.3`).

### Hearing and seeing it without a server

```sh
python -m huri_think say thinking.wav --voice en_US-lessac-medium.onnx  # every phrase
python -m huri_think motion thinking.npz               # 8 s of poses (SMPL-X, Motion layout)
```

## Using it in HuRI

On the server, enable the modules next to the pipeline:

```sh
HURI_MODULES=vad,stt,tag,emo,eag,qag,rag,tts,mov,think,think_say,think_token,think_stop
```

```yaml
# client config
modules:
  rag: { name: rag }
  tts: { name: tts }
  think: { name: think, args: { intensity: 1.0, arms: true } }
  think_say: { name: think_say }   # the server sets HURI_PIPER_VOICE for the TTS
  think_token: { name: think_token }
  think_stop: { name: think_stop }
  # ... vad, stt, tag, emo, eag, qag, mov, inmoov
subscribe: [huri.motion, huri.audio.out]
```

Use `think` alone for motion only, or `think_say` alone for speech only. Keep
`think_stop` with either: without it, the thinking runs until `max_s`. It
needs a TTS: in a text-only session, the thinking also runs until `max_s`.
Keep `think_token` with `think_say`: without it, the robot may say "let me
think" while the answer is already written, right before saying it.

With [HuRI-inmoov](https://github.com/Lucy-Robotics/HuRI-inmoov), the
`inmoov` module retargets `think`'s poses like `mov`'s: the head lifts and
turns away, the torso follows, and the arms take the thinking poses when
`arms` is on. Servo limits stay those of the
robot's calibration. The robot's player treats each thinking episode as an
utterance (pts from 0). The speech gestures that follow start a new one, after
the thinking eased back to rest.

### Arguments

`think`:

| Arg | Default | |
|---|---|---|
| `intensity` | 1.0 | amplitude of every angle: 0 still, 1 natural, 2 theatrical |
| `arms` | true | allow the thinking poses (hand at the chin, arms crossed...); off, only the head and torso move |
| `delay_s` | 0.3 | nothing moves if the answer starts within this |
| `max_s` | 45.0 | stop thinking after this, answer or not |
| `settle_s` | 0.6 | time to ease back to the neutral pose (longer with a raised hand) |
| `fps`, `chunk_s`, `lead_s` | 30, 0.2, 0.15 | frame rate, chunk length, how far ahead of real time chunks are sent |
| `seed` | random | a fixed seed replays the same episode |

`think_say`:

| Arg | Default | |
|---|---|---|
| `voice` | `$HURI_PIPER_VOICE` | the Piper voice (.onnx); by default the TTS's |
| `hum_s` | 1.5 | a hum ("hmm...") if the answer is not written after this |
| `think_s` | 4.0 | words ("let me think about that") if it is still not written after this |
| `stall_s` | 8.0 | then "still thinking..." every this long |
| `max_stalls` | 2 | how many "still thinking..." at most |
| `max_s` | 45.0 | stop after this, answer or not |
| `seed` | random | a fixed seed replays the same choices |

### How they work together

`think_token` and `think_stop` record, per user id, when the answer was
written and heard; `think` and `think_say` check it while they wait. The modules of a session run in the same
process, the HuRI deployment, and share nothing else. A thinking episode ends
in three cases: the answer starts, a newer question arrives (it takes over),
or `max_s` passes. The answer starts with its voice, the first sound on
`huri.audio.out` that is not a thinking phrase, rather than its first token:
the TTS waits for a whole sentence, seconds during which the robot would
freeze again.

Things to know:

- `mov` gestures on everything on `huri.audio.out`, so it also gestures over
  the thinking phrases, on the same `huri.motion` as `think`. If the result is
  too busy, use `think` without `think_say`, or `think_say` without `think`.
- A phrase already sent plays to its end (one or two seconds) before the
  answer's audio. Phrases stop at the first token for that reason.
- The phrases are in English, like the default Piper voice; edit `HUMS`,
  `OPENERS` and `STALLS` in `huri_think/speech.py` for another language.
- The voice of a previous answer still speaking when a new question arrives
  ends the new thinking at once.

## What is in it

| Path | What it is |
|---|---|
| `huri_think/module.py` | the three modules and the answer signal they share |
| `huri_think/motion.py` | `ThinkingMotion`: the thinking pose generator (SMPL-X, numpy) |
| `huri_think/speech.py` | `ThinkingVoice`: the thinking phrases, said with a Piper voice |
| `huri_think/__main__.py` | `python -m huri_think`: write the phrases to WAV, poses to .npz |
| `huri_think/manifests/*/huri_module.toml` | one manifest per module, read by HuRI's startup check and the hub |
| `pyproject.toml` | the `huri.modules` entry points (`think`, `think_say`, `think_token`, `think_stop`) |
| `tests/` | the generators, the modules on a fake clock, and the server's startup checks |

## Getting it listed

Tag the repository with the GitHub topic `huri-module` to be found by
`huri hub search` (the `huri-hub` package of
[huri-modules](https://github.com/Lucy-Robotics/huri-modules)). To be listed
as verified, open a pull request to huri-modules adding the modules to its
`registry.toml` (see its ACCEPTANCE.md), pinned at a release tag of this
repository:

```toml
[[module]]
name = "think"
source = { git = "https://github.com/<org>/huri-module-think", tag = "<tag>", submodule = "modules/think" }
```

and the same for `think_say`, `think_token` and `think_stop`. Point `[project.urls]` in
`pyproject.toml` at the repository first.

HuRI as a whole is presented at
[Lucy-Robotics/HuRI-landing](https://github.com/Lucy-Robotics/HuRI-landing).

## License

GPL-3.0, like HuRI: see [LICENSE](LICENSE).
