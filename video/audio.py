"""Synthesized soundtrack for the demo video (no external audio, no licensing issues).

Music: ambient electronic bed (pads, bass, arpeggio, soft beat) in A minor, ~96 BPM,
with an intensity curve. SFX: whoosh (scene changes), tick/pop (reveals), typing,
chime (results), alarm (kill-switch), boom (outro).
"""
from __future__ import annotations

import wave

import numpy as np

SR = 44100
RNG = np.random.default_rng(5)


def _note(n: int) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def _env(n: int, a: float, r: float) -> np.ndarray:
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / max(r, 1e-4))
    return e


def _place(buf: np.ndarray, sig: np.ndarray, t: float, gain: float = 1.0, pan: float = 0.0) -> None:
    i = int(t * SR)
    if i >= buf.shape[0] or i + len(sig) <= 0:
        return
    s = sig[: max(0, buf.shape[0] - i)]
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    buf[i:i + len(s), 0] += s * gain * l
    buf[i:i + len(s), 1] += s * gain * r


def _bandnoise(n: int, lo: float, hi: float) -> np.ndarray:
    x = RNG.normal(0, 1, n)
    f = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    f[(freqs < lo) | (freqs > hi)] = 0
    y = np.fft.irfft(f, n)
    return y / (np.abs(y).max() + 1e-9)


# ------------------------------------------------------------------ SFX
def whoosh() -> np.ndarray:
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    env = np.sin(np.pi * np.clip(t / 0.9, 0, 1)) ** 2
    lo = _bandnoise(n, 300, 1800)
    hi = _bandnoise(n, 1800, 6000)
    mix = lo * (1 - t / 0.9) + hi * (t / 0.9)
    return mix * env * 0.55


def tick() -> np.ndarray:
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 2200 * t) * np.exp(-t / 0.012) * 0.35


def pop() -> np.ndarray:
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    f = 520 + 480 * (t / 0.16)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05) * 0.5


def typing() -> np.ndarray:
    out = np.zeros(int(1.2 * SR))
    for k in range(9):
        c = _bandnoise(int(0.025 * SR), 2000, 7000) * np.exp(-np.arange(int(0.025 * SR)) / SR / 0.006)
        i = int((k * 0.12 + RNG.uniform(0, 0.04)) * SR)
        out[i:i + len(c)] += c * RNG.uniform(0.12, 0.22)
    return out


def chime() -> np.ndarray:
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f0, d in ((_note(81), 1.0), (_note(88), 0.7), (_note(93), 0.5)):
        for ratio, amp in ((1, 1), (2.76, 0.35), (5.4, 0.15)):
            out += amp * np.sin(2 * np.pi * f0 * ratio * t) * np.exp(-t / (0.9 * d))
    return out / np.abs(out).max() * 0.35


def alarm() -> np.ndarray:
    n = int(1.4 * SR)
    t = np.arange(n) / SR
    f = np.where((t * 4).astype(int) % 2 == 0, 330, 247)
    s = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.25 + np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.5
    return s * np.exp(-t / 0.9) * 0.35


def boom() -> np.ndarray:
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    f = 70 * np.exp(-t / 0.6) + 38
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
    return (s + _bandnoise(n, 40, 400) * np.exp(-t / 0.25) * 0.4) * 0.8


SFX = {"whoosh": whoosh, "tick": tick, "pop": pop, "type": typing, "chime": chime, "alarm": alarm, "boom": boom}
GAIN = {"whoosh": 0.8, "tick": 0.55, "pop": 0.6, "type": 0.9, "chime": 0.75, "alarm": 0.7, "boom": 1.0}


# ------------------------------------------------------------------ music
def music(duration: float, intensity) -> np.ndarray:
    n = int(duration * SR)
    out = np.zeros((n, 2))
    bpm = 96
    beat = 60 / bpm
    bar = 4 * beat
    prog = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [48, 52, 55]), (55, [55, 59, 62])]  # Am F C G
    t_bar = 0.0
    k = 0
    while t_bar < duration:
        root, chord = prog[(k // 2) % 4]
        ln = int(bar * 2 * SR)
        tt = np.arange(ln) / SR
        # pad: additive saw-ish, slow attack/release, slight detune for width
        pad = np.zeros((ln, 2))
        env = np.minimum(1, tt / 1.2) * np.minimum(1, (bar * 2 - tt) / 1.0)
        for note in chord:
            f = _note(note)
            for h, amp in ((1, 1), (2, .35), (3, .18), (4, .08)):
                pad[:, 0] += amp * np.sin(2 * np.pi * f * h * tt * 0.9985)
                pad[:, 1] += amp * np.sin(2 * np.pi * f * h * tt * 1.0015)
        pad *= (env * 0.045)[:, None]
        i = int(t_bar * SR)
        seg = out[i:i + ln]
        seg += pad[: len(seg)]
        level = float(intensity(t_bar))
        for b in range(8):                                  # two bars of beats
            tb = t_bar + b * beat
            if tb >= duration:
                break
            # bass
            bl = int(beat * 0.9 * SR)
            bt = np.arange(bl) / SR
            bass = np.sin(2 * np.pi * _note(root - 12) * bt) * np.exp(-bt / 0.35) * 0.22
            _place(out, bass, tb)
            # soft kick + hat when intensity is up
            if level > 0.35 and b % 2 == 0:
                kl = int(0.25 * SR)
                kt = np.arange(kl) / SR
                kick = np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-kt / 0.04)) / SR) * np.exp(-kt / 0.12)
                _place(out, kick, tb, 0.32 * level)
            if level > 0.5:
                hat = _bandnoise(int(0.04 * SR), 6000, 12000) * np.exp(-np.arange(int(0.04 * SR)) / SR / 0.01)
                _place(out, hat, tb + beat / 2, 0.05 * level, pan=0.3)
            # arpeggio (eighth notes)
            for e in range(2):
                note = chord[(b * 2 + e) % 3] + 12
                al = int(0.32 * SR)
                at = np.arange(al) / SR
                pl = (np.sin(2 * np.pi * _note(note) * at) + 0.3 * np.sin(4 * np.pi * _note(note) * at)) * np.exp(-at / 0.13)
                _place(out, pl, tb + e * beat / 2, 0.05 + 0.06 * level, pan=-0.35 if e else 0.35)
        t_bar += bar * 2
        k += 2
    # simple stereo delay for space
    d = int(0.33 * SR)
    out[d:, 0] += out[:-d, 1] * 0.22
    out[d:, 1] += out[:-d, 0] * 0.22
    return out


def make_track(duration: float, cues: list[dict], path: str, intensity_points: list[tuple[float, float]] | None = None) -> None:
    pts = intensity_points or [(0, .2), (13, .45), (78, .55), (176, .6), (207, .8), (255, .9), (298, .5), (duration, .2)]
    xs, ys = zip(*pts)

    def intensity(t):
        return np.interp(t, xs, ys)

    mus = music(duration, intensity)
    n = mus.shape[0]
    t = np.arange(n) / SR
    fade = np.minimum(1, t / 3.0) * np.minimum(1, (duration - t) / 4.0)
    mus *= fade[:, None] * 0.9
    sfx = np.zeros_like(mus)
    cache = {}
    for c in cues:
        kind = c.get("type")
        if kind not in SFX:
            continue
        if kind not in cache:
            cache[kind] = SFX[kind]()
        _place(sfx, cache[kind], float(c["t"]), GAIN[kind], pan=float(RNG.uniform(-0.25, 0.25)))
    # duck the music under SFX
    env = np.abs(sfx).max(axis=1)
    k = int(0.15 * SR)
    duck = np.convolve(env, np.ones(k) / k, mode="same")
    mix = mus * (1 - np.clip(duck * 1.2, 0, 0.45))[:, None] + sfx
    mix /= max(np.abs(mix).max(), 1e-9) / 0.89
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
