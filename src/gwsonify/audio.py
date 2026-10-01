"""Audio signal processing for sonification.

Four distinct operations change how a signal sounds. They are often confused, so
gwsonify keeps them separate and names them precisely:

=====================  ==================  ===================  ======================
operation              duration            frequencies          method
=====================  ==================  ===================  ======================
``speed=X``            divided by X        multiplied by X      band-limited resampling
                                                                ("tape speed")
``stretch=X``          multiplied by X     unchanged            phase vocoder
``pitch=S`` semitones  unchanged           multiplied by 2^S/12 phase vocoder + resampling
``fshift=HZ``          unchanged           +HZ added to each    single-sideband shift of
                                                                the analytic signal
=====================  ==================  ===================  ======================

``speed`` and ``pitch`` preserve frequency *ratios* (a chirp still sounds like the
same chirp, just higher or lower). ``fshift`` adds a constant, so it changes
frequency ratios. It is what the GWOSC audio releases use ("+400 Hz"), because it
lifts the 30-300 Hz signal into the most sensitive part of human hearing while
keeping the timing exact. ``speed`` is the only operation that leaves the waveform
sample values untouched (it is a relabelling of time), so it is the most faithful
choice for teaching.

The pipeline applies the operations in this order: stretch/pitch (at the native
sample rate), then resampling to :data:`AUDIO_RATE` (which also implements speed),
then frequency shift, then fades, then normalisation. Channels are normalised
*jointly*, so relative loudness between detectors is preserved.

References
----------
* Flanagan & Golden (1966), "Phase vocoder", Bell System Technical Journal 45, 1493.
* Laroche & Dolson (1999), "Improved phase vocoder time-scale modification of audio",
  IEEE Trans. Speech Audio Process. 7, 323.
* GWOSC audio releases, https://gwosc.org/audio (whitened, bandpassed, +400 Hz shift).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from gwsonify import GwsonifyError, GwsonifyWarning

AUDIO_RATE = 44100
"""Output sample rate of every WAV file gwsonify writes (Hz)."""


@dataclass
class AudioSettings:
    """How to turn a signal into audio. See the module docstring for definitions.

    Attributes
    ----------
    speed : playback-rate factor (duration / speed, frequencies * speed).
    stretch : duration factor with pitch preserved.
    pitch : pitch shift in semitones with duration preserved.
    fshift : frequency shift in Hz added to every component.
    fade : fade-in and fade-out length in seconds (raised cosine).
    norm : ``"peak"`` (loudest sample at ``level`` dBFS) or ``"fixed"`` (a physical
        reference amplitude ``reference`` maps to ``level`` dBFS, so loudness is
        comparable across runs).
    level : target level in dBFS. ``None`` means -3 (peak) or -30 (fixed).
    reference : amplitude that maps to ``level`` in fixed mode (set by the pipeline:
        1 noise sigma for whitened data, 1e-21 strain for models).
    """

    speed: float = 1.0
    stretch: float = 1.0
    pitch: float = 0.0
    fshift: float = 0.0
    fade: float = 0.05
    norm: str = "peak"
    level: float | None = None
    reference: float = 1.0

    def __post_init__(self) -> None:
        if self.speed <= 0 or self.stretch <= 0:
            raise GwsonifyError("--speed and --stretch must be positive.")
        if self.norm not in ("peak", "fixed"):
            raise GwsonifyError("--norm must be 'peak' or 'fixed'.")
        if self.fade < 0:
            raise GwsonifyError("--fade must be non-negative.")
        if self.level is None:
            self.level = -3.0 if self.norm == "peak" else -30.0

    @property
    def pitch_ratio(self) -> float:
        return 2.0 ** (self.pitch / 12.0)

    @property
    def duration_factor(self) -> float:
        """Output duration / input duration."""
        return self.stretch / self.speed

    @property
    def frequency_map(self) -> tuple[float, float]:
        """``(a, b)`` such that an input frequency f is heard at ``a * f + b``."""
        return self.speed * self.pitch_ratio, self.fshift

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------------------


def resample(x: np.ndarray, rate_in: float, rate_out: float) -> np.ndarray:
    """Band-limited polyphase resampling from ``rate_in`` to ``rate_out`` (Hz).

    The ratio is approximated by a fraction with denominator <= 2000, which is
    accurate to better than 1e-6 for any practical ratio.
    """
    from scipy import signal

    ratio = Fraction(rate_out / rate_in).limit_denominator(2000)
    if ratio == 1:
        return np.asarray(x, dtype=float).copy()
    return signal.resample_poly(np.asarray(x, dtype=float), ratio.numerator,
                                ratio.denominator, axis=0)


def change_speed(x: np.ndarray, fs: float, factor: float) -> np.ndarray:
    """Tape-style speed change at a fixed sample rate: duration / factor, pitch * factor."""
    return resample(x, fs * factor, fs)


def _hann(nfft: int) -> np.ndarray:
    """Periodic Hann window (the STFT-friendly variant)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)


def _stft(x: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    win = _hann(nfft)
    pad = np.pad(x, (nfft, nfft + hop))
    n = 1 + (len(pad) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    return np.fft.rfft(pad[idx] * win, axis=1).T  # (bins, frames)


def _istft(spec: np.ndarray, nfft: int, hop: int, length: int) -> np.ndarray:
    win = _hann(nfft)
    frames = np.fft.irfft(spec.T, n=nfft, axis=1) * win
    n = frames.shape[0]
    out = np.zeros(nfft + hop * (n - 1))
    norm = np.zeros_like(out)
    for i in range(n):
        out[i * hop : i * hop + nfft] += frames[i]
        norm[i * hop : i * hop + nfft] += win**2
    good = norm > 1e-8
    out[good] /= norm[good]
    return out[nfft : nfft + length]


def time_stretch(x: np.ndarray, fs: float, factor: float, *, window: float = 0.1) -> np.ndarray:
    """Change duration by ``factor`` without changing pitch (phase vocoder).

    ``window`` is the analysis window length in seconds (rounded up to a power of two
    in samples). About 0.1 s resolves the 10 Hz-scale structure of a low-frequency
    chirp. Transients near merger are smeared by roughly one window ("phasiness"),
    which is inherent to the phase vocoder.
    """
    x = np.asarray(x, dtype=float)
    if factor == 1:
        return x.copy()
    nfft = int(2 ** np.ceil(np.log2(max(window * fs, 16))))
    hop = nfft // 4
    spec = _stft(x, nfft, hop)
    rate = 1.0 / factor
    steps = np.arange(0, spec.shape[1] - 1, rate)
    omega = 2 * np.pi * hop * np.arange(spec.shape[0]) / nfft  # expected phase advance
    spec = np.concatenate([spec, np.zeros((spec.shape[0], 1), complex)], axis=1)
    bins = np.arange(spec.shape[0])
    phase = np.angle(spec[:, 0])
    out = np.empty((spec.shape[0], len(steps)), dtype=complex)
    for t, step in enumerate(steps):
        i = int(step)
        frac = step - i
        c0, c1 = spec[:, i], spec[:, i + 1]
        mag = (1 - frac) * np.abs(c0) + frac * np.abs(c1)
        # Identity phase locking (Laroche & Dolson 1999): every bin in a peak's region
        # keeps its analysis phase offset from the peak, so the few bins that describe
        # one sinusoid stay coherent, whatever the initial phases were.
        peaks = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])) + 1
        if peaks.size:
            edges = (peaks[:-1] + peaks[1:]) / 2
            owner = peaks[np.searchsorted(edges, bins)]
            locked = phase[owner] + np.angle(c0) - np.angle(c0[owner])
        else:
            locked = phase
        out[:, t] = mag * np.exp(1j * locked)
        dphi = np.angle(c1) - np.angle(c0) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase = locked + omega + dphi
    return _istft(out, nfft, hop, int(round(len(x) * factor)))


def pitch_shift(x: np.ndarray, fs: float, semitones: float, **kw) -> np.ndarray:
    """Shift pitch by ``semitones`` keeping the duration (stretch, then resample)."""
    r = 2.0 ** (semitones / 12.0)
    if r == 1:
        return np.asarray(x, dtype=float).copy()
    y = time_stretch(x, fs, r, **kw)
    y = change_speed(y, fs, r)
    return _fit_length(y, len(x))


def frequency_shift(x: np.ndarray, fs: float, shift: float) -> np.ndarray:
    """Add ``shift`` Hz to every frequency component (single-sideband modulation).

    Computes the analytic signal x + i H[x] (H = Hilbert transform), multiplies by
    exp(2 pi i shift t) and keeps the real part. This is the operation behind the
    GWOSC "+400 Hz" audio files. Components pushed above Nyquist or below 0 Hz would
    alias, so this raises an error when the shift cannot fit.
    """
    from scipy import signal
    from scipy.fft import next_fast_len

    x = np.asarray(x, dtype=float)
    if shift == 0:
        return x.copy()
    if shift >= fs / 2:
        raise GwsonifyError(f"Frequency shift {shift} Hz exceeds Nyquist ({fs / 2} Hz).")
    n = len(x)
    pad = int(0.1 * fs)
    analytic = signal.hilbert(np.pad(x, pad), N=next_fast_len(n + 2 * pad))
    t = np.arange(len(analytic)) / fs
    shifted = np.real(analytic * np.exp(2j * np.pi * shift * t))
    return shifted[pad : pad + n]


def fade(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    """Apply a raised-cosine fade-in and fade-out of ``seconds`` each.

    The first and last samples become exactly zero, which removes the click that an
    abrupt start or stop produces.
    """
    y = np.array(x, dtype=float, copy=True)
    n = min(int(round(seconds * fs)), len(y) // 2)
    if n < 1:
        return y
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
    shape = (n,) + (1,) * (y.ndim - 1)
    y[:n] *= ramp.reshape(shape)
    y[len(y) - n :] *= ramp[::-1].reshape(shape)
    return y


def normalize(x: np.ndarray, norm: str = "peak", level: float = -3.0,
              reference: float = 1.0) -> tuple[np.ndarray, float]:
    """Scale to a target level. Returns ``(scaled, gain)``.

    ``peak``: the largest absolute sample (over all channels) goes to ``level`` dBFS.
    ``fixed``: an amplitude of ``reference`` goes to ``level`` dBFS. The gain does not
    depend on the data, so loudness can be compared across events and detectors.
    Samples beyond full scale are clipped with a :class:`GwsonifyWarning`.
    """
    target = 10.0 ** (level / 20.0)
    if norm == "peak":
        peak = float(np.max(np.abs(x))) if np.size(x) else 0.0
        gain = target / peak if peak > 0 else 1.0
    elif norm == "fixed":
        gain = target / reference
    else:
        raise GwsonifyError(f"Unknown normalisation {norm!r}")
    y = np.asarray(x, dtype=float) * gain
    over = np.abs(y) > 1.0
    if over.any():
        warnings.warn(
            f"{over.mean():.2%} of samples clipped at full scale; lower --level "
            f"(peak would be {20 * np.log10(np.max(np.abs(y))):+.1f} dBFS).",
            GwsonifyWarning, stacklevel=2,
        )
        y = np.clip(y, -1.0, 1.0)
    return y, gain


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


# ---------------------------------------------------------------------------------------
# Full chain
# ---------------------------------------------------------------------------------------


def transform(x: np.ndarray, fs: float, settings: AudioSettings) -> np.ndarray:
    """Apply stretch, pitch, speed and frequency shift; return samples at AUDIO_RATE.

    Does *not* fade or normalise (see :func:`render`), so several channels can be
    normalised together.
    """
    x = np.asarray(x, dtype=float)
    r = settings.pitch_ratio
    # stretch * r then speed * r  ==  stretch with a pitch change of r
    y = time_stretch(x, fs, settings.stretch * r)
    y = resample(y, fs * settings.speed * r, AUDIO_RATE)
    if settings.fshift:
        top = AUDIO_RATE / 2
        if settings.fshift > 0.45 * top:
            raise GwsonifyError(f"--fshift must be below {0.45 * top:.0f} Hz.")
        y = frequency_shift(y, AUDIO_RATE, settings.fshift)
    return y


def render(channels: list[np.ndarray], fs: float, settings: AudioSettings
           ) -> tuple[np.ndarray, float]:
    """Transform, fade and jointly normalise one or more channels.

    Returns ``(audio, gain)`` where ``audio`` has shape ``(n,)`` for one channel or
    ``(n, channels)`` otherwise, at :data:`AUDIO_RATE`, in the range [-1, 1].
    """
    ys = [transform(c, fs, settings) for c in channels]
    n = min(len(y) for y in ys)
    stacked = np.stack([y[:n] for y in ys], axis=1) if len(ys) > 1 else ys[0]
    stacked = fade(stacked, AUDIO_RATE, settings.fade)
    return normalize(stacked, settings.norm, settings.level, settings.reference)


def write_wav(path: str | Path, audio: np.ndarray, rate: int = AUDIO_RATE) -> Path:
    """Write float audio in [-1, 1] as 16-bit PCM WAV (mono or multi-channel)."""
    from scipy.io import wavfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, int(rate), pcm)
    return path
