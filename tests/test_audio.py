"""Analytic tests of the audio DSP (no network, no GW dependencies)."""

import numpy as np
import pytest
from scipy.io import wavfile

from gwsonify import GwsonifyError, GwsonifyWarning
from gwsonify import audio as A

FS = 4096


def sine(freq, dur=4.0, fs=FS, amp=1.0):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def dominant_freq(x, fs):
    """Peak frequency with parabolic interpolation (sub-bin accuracy)."""
    w = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(w, n=8 * len(x)))
    k = int(np.argmax(spec))
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-300)
    k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return k * fs / (8 * len(x))


def core(x, frac=0.2):
    """Drop edges to avoid window/filter transients."""
    n = int(len(x) * frac)
    return x[n : len(x) - n]


# -- individual operations ----------------------------------------------------------------


@pytest.mark.parametrize("shift", [50.0, 400.0, -30.0])
def test_frequency_shift_moves_sine_by_exact_hz(shift):
    x = sine(200.0)
    y = A.frequency_shift(x, FS, shift)
    assert len(y) == len(x)
    assert dominant_freq(core(y), FS) == pytest.approx(200.0 + shift, abs=0.2)
    # SSB modulation preserves the envelope of a pure tone
    assert np.std(core(y)) == pytest.approx(np.std(core(x)), rel=0.01)


def test_frequency_shift_adds_constant_not_ratio():
    x = sine(100.0) + sine(300.0)
    y = A.frequency_shift(x, FS, 100.0)
    spec = np.abs(np.fft.rfft(core(y) * np.hanning(len(core(y)))))
    freqs = np.fft.rfftfreq(len(core(y)), 1 / FS)
    peaks = sorted(freqs[np.argsort(spec)[-2:]])
    assert peaks == pytest.approx([200.0, 400.0], abs=1.0)  # not [200, 600]


def test_frequency_shift_rejects_above_nyquist():
    with pytest.raises(GwsonifyError):
        A.frequency_shift(sine(100.0), FS, FS)


@pytest.mark.parametrize("factor", [2.0, 0.5, 3.3])
def test_speed_scales_duration_and_frequency(factor):
    x = sine(150.0)
    y = A.change_speed(x, FS, factor)
    assert len(y) == pytest.approx(len(x) / factor, abs=2)
    assert dominant_freq(core(y), FS) == pytest.approx(150.0 * factor, rel=1e-3)


@pytest.mark.parametrize("factor", [2.0, 4.0, 0.5])
def test_time_stretch_keeps_frequency_changes_duration(factor):
    x = sine(150.0)
    y = A.time_stretch(x, FS, factor)
    assert len(y) == round(len(x) * factor)
    assert dominant_freq(core(y), FS) == pytest.approx(150.0, rel=2e-3)
    # amplitude of a steady tone is preserved
    assert np.std(core(y)) == pytest.approx(np.std(core(x)), rel=0.05)


@pytest.mark.parametrize("semitones", [12.0, -12.0, 7.0])
def test_pitch_shift_keeps_duration_changes_frequency(semitones):
    x = sine(160.0)
    y = A.pitch_shift(x, FS, semitones)
    assert len(y) == len(x)
    assert dominant_freq(core(y), FS) == pytest.approx(160.0 * 2 ** (semitones / 12), rel=3e-3)


def test_fade_is_click_free():
    x = np.ones(FS)  # worst case: DC step at both ends
    y = A.fade(x, FS, 0.05)
    assert y[0] == 0.0
    assert y[-1] == pytest.approx(0.0, abs=1e-3)
    # the largest sample-to-sample jump is tiny compared with an abrupt 0 -> 1 step
    assert np.max(np.abs(np.diff(y))) < 0.01
    assert np.array_equal(y[len(y) // 2], x[len(x) // 2])


def test_fade_multichannel():
    y = A.fade(np.ones((FS, 2)), FS, 0.1)
    assert np.all(y[0] == 0) and np.all(y[FS // 2] == 1)


def test_peak_normalisation_level():
    y, gain = A.normalize(sine(100.0, amp=3e-21), "peak", -6.0)
    assert np.max(np.abs(y)) == pytest.approx(10 ** (-6 / 20), rel=1e-6)
    assert gain == pytest.approx(10 ** (-6 / 20) / 3e-21, rel=1e-3)


def test_fixed_normalisation_is_data_independent_and_comparable():
    loud, g1 = A.normalize(sine(100.0, amp=2.0), "fixed", -20.0, reference=1.0)
    quiet, g2 = A.normalize(sine(100.0, amp=0.5), "fixed", -20.0, reference=1.0)
    assert g1 == g2
    ratio_db = 20 * np.log10(np.max(np.abs(loud)) / np.max(np.abs(quiet)))
    assert ratio_db == pytest.approx(20 * np.log10(4), abs=1e-3)


def test_clipping_warns():
    with pytest.warns(GwsonifyWarning, match="clipped"):
        y, _ = A.normalize(sine(100.0, amp=10.0), "fixed", 0.0, reference=1.0)
    assert np.max(np.abs(y)) <= 1.0


# -- full chain ---------------------------------------------------------------------------


def test_render_defaults_resample_to_audio_rate_preserving_everything():
    x = sine(120.0, dur=2.0)
    audio, _ = A.render([x], FS, A.AudioSettings())
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(2.0 * A.AUDIO_RATE, abs=2)
    assert dominant_freq(core(audio), A.AUDIO_RATE) == pytest.approx(120.0, rel=1e-3)
    assert np.max(np.abs(audio)) == pytest.approx(10 ** (-3 / 20), rel=1e-3)
    assert audio[0] == 0.0


def test_render_combined_operations_follow_frequency_map():
    s = A.AudioSettings(speed=0.5, stretch=2.0, pitch=12.0, fshift=100.0)
    a, b = s.frequency_map
    assert (a, b) == (pytest.approx(1.0), 100.0)
    x = sine(200.0, dur=2.0)
    audio, _ = A.render([x], FS, s)
    assert len(audio) == pytest.approx(2.0 * s.duration_factor * A.AUDIO_RATE, rel=1e-3)
    assert dominant_freq(core(audio), A.AUDIO_RATE) == pytest.approx(a * 200 + b, rel=3e-3)


def test_render_stereo_joint_normalisation_preserves_relative_loudness():
    left, right = sine(100.0, amp=1.0), sine(100.0, amp=0.25)
    audio, _ = A.render([left, right], FS, A.AudioSettings())
    assert audio.shape[1] == 2
    ratio = np.max(np.abs(core(audio[:, 0]))) / np.max(np.abs(core(audio[:, 1])))
    assert ratio == pytest.approx(4.0, rel=1e-2)


def test_settings_validation():
    with pytest.raises(GwsonifyError):
        A.AudioSettings(speed=0)
    with pytest.raises(GwsonifyError):
        A.AudioSettings(norm="loud")
    assert A.AudioSettings(norm="fixed").level == -30.0


def test_write_wav_roundtrip(tmp_path):
    x = 0.5 * sine(440.0, dur=0.5, fs=A.AUDIO_RATE)
    path = A.write_wav(tmp_path / "a" / "t.wav", np.stack([x, -x], axis=1))
    rate, data = wavfile.read(path)
    assert rate == A.AUDIO_RATE
    assert data.dtype == np.int16 and data.shape == (len(x), 2)
    assert np.max(np.abs(data)) == pytest.approx(0.5 * 32767, abs=2)
