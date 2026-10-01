"""Strain conditioning on synthetic coloured noise (real gwpy code paths, no network)."""

import numpy as np
import pytest

from gwsonify import GwsonifyError
from gwsonify import strain as S

T0 = 1126259462.4


def settings(**kw):
    kw.setdefault("window", (-4.0, 2.0))
    kw.setdefault("band", (20.0, 500.0))
    return S.StrainSettings(**kw)


def _band_power(x, dt, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), dt)
    return np.mean(spec[(f > lo) & (f < hi)])


def test_whitening_flattens_the_spectrum(synthetic_strain):
    ts = synthetic_strain()
    out = S.condition(ts, T0, settings(mode="whiten", band=(20.0, 1300.0))).data
    assert out.duration.value == pytest.approx(6.0)
    raw = ts.crop(T0 - 4, T0 + 2).value
    dt = out.dt.value
    # compare two bands well inside the passband (away from the filter roll-off)
    raw_tilt = _band_power(raw, dt, 100, 200) / _band_power(raw, dt, 600, 900)
    white_tilt = _band_power(out.value, dt, 100, 200) / _band_power(out.value, dt, 600, 900)
    assert raw_tilt > 5  # strongly red input
    # flat after whitening, up to the passband ripple of gwpy's default IIR bandpass
    # (gpass = 2 dB, applied forwards and backwards)
    assert 0.5 < white_tilt < 2.0
    # in-band whitened noise has the variance expected for unit-variance white noise
    assert np.std(out.value) == pytest.approx(np.sqrt(1280 / 2048), rel=0.25)


def test_whitening_reveals_injection_at_the_right_time(synthetic_strain):
    ts = synthetic_strain(inject=5.0)  # 150 Hz sine-Gaussian burst at the merger time
    out = S.condition(ts, T0, settings()).data
    peak_t = out.times.value[np.argmax(np.abs(out.value))] - T0
    assert abs(peak_t) < 0.05
    assert np.max(np.abs(out.value)) > 5 * np.std(out.crop(T0 - 4, T0 - 1).value)


def test_bandpass_mode_notches_mains_line(synthetic_strain):
    ts = synthetic_strain(line=60.0)
    out = S.condition(ts, T0, settings(mode="bandpass", band=(30.0, 400.0))).data
    x = out.value * np.hanning(len(out))
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), out.dt.value)
    line = spec[np.argmin(np.abs(f - 60.0))]
    neighbours = np.median(spec[(f > 70) & (f < 110)])
    assert line < 5 * neighbours


def test_gating_removes_loud_glitch(synthetic_strain):
    ts = synthetic_strain(glitch=-2.5)
    plain = S.condition(ts, T0, settings(gate=False))
    gated = S.condition(ts, T0, settings(gate=True))
    assert gated.gates, "a gate should be reported"
    a, b = gated.gates[0]
    assert a - 0.1 <= T0 - 2.5 <= b + 0.1
    assert np.max(np.abs(gated.data.value)) < 0.2 * np.max(np.abs(plain.data.value))


def test_raw_mode_is_untouched_except_mean(synthetic_strain):
    ts = synthetic_strain()
    out = S.condition(ts, T0, settings(mode="raw")).data
    ref = ts.crop(T0 - 4, T0 + 2).value
    assert np.allclose(out.value, ref - np.mean(ts.value), rtol=0, atol=1e-30)


def test_whiten_like_places_template_on_data_grid(synthetic_strain):
    from gwpy.timeseries import TimeSeries

    ts = synthetic_strain()
    cond = S.condition(ts, T0, settings())
    t = np.arange(-1, 0.05, 1 / 4096)
    tmpl = TimeSeries(1e-21 * np.sin(2 * np.pi * 100 * t) * np.hanning(len(t)),
                      t0=T0 - 1, sample_rate=4096)
    w = S.whiten_like(tmpl, cond.asd, (20.0, 500.0), ts)
    assert w.t0.value == ts.t0.value and len(w) == len(ts)
    peak_t = w.times.value[np.argmax(np.abs(w.value))] - T0
    assert -1 < peak_t < 0.05


def test_settings_validation():
    with pytest.raises(GwsonifyError, match="Nyquist"):
        S.StrainSettings(band=(20.0, 3000.0), sample_rate=4096)
    with pytest.raises(GwsonifyError):
        S.StrainSettings(mode="loud")
    with pytest.raises(GwsonifyError):
        S.StrainSettings(window=(2.0, -1.0))
    with pytest.raises(GwsonifyError):
        S.StrainSettings(sample_rate=8192)


def test_span_pads_and_has_minimum_length():
    start, end = S._span(T0, (-3.0, 1.0))
    assert end - start == pytest.approx(S.MIN_SPAN)
    assert start < T0 - 3 - 1 and end > T0 + 1 + 1
    start, end = S._span(T0, (-100.0, 1.0))
    assert start == pytest.approx(T0 - 100 - S.PAD)
