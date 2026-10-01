"""Detector strain: locate, load and condition GWOSC data for listening.

All signal conditioning is delegated to gwpy, using the standard LIGO-Virgo-KAGRA
procedures (see Abbott et al. 2020, "A guide to LIGO-Virgo detector noise and
extraction of transient gravitational-wave signals", CQG 37, 055002):

``whiten`` (default)
    Divide the data by an estimate of the noise amplitude spectral density (ASD).
    The ASD is a median-averaged Welch estimate (4 s Hann segments, 50% overlap)
    over at least 28 s around the requested window, computed by
    :meth:`gwpy.timeseries.TimeSeries.whiten`. Then bandpass to ``band``. Whitening
    removes the huge low-frequency seismic noise and the narrow spectral lines, so
    what is left is roughly white noise in units of the noise standard deviation.
    A loud signal stands out from it as an audible chirp. This is the recipe
    behind the GWOSC audio files and Fig. 1 of the GW150914 discovery paper.
``bandpass``
    Bandpass to ``band`` and notch the mains-power harmonics (60 Hz for LIGO and
    KAGRA, 50 Hz for Virgo and GEO) inside the band. Without whitening the
    noise stays coloured: this is what the instrument "sounds like".
``raw``
    The calibrated strain with only the mean removed. It is dominated by seismic noise
    below 20 Hz, so it is mostly inaudible rumble. It is included for teaching.

Optional ``gate=True`` applies :meth:`gwpy.timeseries.TimeSeries.gate` (inverse-Tukey
zeroing of samples whose whitened amplitude exceeds 50 sigma), the method used to
remove the glitch in LIGO-Livingston data near GW170817.

Filtering is done on a segment padded on both sides and then cropped, so filter edge
effects never reach the audio.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from gwsonify import GwsonifyError
from gwsonify.cache import fetch
from gwsonify.catalog import Event, StrainFile

log = logging.getLogger(__name__)

MODES = ("whiten", "bandpass", "raw")
SAMPLE_RATES = (4096, 16384)
MAINS = {"H1": 60.0, "L1": 60.0, "K1": 60.0, "V1": 50.0, "G1": 50.0}

PAD = 6.0          # seconds of extra data on each side for filter edges
MIN_SPAN = 28.0    # minimum data span used for the ASD estimate
FFTLENGTH = 4.0


@dataclass
class StrainSettings:
    """How to condition strain. ``window`` is (start, end) in seconds relative to
    the reference time (merger). ``band`` is (low, high) in Hz."""

    mode: str = "whiten"
    window: tuple[float, float] = (-3.0, 1.0)
    band: tuple[float, float] = (20.0, 500.0)
    sample_rate: int = 4096
    gate: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise GwsonifyError(f"--mode must be one of {', '.join(MODES)}")
        if int(self.sample_rate) not in SAMPLE_RATES:
            raise GwsonifyError("--sample-rate must be 4096 or 16384 for GWOSC strain.")
        lo, hi = self.band
        if not 0 < lo < hi:
            raise GwsonifyError(f"Invalid band {self.band}: need 0 < LOW < HIGH.")
        # the bandpass filter's stopband sits at 1.5 x the upper edge (gwpy default)
        if hi * 1.5 >= 0.5 * self.sample_rate:
            raise GwsonifyError(
                f"Band upper edge {hi:g} Hz is too close to Nyquist ({self.sample_rate / 2:.0f} Hz);"
                f" use at most {self.sample_rate / 3:.0f} Hz or --sample-rate 16384."
            )
        start, end = self.window
        if not start < end:
            raise GwsonifyError(f"Invalid window {self.window}: START must be before END.")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------
# Defaults that depend on the source
# ---------------------------------------------------------------------------------------


def default_window(event: Event | None, f_low: float = 20.0) -> tuple[float, float]:
    """Listening window around merger.

    Starts ``tau + 1`` s before merger, where ``tau`` is the leading-order time the
    signal takes to sweep from ``f_low`` to merger (:meth:`Event.time_to_merger`).
    The lead is clipped to 3-120 s, and the window ends 1 s after merger. That gives
    [-3, 1] for GW150914 and [-120, 1] for GW170817. Without source information the
    window is [-8, 2].
    """
    tau = event.time_to_merger(f_low) if event is not None else None
    if tau is None:
        return (-8.0, 2.0)
    lead = float(np.clip(tau + 1.0, 3.0, 120.0))
    return (-round(lead, 1), 1.0)


def default_band(event: Event | None, sample_rate: int = 4096) -> tuple[float, float]:
    """Bandpass edges: 20 Hz up to clip(3 f_ISCO, 300, 1000) Hz.

    f_ISCO = c^3 / (6^1.5 pi G M) is the GW frequency at the innermost stable
    circular orbit of the total (detector-frame) mass. Three times f_ISCO keeps the
    merger and ringdown of black-hole binaries: 300 Hz for GW150914, close to the
    35-350 Hz band of the discovery paper. For neutron-star binaries the upper edge
    rises to 1000 Hz. Without source information the band is 20-500 Hz. The upper
    edge never exceeds one third of the sample rate (room for the filter's stopband).
    """
    f = event.f_isco() if event is not None else None
    high = 500.0 if f is None else float(np.clip(3 * f, 300.0, 1000.0))
    return (20.0, min(round(high), int(sample_rate / 3.1)))


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------


def _span(t0: float, window: tuple[float, float]) -> tuple[float, float]:
    start, end = t0 + window[0] - PAD, t0 + window[1] + PAD
    if end - start < MIN_SPAN:
        mid, half = (start + end) / 2, MIN_SPAN / 2
        start, end = mid - half, mid + half
    return start, end


def pick_file(files: list[StrainFile], detector: str, start: float, end: float
              ) -> StrainFile | None:
    """The file for ``detector`` that fully covers [start, end], preferring the chosen
    event version's release (then newer, then older releases, then bulk run files) and,
    within a release, the smallest file."""
    cover = [f for f in files if f.detector == detector
             and f.gps_start <= start and f.gps_end >= end]
    return min(cover, key=lambda f: (f.rank, f.duration)) if cover else None


def load(detector: str, start: float, end: float, *, sample_rate: int = 4096,
         event: Event | None = None, quiet: bool = False):
    """Return ``(TimeSeries, [(url, local_path), ...])`` of GWOSC strain for [start, end].

    With ``event``, the file is chosen from the event's GWOSC strain listing (the
    smallest file that covers the span). Otherwise it is located with
    :func:`gwosc.locate.get_urls`. Files are cached (see :mod:`gwsonify.cache`).
    """
    from gwpy.timeseries import TimeSeries

    urls: list[str] = []
    if event is not None:
        files = event.strain_files(sample_rate)
        chosen = pick_file(files, detector, start, end)
        if chosen is None:
            have = sorted({f.detector for f in files})
            if detector not in have:
                raise GwsonifyError(
                    f"No {detector} strain is published for {event.name}. "
                    f"Available: {', '.join(have) or 'none'}."
                )
            raise GwsonifyError(
                f"The {detector} data for {event.name} do not cover GPS {start:.1f}-{end:.1f}; "
                "use a shorter --window."
            )
        urls = [chosen.url]
    else:
        from gwosc.locate import get_urls

        try:
            urls = get_urls(detector, int(np.floor(start)), int(np.ceil(end)),
                            sample_rate=sample_rate, format="hdf5")
        except ValueError as exc:
            raise GwsonifyError(
                f"GWOSC has no {detector} data at GPS {start:.0f}-{end:.0f} ({exc})."
            ) from None
        if not urls:
            raise GwsonifyError(f"GWOSC has no {detector} data at GPS {start:.0f}-{end:.0f}.")

    paths = [fetch(u, "strain", quiet=quiet) for u in urls]
    files = list(zip(urls, paths, strict=True))
    try:
        data = TimeSeries.read(paths if len(paths) > 1 else paths[0], format="hdf5.gwosc",
                               start=start, end=end)
    except Exception as exc:  # corrupt download, gap between files, ...
        raise GwsonifyError(f"Could not read {detector} strain: {exc}") from exc
    data.name = detector
    bad = ~np.isfinite(data.value)
    if bad.any():
        raise GwsonifyError(
            f"{detector} data contain gaps ({bad.mean():.0%} of samples invalid) between "
            f"GPS {start:.1f} and {end:.1f}; the detector was not observing throughout. "
            "Try another detector or a shorter --window."
        )
    return data, files


# ---------------------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------------------


@dataclass
class Conditioned:
    """Output of :func:`condition`."""

    data: object          # gwpy TimeSeries cropped to the window
    asd: object | None    # gwpy FrequencySeries used for whitening (whiten mode)
    gates: list           # [(start, end)] GPS intervals zeroed by gating
    unit: str             # human description of the amplitude unit


def condition(ts, t0: float, settings: StrainSettings, detector: str | None = None
              ) -> Conditioned:
    """Condition a padded strain segment and crop it to ``t0 + settings.window``."""
    det = detector or str(ts.name)
    lo, hi = settings.band
    gates: list = []
    work = ts.detrend("constant")
    if settings.gate:
        gated = work.gate(whiten=True, threshold=50.0)
        zero = np.flatnonzero((gated.value == 0) & (work.value != 0))
        if zero.size:
            times = gated.times.value[zero]
            breaks = np.flatnonzero(np.diff(times) > 2 * gated.dt.value)
            starts = np.r_[times[0], times[breaks + 1]]
            ends = np.r_[times[breaks], times[-1]]
            gates = [(float(a), float(b)) for a, b in zip(starts, ends, strict=True)]
            for a, b in gates:
                log.info("%s: gated a loud glitch at GPS %.2f-%.2f", det, a, b)
                if a - 0.5 < t0 < b + 0.5:
                    log.warning("%s: a gate overlaps the merger time; the signal may be removed."
                                " Rerun without --gate to compare.", det)
        work = gated

    asd = None
    if settings.mode == "whiten":
        asd = work.asd(fftlength=FFTLENGTH, overlap=FFTLENGTH / 2, method="median",
                       window="hann")
        out = work.whiten(asd=asd, fduration=2).bandpass(lo, hi)
        unit = "whitened strain (units of noise standard deviation)"
    elif settings.mode == "bandpass":
        out = work.bandpass(lo, hi)
        base = MAINS.get(det[:2], 60.0)
        for f in np.arange(base, hi, base):
            if f > lo:
                out = out.notch(f)
        unit = "strain (dimensionless), bandpassed"
    else:
        out = work
        unit = "strain (dimensionless), mean removed"
    start, end = t0 + settings.window[0], t0 + settings.window[1]
    return Conditioned(out.crop(start, end, copy=True), asd, gates, unit)


def whiten_like(template, asd, band: tuple[float, float], like):
    """Whiten and bandpass a model waveform exactly as the data were.

    ``template`` (gwpy TimeSeries, detector-projected, absolute GPS times) is placed
    on the time grid of ``like`` (the padded raw data segment), whitened with the data's
    ``asd`` and bandpassed. This is the "whitened template" that GWOSC tutorials overlay
    on whitened data.
    """
    from gwpy.timeseries import TimeSeries

    grid = np.zeros(len(like))
    dt = like.dt.value
    tmpl = template.resample(like.sample_rate.value) if template.dt.value != dt else template
    offset = int(round((tmpl.t0.value - like.t0.value) / dt))
    src = np.asarray(tmpl.value)
    lo_i, hi_i = max(offset, 0), min(offset + len(src), len(grid))
    if hi_i <= lo_i:
        raise GwsonifyError("The model waveform does not overlap the data segment.")
    grid[lo_i:hi_i] = src[lo_i - offset : hi_i - offset]
    ts = TimeSeries(grid, t0=like.t0, dt=like.dt, name=f"{like.name} template")
    return ts.whiten(asd=asd, fduration=2).bandpass(*band)
