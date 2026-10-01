"""High-level API: from an event name to audio, figures, video and provenance.

:func:`sonify_data` sonifies detector strain and :func:`sonify_model` sonifies the
waveform model of a posterior sample. Both write into ``outdir/<EVENT>/`` and return a
:class:`Result` that also holds the audio arrays (handy in notebooks: ``result.play()``).
The CLI is a thin wrapper around these two functions.

Output names are predictable: ``<EVENT>_<DET>_<kind>[_<tag>].wav`` (one per detector)
or ``<EVENT>_<kind>[_<tag>]_stereo.wav``, where ``kind`` is ``data-whiten``,
``data-bandpass``, ``data-raw`` or ``model``. ``tag`` lists non-default audio
settings (for example ``speed0.5_fshift400``), so runs with different settings do not
overwrite each other. Every run writes ``<EVENT>_<kind>[_<tag>].provenance.json``.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gwsonify import GwsonifyError, GwsonifyWarning, MissingDependencyError
from gwsonify import audio as A
from gwsonify import provenance as P
from gwsonify import strain as S
from gwsonify.catalog import Event, get_event

log = logging.getLogger(__name__)

FIXED_REFERENCE = {"whiten": 1.0, "bandpass": 1e-21, "raw": 1e-18, "model": 1e-21}
"""Amplitude mapped to ``--level`` dBFS in ``--norm fixed`` mode, per signal kind:
1 noise standard deviation for whitened data, strain 1e-21 for bandpassed data and
models, 1e-18 for raw strain."""


@dataclass
class Result:
    """What a sonification run produced."""

    kind: str
    name: str
    outdir: Path
    files: list[Path] = field(default_factory=list)
    audio: dict[str, np.ndarray] = field(default_factory=dict)
    rate: int = A.AUDIO_RATE
    series: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    event: Event | None = None
    gain: float = 1.0  # audio samples = signal x gain (before 16-bit quantisation)

    def play(self, key: str | None = None):
        """Return an ``IPython.display.Audio`` player for one output (default: the first)."""
        from IPython.display import Audio

        key = key or next(iter(self.audio))
        data = self.audio[key]
        return Audio(data.T if data.ndim == 2 else data, rate=self.rate, normalize=False)

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "outdir": str(self.outdir),
            "files": [str(f) for f in self.files],
            "audio_seconds": {k: len(v) / self.rate for k, v in self.audio.items()},
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


def _outdir(outdir: str | Path | None, name: str) -> Path:
    base = Path(outdir) if outdir else Path(os.environ.get("GWSONIFY_OUTDIR", "gwsonify-output"))
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tag(s: A.AudioSettings) -> str:
    d = A.AudioSettings()
    parts = []
    for key in ("speed", "stretch", "pitch", "fshift"):
        v = getattr(s, key)
        if v != getattr(d, key):
            parts.append(f"{key}{v:g}")
    if s.norm != "peak":
        parts.append(f"{s.norm}{s.level:g}dB")
    return "_".join(parts)


def _stem(name: str, kind: str, tag: str) -> str:
    return f"{name}_{kind}" + (f"_{tag}" if tag else "")


def _pan(channels: np.ndarray) -> np.ndarray:
    """Equal-power pan of k channels across the stereo field (k=2: hard left/right)."""
    k = channels.shape[1]
    if k == 1:
        return np.repeat(channels, 2, axis=1)
    theta = (np.linspace(-1, 1, k) + 1) * np.pi / 4
    mix = np.stack([channels @ np.cos(theta), channels @ np.sin(theta)], axis=1)
    return mix / max(np.max(np.abs(mix)) / max(np.max(np.abs(channels)), 1e-30), 1.0)


def _audio_settings(speed, stretch, pitch, fshift, fade, norm, level, kind) -> A.AudioSettings:
    return A.AudioSettings(speed=speed, stretch=stretch, pitch=pitch, fshift=fshift, fade=fade,
                           norm=norm, level=level, reference=FIXED_REFERENCE[kind])


def _write_outputs(result: Result, names: list[str], channels: list[np.ndarray], fs: float,
                   settings: A.AudioSettings, stereo: bool, stem: str, name: str,
                   kind: str, tag: str) -> np.ndarray:
    """Render channels jointly and write WAVs. Returns the stereo/mono mix for video."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GwsonifyWarning)
        rendered, gain = A.render(channels, fs, settings)
    for w in caught:
        result.notes.append(str(w.message))
        log.warning("%s", w.message)
    block = rendered if rendered.ndim == 2 else rendered[:, None]
    result.gain = gain
    data_cols = [i for i, n in enumerate(names) if "template" not in n]
    mix = _pan(block[:, data_cols]) if len(data_cols) > 1 else block[:, data_cols[0]]
    if stereo:
        path = result.outdir / f"{stem}_stereo.wav"
        A.write_wav(path, mix if mix.ndim == 2 else _pan(mix[:, None]))
        result.files.append(path)
        result.audio["stereo"] = mix if mix.ndim == 2 else _pan(mix[:, None])
    else:
        for i, det in enumerate(names):
            if "template" in det:
                continue
            path = result.outdir / f"{name}_{det}_{kind}" f"{('_' + tag) if tag else ''}.wav"
            A.write_wav(path, block[:, i])
            result.files.append(path)
            result.audio[det] = block[:, i]
    # template overlays: data left, template right
    for i, n in enumerate(names):
        if n.endswith(" template"):
            det = n.split()[0]
            j = names.index(det)
            base = f"{name}_{det}_{kind}" + (f"_{tag}" if tag else "")
            tpath = result.outdir / f"{base.replace(kind, 'template-whiten')}.wav"
            A.write_wav(tpath, block[:, i])
            opath = result.outdir / f"{base}_overlay.wav"
            A.write_wav(opath, np.stack([block[:, j], block[:, i]], axis=1))
            result.files += [tpath, opath]
            result.audio[f"{det} template"] = block[:, i]
            result.audio[f"{det} overlay"] = np.stack([block[:, j], block[:, i]], axis=1)
    return mix


def _video(result: Result, fig, axes, mix, window, settings, title, lines, stem) -> None:
    from gwsonify import plots, video

    try:
        video.find_ffmpeg()
    except MissingDependencyError as exc:
        result.notes.append(f"Video skipped: {exc}")
        log.warning("Video skipped: %s", exc)
        return
    log.info("Rendering video ...")
    card = plots.title_card(title, lines)
    path = video.render(result.outdir / f"{stem}.mp4", fig, axes,
                        mix, window, settings.duration_factor, card)
    result.files.append(path)


def _event_lines(event: Event | None) -> list[str]:
    if event is None:
        return []
    p = event.parameters
    lines = [f"{event.utc} · {event.catalog}"]
    bits = []
    if "mass_1_source" in p and "mass_2_source" in p:
        bits.append(f"{p['mass_1_source']:.3g} + {p['mass_2_source']:.3g} solar masses")
    if "luminosity_distance" in p:
        bits.append(f"{p['luminosity_distance']:.0f} Mpc away")
    if bits:
        lines.append(" · ".join(bits))
    return lines


def _processing_line(settings: A.AudioSettings) -> str:
    ops = []
    if settings.speed != 1:
        ops.append(f"speed ×{settings.speed:g}")
    if settings.stretch != 1:
        ops.append(f"stretched ×{settings.stretch:g} (pitch kept)")
    if settings.pitch:
        ops.append(f"pitch {settings.pitch:+g} semitones")
    if settings.fshift:
        ops.append(f"frequencies shifted {settings.fshift:+g} Hz")
    return "audio: " + (", ".join(ops) if ops else "real time, real frequencies")


def _resolve(event) -> Event:
    return event if isinstance(event, Event) else get_event(str(event))


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------


def sonify_data(event: str | Event | None = None, *, gps: float | None = None,
                detectors: list[str] | None = None, mode: str = "whiten",
                window: tuple[float, float] | None = None,
                band: tuple[float, float] | None = None, sample_rate: int = 4096,
                gate: bool | None = None, template: bool = False,
                speed: float = 1.0, stretch: float = 1.0, pitch: float = 0.0,
                fshift: float = 0.0, fade: float = 0.05, norm: str = "peak",
                level: float | None = None, stereo: bool = False, video: bool = False,
                plots: bool = True, outdir: str | Path | None = None,
                quiet: bool = False, yes: bool = False) -> Result:
    """Sonify GWOSC detector strain around an event (or any GPS time).

    Parameters mirror ``gwsonify data --help``. ``gate`` defaults to True in whiten mode
    and False otherwise. Returns a :class:`Result`.
    """
    if (event is None) == (gps is None):
        raise GwsonifyError("Give an event name or --gps (not both).")
    ev = _resolve(event) if event is not None else None
    t0 = ev.gps if ev is not None else float(gps)
    name = ev.name if ev is not None else f"GPS{t0:.1f}".replace(".0", "")
    if ev is None and not detectors:
        raise GwsonifyError("With --gps, choose detectors with -d (for example -d H1,L1).")
    sample_rate = int(sample_rate)
    if gate is None:
        gate = mode == "whiten"
    if template and mode != "whiten":
        raise GwsonifyError("--template needs --mode whiten (templates are whitened like the data).")
    if template and ev is None:
        raise GwsonifyError("--template needs an event (a PE release), not --gps.")

    available = ev.detectors(sample_rate) if ev is not None else None
    if ev is not None and not available:
        raise GwsonifyError(f"GWOSC publishes no strain files for {ev.name}.")
    if detectors:
        detectors = [d.strip().upper() for d in detectors]
        if available is not None:
            missing = [d for d in detectors if d not in available]
            if missing:
                raise GwsonifyError(
                    f"No {', '.join(missing)} strain for {ev.name}; available: {', '.join(available)}."
                )
    else:
        detectors = available

    settings = S.StrainSettings(
        mode=mode,
        window=tuple(window) if window else S.default_window(ev),
        band=tuple(band) if band else S.default_band(ev, sample_rate),
        sample_rate=sample_rate, gate=gate,
    )
    kind = f"data-{mode}"
    audio_settings = _audio_settings(speed, stretch, pitch, fshift, fade, norm, level, mode)
    tag = _tag(audio_settings)
    stem = _stem(name, kind, tag)
    result = Result(kind=kind, name=name, outdir=_outdir(outdir, name), event=ev)
    log.info("%s: %s from %s, window %+g to %+g s, band %g-%g Hz, %d Hz", name, mode,
             ", ".join(detectors), *settings.window, *settings.band, sample_rate)

    start, end = S._span(t0, settings.window)
    raw, cond, data_files = {}, {}, []
    for det in detectors:
        try:
            ts, files = S.load(det, start, end, sample_rate=sample_rate, event=ev, quiet=quiet)
        except GwsonifyError as exc:
            if len(detectors) > 1 and "gaps" in str(exc):
                result.notes.append(str(exc))
                log.warning("Skipping %s: %s", det, exc)
                continue
            raise
        data_files += files
        raw[det] = ts
        cond[det] = S.condition(ts, t0, settings, det)
        for a, b in cond[det].gates:
            result.notes.append(f"{det}: gated a loud glitch at GPS {a:.2f}-{b:.2f}")
    if not cond:
        raise GwsonifyError("No usable detector data.")
    dets = list(cond)

    names = list(dets)
    channels = [np.asarray(cond[d].data.value) for d in dets]
    waveform = None
    overlays = {}
    if template:
        from gwsonify import model

        waveform = model.generate(ev, dets, sample_rate=sample_rate, quiet=quiet, yes=yes)
        for det in dets:
            wt = S.whiten_like(waveform.strains[det], cond[det].asd, settings.band, raw[det])
            wt = wt.crop(t0 + settings.window[0], t0 + settings.window[1])
            vals = np.asarray(wt.value)[: len(channels[dets.index(det)])]
            overlays[det] = vals
            names.append(f"{det} template")
            channels.append(vals)
    n = min(len(c) for c in channels)
    channels = [c[:n] for c in channels]

    mix = _write_outputs(result, names, channels, sample_rate, audio_settings, stereo, stem,
                         name, kind, tag)
    result.series = {d: cond[d].data for d in dets}

    title = f"{name} · {'whitened' if mode == 'whiten' else mode} detector data"
    subtitle = (f"{ev.catalog} · GPS {t0:.1f} · " if ev else f"GPS {t0:.1f} · ") + \
        f"band {settings.band[0]:g}-{settings.band[1]:g} Hz · " + _processing_line(audio_settings)
    fig = None
    if plots or video:
        from gwsonify import plots as PL

        panels = []
        for det in dets:
            d = cond[det].data
            q = PL.qtransform(raw[det], t0, settings.window, settings.band, whiten=True)
            panels.append(PL.Panel(det, d.times.value[:n] - t0, np.asarray(d.value)[:n],
                                   "whitened\n[σ]" if mode == "whiten" else "strain", q,
                                   overlays.get(det), "whitened model" if template else ""))
        fig, axes = PL.signal_figure(panels, title, subtitle, t0, settings.band)
        if plots:
            png = result.outdir / f"{stem}.png"
            fig.savefig(png)
            result.files.append(png)
        if video:
            _video(result, fig, axes, mix, settings.window, audio_settings,
                   name, [*_event_lines(ev), _processing_line(audio_settings)], stem)
        import matplotlib.pyplot as plt

        plt.close(fig)

    record = P.build(
        kind, event=ev,
        inputs={"event": event if isinstance(event, str) else (ev.spec if ev else None),
                "gps": t0, "detectors": dets},
        processing={"strain": settings.to_dict(), "audio": audio_settings.to_dict(),
                    "audio_rate": A.AUDIO_RATE, "gain": result.gain,
                    "whitening": {"fftlength": S.FFTLENGTH, "overlap": S.FFTLENGTH / 2,
                                  "method": "median", "window": "hann"} if mode == "whiten" else None,
                    "gates": {d: cond[d].gates for d in dets},
                    "units": {d: cond[d].unit for d in dets}},
        outputs=result.files, data_files=data_files + (
            [(waveform.posterior.url, waveform.posterior.path)] if waveform else []),
        model=waveform.provenance() if waveform else None, notes=result.notes,
    )
    prov = result.outdir / f"{stem}.provenance.json"
    P.write(prov, record)
    result.files.append(prov)
    result.provenance = record
    return result


# ---------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------


def sonify_model(event: str | Event, *, detectors: list[str] | None = None,
                 label: str | None = None, approximant: str | None = None,
                 sample: str | int = "maxl", f_low: float = 20.0,
                 sample_rate: float | None = None,
                 speed: float = 1.0, stretch: float = 1.0, pitch: float = 0.0,
                 fshift: float = 0.0, fade: float = 0.05, norm: str = "peak",
                 level: float | None = None, stereo: bool = False, video: bool = False,
                 plots: bool = True, outdir: str | Path | None = None,
                 quiet: bool = False, yes: bool = False) -> Result:
    """Sonify the detector-projected waveform model of one posterior sample.

    Parameters mirror ``gwsonify model --help``. Channels share one absolute time grid,
    so the millisecond arrival-time differences between detectors are preserved.
    """
    from gwsonify import model

    ev = _resolve(event)
    if isinstance(event, str) and not re.search(r"-v\d+$|@", event.strip()):
        ev = model.version_with_pe(ev)  # only when the user did not pin a version
    if detectors:
        detectors = [d.strip().upper() for d in detectors]
    else:
        detectors = list(ev.listed_detectors) or ["H1", "L1"]
    waveform = model.generate(ev, detectors, label=label, approximant=approximant,
                              sample=sample, f_low=f_low, sample_rate=sample_rate,
                              quiet=quiet, yes=yes)
    fs = waveform.sample_rate
    kind = "model"
    audio_settings = _audio_settings(speed, stretch, pitch, fshift, fade, norm, level, kind)
    tag = _tag(audio_settings)
    stem = _stem(ev.name, kind, tag)
    result = Result(kind=kind, name=ev.name, outdir=_outdir(outdir, ev.name), event=ev)

    # common absolute time grid (keeps inter-detector delays)
    start = min(h.t0.value for h in waveform.strains.values())
    end = max(h.t0.value + h.duration.value for h in waveform.strains.values()) + 0.25
    n = int(round((end - start) * fs))
    channels = []
    for det in detectors:
        h = waveform.strains[det]
        grid = np.zeros(n)
        off = int(round((h.t0.value - start) * fs))
        vals = np.asarray(h.value)[: n - off]
        grid[off : off + len(vals)] = vals
        channels.append(grid)
    mix = _write_outputs(result, list(detectors), channels, fs, audio_settings, stereo, stem,
                         ev.name, kind, tag)
    times = start + np.arange(n) / fs - ev.gps
    window = (float(times[0]), float(times[-1]))
    result.series = dict(waveform.strains)

    title = f"{ev.name} · waveform model ({waveform.approximant})"
    subtitle = (f"{waveform.label} · {waveform.sample_desc} · f_low {f_low:g} Hz · "
                + _processing_line(audio_settings))
    if plots or video:
        from gwpy.timeseries import TimeSeries

        from gwsonify import plots as PL

        panels = []
        hi = min(0.45 * fs, S.default_band(ev, int(fs))[1] * 1.5)
        for det, ch in zip(detectors, channels, strict=True):
            ts = TimeSeries(ch, t0=start, sample_rate=fs, name=det)
            q = PL.qtransform(ts, ev.gps, window, (max(f_low * 0.8, 10.0), hi), whiten=False)
            panels.append(PL.Panel(det, times, ch, "strain", q, extras={"qnorm": False}))
        fig, axes = PL.signal_figure(panels, title, subtitle, ev.gps, (max(f_low * 0.8, 10.0), hi))
        if plots:
            png = result.outdir / f"{stem}.png"
            fig.savefig(png)
            result.files.append(png)
            post_fig = PL.posterior_figure(waveform.posterior.samples(waveform.label),
                                           waveform.sample, f"{ev.name} · {waveform.label}")
            if post_fig is not None:
                ppath = result.outdir / f"{ev.name}_posterior.png"
                post_fig.savefig(ppath)
                result.files.append(ppath)
                import matplotlib.pyplot as plt

                plt.close(post_fig)
        if video:
            _video(result, fig, axes, mix, window, audio_settings, ev.name,
                   [*_event_lines(ev), f"waveform model: {waveform.approximant}",
                    _processing_line(audio_settings)], stem)
        import matplotlib.pyplot as plt

        plt.close(fig)

    record = P.build(
        kind, event=ev, inputs={"event": event if isinstance(event, str) else ev.spec,
                                "detectors": detectors},
        processing={"audio": audio_settings.to_dict(), "audio_rate": A.AUDIO_RATE,
                    "gain": result.gain},
        outputs=result.files, data_files=[(waveform.posterior.url, waveform.posterior.path)],
        model=waveform.provenance(), notes=result.notes,
    )
    prov = result.outdir / f"{stem}.provenance.json"
    P.write(prov, record)
    result.files.append(prov)
    result.provenance = record
    return result
