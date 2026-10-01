"""Figures: time-frequency maps, traces, posterior summaries.

The main figure (:func:`signal_figure`) is also the video background (see
:mod:`gwsonify.video`). Each detector gets a Q-transform, the standard LIGO/Virgo
time-frequency view of transients (Chatterji et al. 2004, CQG 21, S1809; computed by
:meth:`gwpy.timeseries.TimeSeries.q_transform`), above the time series that is being
played. Times are seconds relative to the merger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

COLORS = {"H1": "#e74c3c", "L1": "#3498db", "V1": "#9b59b6", "K1": "#27ae60", "G1": "#7f8c8d"}
NAMES = {"H1": "LIGO Hanford", "L1": "LIGO Livingston", "V1": "Virgo", "K1": "KAGRA",
         "G1": "GEO600"}


def _mpl():
    import matplotlib

    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        try:
            matplotlib.use("Agg", force=False)
        except Exception:  # pragma: no cover
            pass
    import matplotlib.pyplot as plt

    # pesummary and gwpy may switch on LaTeX text rendering at import; gwsonify figures
    # use plain matplotlib text so they work without a TeX installation.
    plt.rcParams["text.usetex"] = False
    return plt


@dataclass
class Panel:
    """What to draw for one detector."""

    detector: str
    times: np.ndarray            # seconds relative to merger
    values: np.ndarray           # the series being sonified (native rate)
    unit: str
    qspec: object | None = None  # gwpy Spectrogram (Q-transform) or None
    overlay: np.ndarray | None = None      # e.g. whitened template on same grid
    overlay_label: str = ""
    extras: dict = field(default_factory=dict)


def qtransform(ts, t0: float, window: tuple[float, float], frange: tuple[float, float],
               whiten: bool):
    """Q-transform of ``ts`` restricted to ``t0 + window`` (None if it fails).

    Data are whitened and median-normalised (the usual "normalised energy"); noiseless
    model waveforms are neither, and are shown on a logarithmic colour scale.
    """
    try:
        return ts.q_transform(outseg=(t0 + window[0], t0 + window[1]), frange=frange,
                              qrange=(4, 64), whiten=whiten, norm="median" if whiten else False,
                              tres=(window[1] - window[0]) / 1500, logf=True, fres=300)
    except Exception as exc:  # pragma: no cover - depends on data
        log.info("Q-transform failed for %s: %s", ts.name, exc)
        return None


def signal_figure(panels: list[Panel], title: str, subtitle: str, t0: float,
                  frange: tuple[float, float], figsize=(16, 9), dpi=120):
    """Draw the main figure. Returns ``(fig, axes)``, where ``axes`` lists every axes that
    shares the time axis (for the video playhead)."""
    plt = _mpl()
    n = len(panels)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    heights = []
    for _ in panels:
        heights += [2.2, 1.0]
    gs = fig.add_gridspec(2 * n, 1, height_ratios=heights, hspace=0.12,
                          left=0.07, right=0.93, top=0.86, bottom=0.07)
    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.07, ha="left", y=0.965)
    fig.text(0.07, 0.905, subtitle, fontsize=11, ha="left", va="center", color="0.25")
    axes = []
    share = None
    for i, p in enumerate(panels):
        color = COLORS.get(p.detector, "k")
        axq = fig.add_subplot(gs[2 * i], sharex=share)
        share = share or axq
        if p.qspec is not None:
            q = p.qspec
            tq = q.times.value - t0
            fq = q.frequencies.value
            if p.extras.get("qnorm", True):
                kw = {"vmin": 0, "vmax": max(25.0, float(np.percentile(q.value, 99.9)))}
            else:
                from matplotlib.colors import LogNorm

                top = float(np.max(q.value)) or 1.0
                kw = {"norm": LogNorm(vmin=top * 1e-4, vmax=top)}
            mesh = axq.pcolormesh(tq, fq, np.maximum(q.value.T, 1e-300), shading="auto",
                                  cmap="viridis", rasterized=True, **kw)
            axq.set_yscale("log")
            axq.set_ylim(*frange)
            cax = axq.inset_axes([1.01, 0.0, 0.012, 1.0])
            fig.colorbar(mesh, cax=cax).set_label(
                "normalised\nenergy" if p.extras.get("qnorm", True) else "energy\n(arb.)",
                fontsize=8)
        else:
            axq.text(0.5, 0.5, "time-frequency map unavailable", ha="center", va="center",
                     transform=axq.transAxes, color="0.5")
        axq.set_ylabel("Frequency [Hz]")
        axq.text(0.005, 0.95, f"{p.detector}  {NAMES.get(p.detector, '')}", transform=axq.transAxes,
                 ha="left", va="top", color="white", fontsize=12, fontweight="bold",
                 bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none", "pad": 3})
        axq.tick_params(labelbottom=False)
        axt = fig.add_subplot(gs[2 * i + 1], sharex=share)
        axt.plot(p.times, p.values, color=color, lw=0.8)
        if p.overlay is not None:
            axt.plot(p.times, p.overlay, color="k", lw=1.0, alpha=0.8, label=p.overlay_label)
            axt.legend(loc="upper left", fontsize=8, frameon=False)
        axt.set_ylabel(p.unit, fontsize=8)
        axt.grid(alpha=0.3)
        if i < n - 1:
            axt.tick_params(labelbottom=False)
        axes += [axq, axt]
    axes[-1].set_xlabel("Time relative to merger [s]")
    axes[-1].set_xlim(panels[0].times[0], panels[0].times[-1])
    fig.text(0.93, 0.012, "Data: GWOSC (gwosc.org) · made with gwsonify", ha="right",
             fontsize=8, color="0.4")
    return fig, axes


def title_card(title: str, lines: list[str], figsize=(16, 9), dpi=120):
    """A plain title card with event metadata (first frames of the video)."""
    plt = _mpl()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#0b1020")
    fig.text(0.5, 0.62, title, ha="center", va="center", fontsize=54, color="white",
             fontweight="bold")
    for i, line in enumerate(lines):
        fig.text(0.5, 0.48 - 0.06 * i, line, ha="center", va="center", fontsize=20,
                 color="#c8d0e0")
    fig.text(0.5, 0.06, "Data: Gravitational Wave Open Science Center (gwosc.org) · gwsonify",
             ha="center", fontsize=12, color="#7f8aa3")
    return fig


def posterior_figure(samples, chosen: dict, title: str, dpi=120):
    """Histograms of key source parameters with the sonified sample marked."""
    plt = _mpl()
    wanted = [
        ("mass_1_source", r"$m_1$ [M$_\odot$]"), ("mass_2_source", r"$m_2$ [M$_\odot$]"),
        ("chirp_mass_source", r"$\mathcal{M}$ [M$_\odot$]"), ("chi_eff", r"$\chi_{\rm eff}$"),
        ("luminosity_distance", r"$d_L$ [Mpc]"), ("mass_ratio", r"$q$"),
    ]
    have = [(k, lab) for k, lab in wanted if k in samples.parameters]
    if not have:
        return None
    fig, axs = plt.subplots(1, len(have), figsize=(3.0 * len(have), 3.0), dpi=dpi)
    axs = np.atleast_1d(axs)
    for ax, (key, lab) in zip(axs, have, strict=True):
        x = np.asarray(samples[key], dtype=float)
        ax.hist(x, bins=50, density=True, histtype="stepfilled", color="#5b8def", alpha=0.6)
        lo, med, hi = np.percentile(x, [5, 50, 95])
        for v in (lo, hi):
            ax.axvline(v, color="0.4", ls=":", lw=1)
        if key in chosen:
            ax.axvline(chosen[key], color="#e74c3c", lw=1.5)
        ax.set_xlabel(lab)
        ax.set_yticks([])
        ax.set_title(f"{med:.3g}$^{{+{hi - med:.2g}}}_{{-{med - lo:.2g}}}$", fontsize=10)
    fig.suptitle(title + "  (red: sonified sample; dotted: 90% interval)", fontsize=11)
    fig.tight_layout()
    return fig
