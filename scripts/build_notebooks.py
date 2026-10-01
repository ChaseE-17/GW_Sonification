#!/usr/bin/env python
"""Generate the teaching notebooks in notebooks/ (without outputs).

The notebooks are generated so that their source stays reviewable in one place. Edit
this file, then run ``python scripts/build_notebooks.py``. The notebooks only call the
gwsonify API; no processing logic lives in them.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''\
# Setup: on Google Colab or Binder this installs gwsonify; locally it does nothing
import sys, importlib.util
if importlib.util.find_spec("gwsonify") is None:
    %pip install -q "gwsonify[model] @ git+https://github.com/ChaseE-17/GW_Sonification"

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, Image, display

import gwsonify
print("gwsonify", gwsonify.__version__)'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def instructor():
    return [
        md("""
# Hearing a black-hole merger: instructor quick tour (≈10 minutes)

This notebook is a short, projector-friendly demo. Every step is one call to
[`gwsonify`](https://github.com/ChaseE-17/GW_Sonification). The longer
**self-study notebook** (`02_self_study.ipynb`) explains the physics and the signal
processing in detail.

**What you will hear:** real data from the two LIGO detectors on 14 September 2015,
the first gravitational wave ever detected (GW150914). Two black holes of about 36 and
31 solar masses merged 1.3 billion light years away.

*Tip for the room: use headphones or decent speakers. The signal is between 30 and
300 Hz, below what laptop speakers reproduce well.*
"""),
        code(SETUP),
        md("""
## 1. The data, whitened: the actual observation

`sonify_data` downloads a few seconds of strain from the Gravitational Wave Open
Science Center (GWOSC), **whitens** it (divides out the detector's noise spectrum),
bandpasses it to 20-300 Hz, and writes audio, a figure and a provenance record.
Listen for the short upward "whoop" just before the end.
"""),
        code("""
data = gwsonify.sonify_data("GW150914")
display(Image(filename=str(data.outdir / "GW150914_data-whiten.png"), width=900))
data.play("H1")
"""),
        md("""
The bright curved track in the time-frequency maps is the chirp. It appears in both
detectors, 3,000 km apart, about 7 ms apart in time.

## 2. Make it easier to hear: shift every frequency up by 400 Hz

This is exactly what the GWOSC audio releases do. A frequency shift adds a constant
to every frequency, so the timing stays exact.
"""),
        code("""
shifted = gwsonify.sonify_data("GW150914", fshift=400, plots=False)
shifted.play("H1")
"""),
        md("""
## 3. What should it sound like? Overlay the waveform model

The LIGO-Virgo-KAGRA analysis found the best-fitting general-relativity waveform. With
`template=True`, gwsonify whitens that model exactly like the data. The overlay file
puts the data in the left ear and the model in the right.
"""),
        code("""
overlay = gwsonify.sonify_data("GW150914", template=True, fshift=400)
display(Image(filename=str(overlay.outdir / "GW150914_data-whiten_fshift400.png"), width=900))
overlay.play("H1 overlay")
"""),
        md("""
## 4. A neutron-star merger is much longer

GW170817 (two neutron stars, 40 Mpc away) spent over two minutes in the detectors'
band. Here is the *waveform model* at real speed, starting at 30 Hz. The pitch rises
slowly for a minute and then races up in the final second.
"""),
        code("""
bns = gwsonify.sonify_model("GW170817", detectors=["L1"], f_low=30)
display(Image(filename=str(bns.outdir / "GW170817_model.png"), width=900))
bns.play("L1")
"""),
        md("""
## 5. For a lecture: make a video

`video=True` writes an MP4 with a title card, the time-frequency map and a moving
playhead (it needs ffmpeg; gwsonify tells you if it is missing). From a terminal the same
thing is `gwsonify GW150914 --fshift 400 --template --video`.
"""),
        code("""
movie = gwsonify.sonify_data("GW150914", fshift=400, stereo=True, video=True, plots=False)
print([p.name for p in movie.files])
"""),
        md("""
## Credit and citation

Every output folder contains a `*.provenance.json` with the data sources, software
versions and the GWOSC acknowledgement. Please acknowledge GWOSC when you show these
results: see https://gwosc.org/acknowledgement/.
"""),
    ]


def self_study():
    return [
        md("""
# Listening to gravitational waves: a self-study notebook

Work through this notebook top to bottom (about 45-60 minutes). It explains

1. what **strain** is and what raw detector data sound like,
2. why we **whiten** and **bandpass**, step by step,
3. why the chirp **sweeps upward**, and how its duration depends on the masses,
4. what an **approximant** (waveform model) is and what the **posterior** tells us,
5. the four different ways to change how a signal sounds (**speed, stretch, pitch,
   frequency shift**) and why they are not the same thing.

Cells marked **✎ Try this** are short exercises. Everything uses public data from the
Gravitational Wave Open Science Center (GWOSC) and the `gwsonify` package.
"""),
        code(SETUP),
        md("""
## 1. Strain: what a detector measures

A passing gravitational wave stretches and squeezes space. A LIGO detector measures
the fractional change in the length of its 4 km arms, the **strain**
$h = \\Delta L / L$. For GW150914 the peak strain was about $10^{-21}$: the arms changed
length by less than a thousandth of a proton's width.

`get_event` asks GWOSC what it knows about an event. Nothing about the event is
hard-coded in gwsonify: names, times, detectors and analyses all come from the GWOSC API.
"""),
        code("""
ev = gwsonify.get_event("GW150914")
print(ev.spec, "from catalog", ev.catalog)
print("merger time (GPS):", ev.gps, "=", ev.utc)
print("detectors with public strain:", ev.detectors())
print({k: ev.parameters[k] for k in ("mass_1_source", "mass_2_source", "luminosity_distance")})
"""),
        md("""
Now load 32 s of raw H1 strain around the merger. `gwsonify.strain.load` picks the
smallest GWOSC file that covers the time span and caches it.
"""),
        code("""
from gwsonify import strain

raw, files = strain.load("H1", ev.gps - 16, ev.gps + 12, event=ev)
print(files[0][0])
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(raw.times.value - ev.gps, raw.value, lw=0.5)
ax.set(xlabel="time relative to merger [s]", ylabel="strain", title="Raw H1 strain")
plt.show()
"""),
        md("""
The raw strain is dominated by slow wandering at about $10^{-18}$, a thousand times
larger than the signal. The merger (at $t=0$) is invisible. Let's *listen* to the raw
data: `mode="raw"` keeps it untouched, apart from removing the mean.
"""),
        code("""
raw_audio = gwsonify.sonify_data("GW150914", detectors=["H1"], mode="raw", plots=False)
raw_audio.play("H1")
"""),
        md("""
Mostly a low rumble: seismic and thermal noise below 20 Hz, which is barely audible.

## 2. Why we whiten

The **amplitude spectral density** (ASD) shows how much noise there is at each
frequency. It spans many orders of magnitude, and it has sharp spectral **lines**
(power mains at 60 Hz, mirror suspensions near 500 Hz, calibration lines).
"""),
        code("""
asd = raw.asd(fftlength=4, overlap=2, method="median")
fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(asd.frequencies.value, asd.value)
ax.set(xlim=(10, 2000), ylim=(1e-24, 1e-19), xlabel="frequency [Hz]",
       ylabel=r"ASD [strain/$\\sqrt{\\mathrm{Hz}}$]", title="H1 noise spectrum around GW150914")
ax.axvspan(20, 300, color="orange", alpha=0.2, label="where the chirp lives")
ax.legend(); plt.show()
"""),
        md("""
**Whitening** divides the data, frequency by frequency, by this ASD. Every frequency
then has the same noise level, and the result is in units of the noise standard deviation
(σ). A signal that is loud *relative to the noise at its own frequencies* now stands
out. This is the standard first step of every LIGO-Virgo-KAGRA transient analysis (see
Abbott et al. 2020, *CQG* 37, 055002).

After whitening we **bandpass** to the band where the signal can be. gwsonify does both
through gwpy, the community's standard tool.
"""),
        code("""
settings = strain.StrainSettings(mode="whiten", window=(-3, 1), band=(20, 300))
white = strain.condition(raw, ev.gps, settings).data
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(white.times.value - ev.gps, white.value, lw=0.7)
ax.set(xlabel="time relative to merger [s]", ylabel="whitened strain [σ]",
       title="Whitened, 20-300 Hz bandpassed H1 strain")
plt.show()
"""),
        md("""
Now the signal is visible by eye near $t = 0$. Listen to the whitened data from both
detectors, panned left (Hanford) and right (Livingston):
"""),
        code("""
white_audio = gwsonify.sonify_data("GW150914", stereo=True)
display(Image(filename=str(white_audio.outdir / "GW150914_data-whiten.png"), width=900))
white_audio.play("stereo")
"""),
        md("""
**✎ Try this.** Change the band to `band=(20, 1000)` or `band=(80, 300)` and listen
again. What happens to the "hiss", and what happens to the chirp? Then try
`mode="bandpass"` (bandpass *without* whitening). Why does it sound so different?
"""),
        code("""
# your turn
gwsonify.sonify_data("GW150914", detectors=["L1"], band=(20, 1000), plots=False).play("L1")
"""),
        md("""
## 3. Why the chirp sweeps upward

Two compact objects orbit each other and radiate gravitational waves at **twice** the
orbital frequency. Radiating energy shrinks the orbit, a smaller orbit is faster, and a
faster orbit radiates more strongly. The frequency and amplitude therefore run away
together: a **chirp**.

To leading (Newtonian) order the time left before merger when the wave frequency is
$f$ depends only on the **chirp mass** $\\mathcal{M} = (m_1 m_2)^{3/5}/(m_1+m_2)^{1/5}$:

$$\\tau(f) = \\frac{5}{256}\\,(\\pi f)^{-8/3}\\left(\\frac{G\\mathcal{M}}{c^3}\\right)^{-5/3}.$$

gwsonify uses exactly this formula (`Event.time_to_merger`) to choose how much data to
play. Let's overlay it on the time-frequency map of the data.
"""),
        code("""
q = white_audio.series["L1"]  # conditioned data
f = np.geomspace(30, 250, 200)
tau = np.array([ev.time_to_merger(fi) for fi in f])
qt = raw.q_transform(outseg=(ev.gps - 0.6, ev.gps + 0.1), frange=(20, 300), qrange=(4, 64))
fig, ax = plt.subplots(figsize=(9, 4))
ax.pcolormesh(qt.times.value - ev.gps, qt.frequencies.value, qt.value.T, vmax=25)
ax.plot(-tau, f, "w--", lw=2, label="Newtonian chirp, $\\\\mathcal{M}$ from the catalog")
ax.set(yscale="log", ylim=(20, 300), xlabel="time relative to merger [s]",
       ylabel="frequency [Hz]", title="H1 Q-transform and the leading-order chirp")
ax.legend(loc="upper left"); plt.show()
"""),
        md("""
The leading-order formula follows the track until the last few cycles, where the
objects merge and general relativity must be solved in full.

**✎ Try this.** Heavier binaries sweep faster and end at lower frequencies. Compare
`gwsonify.get_event("GW190521").time_to_merger(20)` (about 150 solar masses in total)
with GW170817 (two neutron stars). How long is each chirp above 20 Hz?
"""),
        code("""
for name in ["GW170817", "GW150914", "GW190521"]:
    e = gwsonify.get_event(name)
    print(f"{name:10s} {e.source_class:4s} Mc_det ≈ {e.chirp_mass_det:6.2f} Msun   "
          f"time above 20 Hz ≈ {e.time_to_merger(20):7.2f} s")
"""),
        md("""
## 4. Approximants and the posterior

The LVK analyses compare the data with **waveform models**, called *approximants*. They
are approximate solutions of Einstein's equations: phenomenological fits
(`IMRPhenomXPHM`), effective-one-body models (`SEOBNRv4PHM`, `SEOBNRv5PHM`) and
numerical-relativity surrogates (`NRSur7dq4`). The analysis gives a **posterior**: many
thousands of parameter combinations (masses, spins, distance, sky position, ...) that
are consistent with the data, each a *sample*.

`sonify_model` downloads the event's parameter-estimation release, picks a posterior
(the preferred one if it used a single model, otherwise a single-model posterior from the
same release), takes the **maximum-likelihood sample**, generates that waveform with
LALSimulation and projects it onto each detector.
"""),
        code("""
model = gwsonify.sonify_model("GW150914")
m = model.provenance["model"]
print("label:", m["label"], "| approximant:", m["approximant"], "|", m["sample"])
print("why this label:", m["label_choice"])
display(Image(filename=str(model.outdir / "GW150914_posterior.png"), width=900))
display(Image(filename=str(model.outdir / "GW150914_model.png"), width=900))
model.play("H1")
"""),
        md("""
The histograms show the posterior (the spread is our uncertainty). The red line marks
the sample you are hearing.

**✎ Try this.** Listen to a *different* posterior sample, for example `sample=0` or
`sample="median"`, or a different model with `label="C01:SEOBNRv4PHM"`. Can you hear a
difference? (For a loud event like this, the posterior is narrow, so the answer is:
barely.)
"""),
        code("""
# your turn
gwsonify.sonify_model("GW150914", detectors=["H1"], sample="median", plots=False).play("H1")
"""),
        md("""
## 5. Speed, stretch, pitch and frequency shift are different operations

| operation | duration | frequencies | how |
|---|---|---|---|
| `speed=X` | ÷ X | × X | play faster/slower (like tape) |
| `stretch=X` | × X | unchanged | phase vocoder |
| `pitch=S` | unchanged | × 2^(S/12) | phase vocoder + resampling |
| `fshift=HZ` | unchanged | + HZ | single-sideband shift |

`speed` keeps the waveform itself exact: it only relabels time. `fshift` keeps the timing
exact but changes frequency *ratios*: harmonics no longer stack, and the chirp sounds
more like a sweep than a note. `stretch` and `pitch` need a phase vocoder, which smears
the fastest part of the chirp slightly.
"""),
        code("""
for kwargs in [dict(speed=0.25), dict(stretch=4), dict(pitch=12), dict(fshift=300)]:
    r = gwsonify.sonify_model("GW150914", detectors=["L1"], plots=False, **kwargs)
    print(kwargs, f"{len(r.audio['L1']) / r.rate:.2f} s")
    display(r.play("L1"))
"""),
        md("""
**✎ Try this.** Which version would you use to let a room hear the *shape* of the
chirp, and which to demonstrate the *true* frequencies? Why?

## 6. Loudness that can be compared

By default each file is normalised to its own peak (`norm="peak"`), which is good for
listening but hides how loud one event is compared with another. With `norm="fixed"`
the scale is fixed in physical units: 1 σ of whitened noise, or a strain of $10^{-21}$
for models, maps to −30 dBFS. Louder events then really are louder.
"""),
        code("""
for name in ["GW150914", "GW190521"]:
    r = gwsonify.sonify_data(name, detectors=["L1"], norm="fixed", plots=False)
    peak_db = 20 * np.log10(np.max(np.abs(r.audio["L1"])))
    print(f"{name}: peak {peak_db:+.1f} dBFS")
"""),
        md("""
## Where to go next

* `gwsonify list` and `gwsonify info EVENT` in a terminal browse the whole GWOSC catalog.
* The GWOSC tutorials (https://gwosc.org/tutorials/) and the gwpy documentation go much
  deeper into the data analysis.
* Every output folder has a `*.provenance.json` with the exact inputs, software versions
  and data URLs, plus the GWOSC acknowledgement to cite.
"""),
    ]


def write(name, cells, title):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "title": title}
    ROOT.mkdir(exist_ok=True)
    nbf.write(nb, ROOT / name)
    print("wrote", ROOT / name)


if __name__ == "__main__":
    write("01_instructor_quicktour.ipynb", instructor(), "gwsonify instructor quick tour")
    write("02_self_study.ipynb", self_study(), "gwsonify self-study")
