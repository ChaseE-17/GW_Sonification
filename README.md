# gwsonify

**Hear gravitational waves.** `gwsonify` turns public data from the
[Gravitational Wave Open Science Center](https://gwosc.org) (GWOSC) into audio, figures
and lecture-ready video: the real detector data around any event, or the waveform
model from the event's published parameter-estimation (PE) results.

It is built for **researchers** who want a quick, citable listening tool and for
**educators** who want to show a room what a black-hole merger sounds like.

```console
$ gwsonify GW150914
GW150914: whiten from H1, L1, window -3 to +1 s, band 20-300 Hz, 4096 Hz
Downloading H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5 (1.0 MB) to the gwsonify cache ...
Downloading L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5 (1.0 MB) to the gwsonify cache ...
gwsonify-output/GW150914/GW150914_H1_data-whiten.wav
gwsonify-output/GW150914/GW150914_L1_data-whiten.wav
gwsonify-output/GW150914/GW150914_data-whiten.png
gwsonify-output/GW150914/GW150914_data-whiten.provenance.json
```

* Works across the whole GWOSC catalog (GWTC-1 through GWTC-5.0 and the O4 discovery
  papers). Event names, versions, detectors, posterior labels and GPS times are all
  discovered from the GWOSC API and the data files. None are hard-coded.
* Uses the research-standard methods of the LIGO-Virgo-KAGRA collaboration for
  conditioning the data (gwpy whitening, bandpassing and gating) and for regenerating
  waveforms (pesummary and LALSimulation, with the settings recorded in each PE release).
* Pitch and time can be changed with four distinct, correctly named operations.
  Fades prevent clicks, and loudness can be normalised so events are comparable.
* Every run writes a provenance record with software versions, data URLs and checksums,
  parameters, and the GWOSC acknowledgement text.

## Install

```bash
pip install "gwsonify[model,video] @ git+https://github.com/ChaseE-17/GW_Sonification"
```

Python 3.10 or newer. The extras are optional:

| extra | adds | needed for |
|---|---|---|
| *(none)* | numpy, scipy, matplotlib, gwpy, gwosc, h5py, requests | `gwsonify data`, `info`, `list` on every OS |
| `model` | pesummary, lalsuite | `gwsonify model`, `--template` (Linux and macOS wheels) |
| `video` | imageio-ffmpeg | `--video` if ffmpeg is not already installed |
| `skymap` | ligo.skymap | reserved for sky-map figures |
| `notebook` | jupyterlab + model + video | the teaching notebooks |

**conda / Windows.** LALSuite has no Windows wheels on PyPI. Use conda-forge, which
has everything:

```bash
conda create -n gwsonify -c conda-forge python=3.12 gwpy pesummary python-lalsimulation ffmpeg
conda activate gwsonify
pip install "gwsonify @ git+https://github.com/ChaseE-17/GW_Sonification"
```

## Quick start

```bash
gwsonify GW150914                          # whitened H1 + L1 data: listen for the chirp at the end
gwsonify GW150914 --fshift 400             # shifted up 400 Hz (as in the GWOSC audio files)
gwsonify GW150914 --template --stereo      # add the whitened waveform model
gwsonify GW150914 --fshift 400 --video     # MP4 with title card and moving playhead
gwsonify model GW170817 --f-low 30         # the neutron-star inspiral model, about a minute long
gwsonify info GW190521                     # what GWOSC has for an event
gwsonify list GW2501                       # search event names
```

`gwsonify EVENT` is short for `gwsonify data EVENT`. Every command has `--help` with
examples.

## Commands

| command | what it does |
|---|---|
| `gwsonify data EVENT` (default) | sonify detector strain: whitened (default), bandpassed or raw |
| `gwsonify data --gps GPS -d H1,L1` | the same for any GPS time |
| `gwsonify model EVENT` | sonify the detector-projected waveform of one posterior sample |
| `gwsonify info EVENT [--json]` | catalog version, parameters, detectors with data, PE analyses |
| `gwsonify list [PATTERN] [--catalog C] [--json]` | list or search events |
| `gwsonify cache [--clear]` | where downloads are cached, and how much space they use |

`EVENT` can be a short or long name (`GW150914`, `GW190521`, `GW190521_030229`) or an
explicit release (`GW150914-v3`, `GW150914@GWTC-1-confident`). Unknown names get
suggestions (`No GWOSC event named 'GW15091'. Did you mean: GW150914?`).

### Options shared by `data` and `model`

| option | default | meaning |
|---|---|---|
| `-d, --detectors H1,L1` | all available | detectors to use |
| `--speed X` | 1 | playback speed: duration ÷ X, frequencies × X |
| `--stretch X` | 1 | duration × X with the pitch unchanged |
| `--pitch SEMITONES` | 0 | pitch shift with the duration unchanged |
| `--fshift HZ` | 0 | add HZ to every frequency |
| `--fade SEC` | 0.05 | raised-cosine fade in and out |
| `--norm {peak,fixed}` | peak | normalisation (see below) |
| `--level DBFS` | −3 (peak) / −30 (fixed) | target level |
| `--stereo` | off | one stereo file, detectors panned left to right |
| `--video` | off | also write an MP4 (needs ffmpeg) |
| `-o, --outdir DIR` | `./gwsonify-output` | outputs go to `DIR/<EVENT>/` |
| `--no-plots`, `--json`, `-y/--yes`, `-q`, `-v` | | skip figures; JSON summary on stdout; allow very large downloads; quiet; verbose |

### `data` options

| option | default | meaning |
|---|---|---|
| `--mode {whiten,bandpass,raw}` | whiten | whiten + bandpass; bandpass + mains notches; untouched |
| `--window START END` | from the chirp duration | seconds relative to merger |
| `--band LOW HIGH` | 20 Hz to 300-1000 Hz | bandpass edges |
| `--sample-rate {4096,16384}` | 4096 | GWOSC strain sample rate |
| `--no-gate` | gating on in whiten mode | turn off gating of loud glitches |
| `--template` | off | add the whitened waveform model plus data/model stereo overlays |

### `model` options

| option | default | meaning |
|---|---|---|
| `--label L` | see below | posterior label in the PE file (`gwsonify info` lists them) |
| `--approximant A` | the label's | waveform model |
| `--sample {maxl,maxp,median,INDEX}` | maxl | which posterior sample |
| `--f-low HZ` | 20 | waveform start frequency |
| `--sample-rate HZ` | 4096, or 8192 with a neutron star | waveform sampling |

## How the defaults are chosen

* **Event version.** Uses the newest version from a confident catalog; marginal,
  auxiliary and preliminary catalogs are used only if nothing else exists. For example,
  `GW150914` means `GW150914-v4` (GWTC-2.1).
* **Detectors.** Uses those for which GWOSC actually publishes strain, which is not
  always the same as the event's detector list. For GW250114, V1 is listed but its strain
  is not published.
* **Strain file.** Uses a file from the event release you asked for when one covers the span (then newer, then older releases, then the bulk run files), and the smallest such file: 1 MB
  32-second event files when they exist, otherwise the 4096-second bulk files (40-130 MB).
* **Window.** Starts τ + 1 s before merger, where τ is the leading-order time for the chirp
  to sweep from 20 Hz to merger for the catalog's chirp mass,
  τ = (5/256)(πf)^(−8/3)(G𝓜/c³)^(−5/3). The lead is clipped to 3-120 s, and the window
  ends 1 s after merger. That gives −3 to +1 s for GW150914 and −120 to +1 s for GW170817.
* **Band.** 20 Hz up to 3 f_ISCO of the total mass, clipped to 300-1000 Hz. That is
  300 Hz for GW150914 (the discovery paper used 35-350 Hz) and 1000 Hz for neutron-star
  binaries.
* **Whitening.** gwpy's median-Welch ASD (4 s Hann segments, 50% overlap) from at
  least 28 s of data around the window. Filtering is done on padded data and then
  cropped, so filter edges never reach the audio.
* **Gating (whiten mode).** gwpy's inverse-Tukey gate at 50 σ removes only extreme
  glitches, such as the one next to GW170817 in L1, never normal noise or ordinary chirps.
  Gates are reported and recorded. Turn it off with `--no-gate`.
* **Posterior label (`model`).** Uses the GWOSC-preferred analysis, matched against the
  labels actually in the file (GWOSC and file labels sometimes differ). Mixed-model
  posteriors (`C01:Mixed`, `C00:Mixed`) have no single waveform model. In that case,
  or when the preferred model cannot be generated here (NRSur7dq4 without its data
  files, SEOBNRv5PHM without pyseobnr), gwsonify uses the best single-model posterior in
  the same release. The order is IMRPhenomXPHM, IMRPhenomXO4a, other IMRPhenom models,
  then SEOBNRv4PHM. The choice and the reason are printed and recorded.
* **Waveform settings.** The reference frequency and approximant flags come from the PE
  file (for example `PhenomXPrecVersion=320` for the SpinTaylor variant), and tidal
  deformabilities are passed for neutron-star models. Parameters a file does not contain
  (GWTC-1 files lack the polarisation angle, coalescence phase and time) are filled with
  documented defaults and listed under `assumed_parameters` in the provenance.
  The posterior itself is never modified.
* **Sample.** The maximum-likelihood sample. Files without likelihoods (GWTC-1) use
  the real sample closest to the posterior medians.

## Pitch and time

These four operations are different, and `gwsonify` keeps them separate:

| option | duration | frequencies | method |
|---|---|---|---|
| `--speed X` | ÷ X | × X | band-limited resampling ("tape speed"); the waveform itself is untouched |
| `--stretch X` | × X | unchanged | phase vocoder with identity phase locking |
| `--pitch S` | unchanged | × 2^(S/12) | phase vocoder, then resampling |
| `--fshift HZ` | unchanged | + HZ | single-sideband shift of the analytic signal |

`--fshift 400` reproduces the GWOSC audio releases. It moves the 30-300 Hz chirp into
the most sensitive part of human hearing and keeps the timing exact, but it changes
frequency *ratios*. `--speed` is the most faithful way to slow a chirp down for
teaching. `--stretch` and `--pitch` smear the final, fastest cycles slightly (an
inherent phase-vocoder effect).

## Loudness

`--norm peak` (default) scales each run so its loudest sample is at −3 dBFS. All
detectors in a run share one gain, so their relative loudness is preserved.
`--norm fixed` uses a fixed physical scale instead: 1 σ of whitened noise, or a strain of
10⁻²¹ for waveform models, maps to `--level` (default −30 dBFS). The same settings
then give the same loudness for the same amplitude, across events and detectors.

## Outputs

Everything goes to `gwsonify-output/<EVENT>/` (change it with `-o` or
`$GWSONIFY_OUTDIR`). The file names are predictable:

| file | contents |
|---|---|
| `<EVENT>_<DET>_data-whiten.wav` | mono 16-bit, 44.1 kHz audio per detector |
| `<EVENT>_data-whiten_stereo.wav` | with `--stereo`: detectors panned left to right |
| `<EVENT>_<DET>_template-whiten.wav`, `..._overlay.wav` | with `--template`: whitened model; data left, model right |
| `<EVENT>_data-whiten.png` | Q-transform and time series for each detector |
| `<EVENT>_data-whiten.mp4` | with `--video`: title card, then the figure with a moving playhead |
| `<EVENT>_model*.wav/png/mp4`, `<EVENT>_posterior.png` | the same for `gwsonify model`, plus posterior histograms |
| `*.provenance.json` | inputs, event metadata, data URLs and SHA-256, processing parameters, model sample, software versions, citation text |

Non-default audio settings are added to the name (for example
`GW150914_data-whiten_speed0.5_fshift400.png`), so runs with different settings don't
overwrite each other.

## Python API

The CLI is a thin layer over the library:

```python
import gwsonify

r = gwsonify.sonify_data("GW150914", fshift=400, stereo=True)
r.files            # written paths
r.play("stereo")   # inline player in Jupyter
r.provenance       # the provenance record

m = gwsonify.sonify_model("GW170817", detectors=["L1"], f_low=30, speed=0.5)
ev = gwsonify.get_event("GW190521")   # name, gps, catalog, parameters, PE analyses, ...
```

Lower-level building blocks live in `gwsonify.audio` (DSP), `gwsonify.strain`
(loading and conditioning), `gwsonify.model` (posteriors and waveforms) and
`gwsonify.catalog` (GWOSC API).

## Teaching notebooks

* [`notebooks/01_instructor_quicktour.ipynb`](notebooks/01_instructor_quicktour.ipynb):
  a 10-minute demo for a lecture.
* [`notebooks/02_self_study.ipynb`](notebooks/02_self_study.ipynb): strain, whitening,
  why the chirp sweeps upward, approximants and posteriors, and pitch versus time, with
  exercises.

They install gwsonify automatically on Google Colab: open them with
`https://colab.research.google.com/github/ChaseE-17/GW_Sonification/blob/main/notebooks/<name>.ipynb`.

## Downloads and cache

Downloads are cached in `~/Library/Caches/gwsonify` (macOS), `~/.cache/gwsonify` (Linux)
or `%LOCALAPPDATA%\gwsonify\Cache` (Windows). Override the location with
`$GWSONIFY_CACHE_DIR`. Each download is announced with its size, and cache hits are
reported. Files over 1 GB ask for confirmation, or need `--yes` when not interactive
(`$GWSONIFY_MAX_DOWNLOAD_MB` changes the limit). Some GWTC-5.0 PE releases are 1.7 GB.
When that happens, gwsonify suggests smaller releases of the same event, for example
`GW250114_082203-v1 (O4_Discovery_Papers, 27 MB)`. `gwsonify cache --clear` empties
the cache.

## Exit codes

`0` success · `1` error (clear message, no traceback; `-v` shows details) · `2` usage
error · `3` missing optional dependency (the message names the extra) · `130`
interrupted.

## Citing and acknowledging

If you use gwsonify, please cite it (see [`CITATION.cff`](CITATION.cff); GitHub shows a
"Cite this repository" button). Please also:

1. **Acknowledge GWOSC.** The text is in every provenance file and at
   https://gwosc.org/acknowledgement/:
   > This research has made use of data or software obtained from the Gravitational Wave
   > Open Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the
   > Virgo Collaboration, and KAGRA. …
2. **Cite the data release paper** for the observing run. Provenance records it under
   `citation.data_release_paper`: O1/O2 SoftwareX 13, 100658 (2021); O3 ApJS 267, 29
   (2023); O4a ApJ 1004, 2329 (2026).
3. **Cite the event's catalog paper and data DOI** (`citation.event_doi`).

## Migrating from `gw_sonification_pipeline.py`

The old single script and its copy-pasted notebook have been replaced by this package.

| old | new |
|---|---|
| `python gw_sonification_pipeline.py GW150914 L1` | `gwsonify model GW150914 -d L1` |
| `--approximant SEOBNRv4PHM` (label `C01:{approximant}` assumed) | `--label C01:SEOBNRv4PHM` (labels are discovered; see `gwsonify info`) |
| `--pitch_shift 2` (did nothing) | `--pitch 12` (one octave) or `--speed 2` or `--fshift HZ` |
| `--time_stretch 10` (also lowered the pitch) | `--stretch 10` (pitch kept) or `--speed 0.1` (pitch lowered) |
| `--gain 0.5` | `--level -6` (dBFS) |
| `--file_name_out x.wav` | `-o DIR` (predictable names inside `DIR/<EVENT>/`) |
| PE file, PNGs, WAV written to the current directory | outputs in `gwsonify-output/<EVENT>/`, downloads in the user cache |
| corner, spin-disk and sky-map plots | `<EVENT>_posterior.png` (key parameter histograms) |

New: `gwsonify data` sonifies the detector data itself (whitened, bandpassed or raw),
with an optional whitened-template overlay.

## Development

```bash
pip install -e ".[dev]"
pytest                       # offline tests (DSP, catalog, selection, CLI, pipeline)
pytest --online -m online    # real GWOSC round trips
pytest --notebooks -m notebook
python scripts/sweep.py      # GWOSC compatibility sweep -> docs/sweep-results.md
python scripts/build_notebooks.py   # regenerate the notebooks from source
ruff check src tests scripts
```

## License

MIT. See [`LICENSE`](LICENSE).
