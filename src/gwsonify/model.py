"""Waveform models from GWOSC parameter-estimation (PE) posteriors.

Needs the ``model`` extra (``pip install "gwsonify[model]"``: pesummary and lalsuite).

How the analysis is chosen
--------------------------
1. GWOSC marks one PE analysis per event version as *preferred*. It gives the release
   file and a label, for example ``C01:Mixed`` or ``C00:IMRPhenomXPHM-SpinTaylor``.
2. The file is opened and its labels are **discovered** (``data.labels``). The API
   label is matched against them. GWOSC and file labels do not always agree: for
   GW250114 the API says ``PhenomXPHM`` but the file says
   ``bilby-IMRPhenomXPHM-SpinTaylor_prod-reweighted``.
3. A waveform can only be regenerated from a posterior produced with one waveform
   model. ``Mixed`` labels combine samples from several models, so they have no
   single approximant. In that case, and whenever the preferred approximant cannot
   be generated here (for example NRSur7dq4 without its data file, or SEOBNRv5PHM
   without pyseobnr), the first *generatable* label in the file is used, in this order:
   IMRPhenomXPHM family, IMRPhenomXO4a, other IMRPhenomX/Pv2/NSBH models (including
   NRTidal variants), SEOBNRv4PHM, then anything else. The choice is logged and written
   to the provenance sidecar. ``label=`` and ``approximant=`` override it.

The sample defaults to the maximum-likelihood sample (``maxl``). ``maxp`` gives the
maximum-posterior sample, ``median`` gives the actual sample closest to the
per-parameter medians of the masses, spins and distance, and an integer selects a
sample by index. Files without likelihoods (GWTC-1) fall back to ``median``.

Waveform settings follow the analysis: the reference frequency and approximant flags
stored in the file (for example ``PhenomXPrecVersion=320`` for the SpinTaylor version
of IMRPhenomXPHM) are passed to LALSimulation, and tidal deformabilities are passed
for neutron-star models. Parameters the file does not contain (for example the
polarisation angle or coalescence time in GWTC-1 releases) are filled with documented
defaults and listed in ``Waveform.assumed``. The posterior itself is never modified.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gwsonify import GwsonifyError, MissingDependencyError
from gwsonify.cache import LargeDownloadError, fetch, max_download_bytes
from gwsonify.catalog import Event, PEAnalysis

log = logging.getLogger(__name__)

SAMPLE_CHOICES = ("maxl", "maxp", "median")

# preference order for regenerating a waveform (regular expressions on approximant)
_PREFERENCE = [
    r"^IMRPhenomXPHM$", r"^IMRPhenomXO4a$", r"^IMRPhenomXP", r"^IMRPhenomXHM$",
    r"^IMRPhenomXAS", r"^IMRPhenomPv2", r"^IMRPhenomNSBH$", r"^IMRPhenomD",
    r"^IMRPhenom", r"^SEOBNRv4PHM$", r"^SEOBNRv4", r"",
]

# label fragments that do not match a LALSimulation name directly
_ALIASES = {
    "IMRPhenomPv2NRT": "IMRPhenomPv2_NRTidal",
    "IMRPhenomPv2NRTidal": "IMRPhenomPv2_NRTidal",
    "PhenomXPHM": "IMRPhenomXPHM",
    "PhenomXO4a": "IMRPhenomXO4a",
    "PhenomXAS": "IMRPhenomXAS",
    "PhenomNSBH": "IMRPhenomNSBH",
    "PhenomPv2": "IMRPhenomPv2",
    "NRSur7dq4": "NRSur7dq4",
}
_SPINTAYLOR_FLAGS = {"PhenomXPrecVersion": 320}


def _require():
    try:
        import lalsimulation  # noqa: F401
        import pesummary
        import pesummary.io  # noqa: F401  (sets up pesummary's log handlers)
    except ImportError as exc:
        raise MissingDependencyError(
            "Waveform models need the 'model' extra: pip install \"gwsonify[model]\" "
            "(or: conda install -c conda-forge pesummary python-lalsimulation). "
            f"Missing: {exc.name}"
        ) from None
    # pesummary logs install hints and file-format notes that are noise here. It resets
    # its logger level internally, so raise the level of its handlers instead.
    if not logging.getLogger("gwsonify").isEnabledFor(logging.DEBUG):
        for handler in logging.getLogger("PESummary").handlers:
            handler.setLevel(logging.ERROR)


def lalsim_approximants() -> list[str]:
    import lalsimulation as ls

    return [
        ls.GetStringFromApproximant(i) for i in range(ls.NumApproximants)
        if ls.SimInspiralImplementedTDApproximants(i) or ls.SimInspiralImplementedFDApproximants(i)
    ]


def approximant_from_label(label: str, known: list[str]) -> tuple[str | None, dict]:
    """Infer ``(approximant, flags)`` from a posterior label (``None`` for mixtures).

    >>> approximant_from_label("C00:IMRPhenomXPHM-SpinTaylor", ["IMRPhenomXPHM"])
    ('IMRPhenomXPHM', {'PhenomXPrecVersion': 320})
    """
    core = label.split(":", 1)[1] if ":" in label else label
    if re.search(r"mixed|overall|combined|publicationsamples", core, re.IGNORECASE):
        return None, {}
    flags = dict(_SPINTAYLOR_FLAGS) if "spintaylor" in core.lower() else {}
    core = re.sub(r"^(bilby|lalinference|rift)[-_]", "", core, flags=re.IGNORECASE)
    tokens = re.split(r"[-_:]", core)
    # try successively shorter prefixes of the tokenised label
    for n in range(len(tokens), 0, -1):
        for joiner in ("_", ""):
            cand = joiner.join(tokens[:n])
            cand = _ALIASES.get(cand, cand)
            if cand in known:
                return cand, flags
    return None, flags


def rank_approximant(approximant: str | None) -> int:
    if not approximant:
        return len(_PREFERENCE) + 1
    for i, pattern in enumerate(_PREFERENCE):
        if re.search(pattern, approximant):
            return i
    return len(_PREFERENCE)  # pragma: no cover


# ---------------------------------------------------------------------------------------
# Posterior files
# ---------------------------------------------------------------------------------------


@dataclass
class Posterior:
    """A PE release file opened with pesummary, with its labels discovered."""

    path: Path
    url: str
    analysis: PEAnalysis
    labels: list[str]
    approximants: dict[str, str | None]
    meta: dict[str, dict]
    _data: object = field(repr=False, default=None)
    _gwtc1: bool = False

    _samples: dict = field(repr=False, default_factory=dict)

    def samples(self, label: str):
        """pesummary ``SamplesDict`` for ``label`` (cached)."""
        if label not in self._samples:
            if self._gwtc1:
                from pesummary.io import read

                self._samples[label] = read(
                    str(self.path), path_to_samples=f"{label}_posterior").samples_dict
            else:
                self._samples[label] = self._data.samples_dict[label]
        return self._samples[label]

    def skymap(self, label: str):
        sky = getattr(self._data, "skymap", None) if self._data is not None else None
        try:
            return sky[label] if sky and label in sky else None
        except Exception:  # pragma: no cover - pesummary internals
            return None


def match_label(api_label: str, labels: list[str]) -> str | None:
    """Find the file label that corresponds to the GWOSC label."""
    if api_label in labels:
        return api_label
    low = {lab.lower(): lab for lab in labels}
    stripped = re.sub(r"_prior$|_posterior$", "", api_label)
    if stripped.lower() in low:
        return low[stripped.lower()]
    if len(labels) == 1:
        return labels[0]
    core = api_label.split(":", 1)[-1].lower()
    hits = [lab for lab in labels if core in lab.lower()]
    return hits[0] if len(hits) == 1 else None


def version_with_pe(event: Event) -> Event:
    """``event`` if it has a PE release, else the newest other version of it that does.

    Some catalog versions (for example a newer marginal re-listing) carry no PE release
    even though an earlier version does. Returns ``event`` unchanged when none has one.
    """
    if event.preferred_pe() is not None:
        return event
    from gwsonify.catalog import get_event

    for version, catalog in sorted(event.other_versions, reverse=True):
        other = get_event(f"{event.name}-v{version}")
        if other.preferred_pe() is not None:
            log.info("%s has no PE release; using %s (%s), which does.", event.spec,
                     other.spec, catalog)
            return other
    return event


def _smaller_alternatives(event: Event, limit: int) -> list[str]:
    """Other versions of ``event`` whose preferred PE file is below ``limit`` bytes."""
    import requests

    from gwsonify.catalog import get_event

    out = []
    for version, catalog in event.other_versions:
        try:
            other = get_event(f"{event.name}-v{version}")
            pe = other.preferred_pe()
            if pe is None:
                continue
            head = requests.head(pe.data_url, allow_redirects=True, timeout=20)
            size = int(head.headers.get("content-length") or 0)
        except Exception:
            continue
        if 0 < size <= limit:
            out.append(f"{other.spec} ({catalog}, {size / 1e6:.0f} MB)")
    return out


def open_posterior(event: Event, analysis: PEAnalysis | None = None, *, quiet: bool = False,
                   yes: bool = False) -> Posterior:
    """Download (cached) and open the PE release file for ``event``."""
    _require()
    import h5py
    from pesummary.io import read

    analysis = analysis or event.preferred_pe()
    if analysis is None:
        raise GwsonifyError(
            f"GWOSC lists no parameter-estimation release for {event.spec} ({event.catalog}), "
            "so no waveform model is available. Try `gwsonify data "
            f"{event.name}`, or another version (see `gwsonify info {event.name}`)."
        )
    try:
        path = fetch(analysis.data_url, "pe", quiet=quiet, yes=yes)
    except LargeDownloadError as exc:
        alts = _smaller_alternatives(event, max_download_bytes())
        if alts:
            exc.args = (f"{exc.args[0]} Smaller PE releases of this event: {', '.join(alts)}.",)
        raise
    known = lalsim_approximants()

    with h5py.File(path, "r") as f:
        tables = [k for k in f if k.endswith("_posterior")]
        gwtc1 = bool(tables) and all(isinstance(f[k], h5py.Dataset) for k in tables)
    if gwtc1:
        labels = [t[: -len("_posterior")] for t in tables]
        approx = {lab: approximant_from_label(lab, known)[0] for lab in labels}
        meta = {lab: {} for lab in labels}
        return Posterior(path, analysis.data_url, analysis, labels, approx, meta, None, True)

    log.info("Reading %s ...", path.name.split("-", 1)[-1])
    try:
        data = read(str(path), disable_prior=True)
    except Exception as exc:
        raise GwsonifyError(f"pesummary could not read {path.name}: {exc}") from exc
    labels = list(getattr(data, "labels", None) or [])
    if not labels:
        raise GwsonifyError(f"No posterior labels found in {path.name}.")
    file_approx = list(getattr(data, "approximant", None) or [None] * len(labels))
    extra = getattr(data, "extra_kwargs", None) or [{}] * len(labels)
    approx, meta = {}, {}
    for lab, fa, ek in zip(labels, file_approx, extra, strict=False):
        md = dict((ek or {}).get("meta_data", {}) or {})
        guess, flags = approximant_from_label(lab, known)
        a = fa if isinstance(fa, str) and fa in known else guess
        if a and not md.get("approximant_flags") and flags:
            md["approximant_flags"] = flags
        approx[lab] = a
        meta[lab] = md
    return Posterior(path, analysis.data_url, analysis, labels, approx, meta, data, False)


def choose_label(post: Posterior, label: str | None = None) -> tuple[str, str]:
    """Return ``(label, reason)`` following the rules in the module docstring."""
    if label is not None:
        if label not in post.labels:
            raise GwsonifyError(
                f"Label {label!r} is not in this PE file. Available: {', '.join(post.labels)}"
            )
        return label, "requested"
    preferred = match_label(post.analysis.label, post.labels)
    if preferred and post.approximants.get(preferred):
        return preferred, f"GWOSC preferred analysis ({post.analysis.label})"
    candidates = sorted(
        (lab for lab in post.labels if post.approximants.get(lab)),
        key=lambda lab: rank_approximant(post.approximants[lab]),
    )
    if not candidates:
        raise GwsonifyError(
            "None of the posteriors in this file has a waveform model LALSimulation can "
            f"generate (labels: {', '.join(post.labels)}). Pass --approximant explicitly."
        )
    why = (f"the GWOSC preferred analysis ({post.analysis.label}) combines several waveform "
           f"models; using the single-model posterior {candidates[0]}")
    return candidates[0], why


# ---------------------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------------------

_INTRINSIC = ("mass_1", "mass_2", "a_1", "a_2", "chi_eff", "luminosity_distance")


def pick_sample(samples, which: str | int = "maxl") -> tuple[dict, str, int]:
    """Return ``(sample, description, index)`` from a pesummary SamplesDict."""
    params = list(samples.parameters)
    n = samples.number_of_samples
    arr = {p: np.asarray(samples[p], dtype=float) for p in params}
    logl = arr.get("log_likelihood")
    has_l = logl is not None and np.ptp(logl) > 0
    if which == "maxl" and not has_l:
        log.info("No likelihood values in this file; using the median sample instead of maxL.")
        which = "median"
    if which == "maxl":
        idx = int(np.argmax(logl))
        desc = "maximum-likelihood sample"
    elif which == "maxp":
        logp = arr.get("log_prior")
        if not has_l or logp is None:
            raise GwsonifyError("This posterior has no likelihood/prior values; use --sample median.")
        idx = int(np.argmax(logl + logp))
        desc = "maximum-posterior sample"
    elif which == "median":
        cols = [arr[p] for p in _INTRINSIC if p in arr and np.std(arr[p]) > 0]
        z = np.stack([(c - np.median(c)) / np.std(c) for c in cols], axis=1)
        idx = int(np.argmin(np.sum(z**2, axis=1)))
        desc = "sample closest to the posterior medians"
    else:
        try:
            idx = int(which)
        except ValueError:
            raise GwsonifyError(f"--sample must be maxl, maxp, median or an index, not {which!r}"
                                ) from None
        if not 0 <= idx < n:
            raise GwsonifyError(f"--sample index {idx} out of range (0-{n - 1}).")
        desc = f"sample #{idx}"
    return {p: float(arr[p][idx]) for p in params}, desc, idx


def complete_sample(sample: dict, event: Event) -> tuple[dict, dict]:
    """Return a copy of ``sample`` with everything waveform generation needs.

    Only fills parameters the file lacks. The second return value lists what was
    assumed (it goes into the provenance sidecar).
    """
    s = dict(sample)
    assumed: dict[str, float] = {}
    for cos, ang in (("cos_theta_jn", "theta_jn"), ("cos_tilt_1", "tilt_1"),
                     ("cos_tilt_2", "tilt_2"), ("cos_iota", "iota")):
        if ang not in s and cos in s:
            s[ang] = float(np.arccos(np.clip(s[cos], -1, 1)))
    defaults = {"phi_jl": 0.0, "phi_12": 0.0, "phase": 0.0, "psi": 0.0,
                "geocent_time": float(event.gps), "a_1": 0.0, "a_2": 0.0,
                "tilt_1": 0.0, "tilt_2": 0.0}
    has_cartesian = any(k in s for k in ("spin_1z", "spin_2z", "chi_1", "chi_2"))
    if has_cartesian:
        for k in ("a_1", "a_2", "tilt_1", "tilt_2", "phi_jl", "phi_12"):
            defaults.pop(k)
        s.setdefault("spin_1z", s.get("chi_1", 0.0))
        s.setdefault("spin_2z", s.get("chi_2", 0.0))
    if "theta_jn" not in s and "iota" in s and not has_cartesian:
        s["theta_jn"] = s["iota"]
    for k, v in defaults.items():
        if k not in s:
            s[k] = v
            assumed[k] = v
    if "theta_jn" not in s and "iota" not in s:
        s["theta_jn"] = s["iota"] = 0.0
        assumed["theta_jn"] = 0.0
    return s, assumed


# ---------------------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------------------


@dataclass
class Waveform:
    """Detector-projected model waveforms plus everything needed to reproduce them."""

    strains: dict            # detector -> gwpy TimeSeries (absolute GPS times)
    label: str
    approximant: str
    reason: str
    sample: dict
    sample_desc: str
    sample_index: int
    f_low: float
    f_ref: float
    sample_rate: float
    flags: dict
    assumed: dict
    posterior: Posterior

    def provenance(self) -> dict:
        return {
            "pe_file_url": self.posterior.url,
            "pe_analysis": self.posterior.analysis.name,
            "gwosc_label": self.posterior.analysis.label,
            "labels_in_file": self.posterior.labels,
            "label": self.label,
            "label_choice": self.reason,
            "approximant": self.approximant,
            "approximant_flags": self.flags,
            "sample": self.sample_desc,
            "sample_index": self.sample_index,
            "sample_parameters": self.sample,
            "assumed_parameters": self.assumed,
            "f_low": self.f_low,
            "f_ref": self.f_ref,
            "sample_rate": self.sample_rate,
        }


def _lal_dict(flags: dict, sample: dict):
    import lal
    import lalsimulation as ls

    d = lal.CreateDict()
    for key, value in flags.items():
        fn = getattr(ls, f"SimInspiralWaveformParamsInsert{key}", None)
        if fn is None:
            log.warning("LALSimulation has no flag %s; ignored.", key)
        else:
            fn(d, int(value))
    for i in (1, 2):
        lam = sample.get(f"lambda_{i}")
        if lam:
            getattr(ls, f"SimInspiralWaveformParamsInsertTidalLambda{i}")(d, float(lam))
    return d


def _quiet_lal():
    """Route LAL's C-level stderr through Python so it can be captured."""
    import lal

    try:
        lal.swig_redirect_standard_output_error(True)
    except AttributeError:  # pragma: no cover
        pass


def _try_labels(post: Posterior, event: Event, detectors: list[str], order: list[str],
                approximant: str | None, sample, f_low: float, sample_rate: float,
                errors: list[str], stop_on_error: bool) -> tuple | None:
    import contextlib
    import io

    from pesummary.gw.waveform import td_waveform

    for lab in order:
        approx = approximant or post.approximants.get(lab)
        if not approx:
            raise GwsonifyError(f"Label {lab!r} has no single approximant; pass --approximant.")
        meta = post.meta.get(lab, {})
        flags = {k: int(v) for k, v in dict(meta.get("approximant_flags") or {}).items()}
        f_ref = float(meta.get("f_ref") or 20.0)
        raw, desc, idx = pick_sample(post.samples(lab), sample)
        full, assumed = complete_sample(raw, event)
        if not meta.get("f_ref"):
            assumed["f_ref"] = f_ref
        strains = {}
        captured = io.StringIO()
        try:
            with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
                for det in detectors:
                    h = td_waveform(full, approx, 1.0 / sample_rate, f_low, f_ref=f_ref,
                                    project=det, LAL_parameters=_lal_dict(flags, full))
                    h = h.taper(side="left")
                    h.name = det
                    strains[det] = h
        except Exception as exc:
            log.debug("LAL output:\n%s", captured.getvalue())
            msg = str(exc).splitlines()[0] if str(exc) else repr(exc)
            hint = " (needs the NRSurrogate data files)" if "NRSur" in approx else ""
            errors.append(f"{lab} ({approx}{hint}): {msg}")
            if stop_on_error:
                return None
            log.info("Could not generate %s with %s%s; trying the next option.", lab, approx, hint)
            continue
        return lab, approx, strains, full, desc, idx, f_ref, flags, assumed
    return None


def generate(event: Event, detectors: list[str], *, label: str | None = None,
             approximant: str | None = None, sample: str | int = "maxl",
             f_low: float = 20.0, sample_rate: float | None = None,
             posterior: Posterior | None = None, quiet: bool = False, yes: bool = False
             ) -> Waveform:
    """Generate the detector-projected waveform of one posterior sample.

    Tries the chosen label first. If it cannot be generated in this environment
    (and no label or approximant was requested), it tries the other labels in the file
    and then the event's other PE files, in the preference order of the module
    docstring. The waveform is tapered at its start (gwpy ``taper``) so its onset at
    ``f_low`` does not click.
    """
    _require()
    _quiet_lal()
    explicit = bool(label or approximant)
    post = posterior or open_posterior(event, quiet=quiet, yes=yes)
    if sample_rate is None:
        sample_rate = 4096.0 if event.source_class in ("BBH", "unknown") else 8192.0
    chosen, reason = choose_label(post, label)
    chosen_approx = approximant or post.approximants.get(chosen)
    errors: list[str] = []

    def ranked(p: Posterior, first: str | None = None) -> list[str]:
        rest = sorted((lab for lab in p.labels if p.approximants.get(lab) and lab != first),
                      key=lambda lb: rank_approximant(p.approximants[lb]))
        return ([first] if first else []) + rest

    order = [chosen] if explicit else ranked(post, chosen)
    got = _try_labels(post, event, detectors, order, approximant, sample, f_low, sample_rate,
                      errors, explicit)
    if got is None and not explicit and posterior is None:
        known = lalsim_approximants()
        others = sorted(
            {pe.data_url: pe for pe in event.pe if pe.data_url != post.url}.values(),
            key=lambda pe: rank_approximant(approximant_from_label(pe.label, known)[0]),
        )
        for pe in others:
            try:
                alt = open_posterior(event, pe, quiet=quiet, yes=yes)
            except GwsonifyError as exc:
                errors.append(f"{pe.label}: {exc}")
                continue
            got = _try_labels(alt, event, detectors, ranked(alt), None, sample, f_low,
                              sample_rate, errors, False)
            if got is not None:
                post = alt
                break
    if got is None:
        raise GwsonifyError("Waveform generation failed: " + "; ".join(errors))
    lab, approx, strains, full, desc, idx, f_ref, flags, assumed = got
    if lab != chosen:
        reason = (f"{reason}; {chosen} ({chosen_approx}) cannot be "
                  f"generated in this installation, so used {lab} ({approx})")
    log.info("Waveform: %s, %s, %s (f_low=%g Hz, f_ref=%g Hz).", lab, approx, desc,
             f_low, f_ref)
    return Waveform(strains, lab, approx, reason, full, desc, idx, f_low, f_ref,
                    float(sample_rate), flags, assumed, post)
