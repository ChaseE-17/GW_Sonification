"""Provenance sidecars and citation text.

Every sonification run writes ``<stem>.provenance.json`` next to its outputs. It
contains everything needed to reproduce and cite the result: the command, software
versions, the GWOSC event version and DOI, every downloaded data URL with its SHA-256,
the processing parameters, the model sample (for waveform models), and the GWOSC
acknowledgement text.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

import gwsonify

GWOSC_ACKNOWLEDGEMENT = (
    "This research has made use of data or software obtained from the Gravitational Wave "
    "Open Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the "
    "Virgo Collaboration, and KAGRA. This material is based upon work supported by NSF's "
    "LIGO Laboratory which is a major facility fully funded by the National Science "
    "Foundation, as well as the Science and Technology Facilities Council (STFC) of the "
    "United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany "
    "for support of the construction of Advanced LIGO and construction and operation of the "
    "GEO600 detector. Additional support for Advanced LIGO was provided by the Australian "
    "Research Council. Virgo is funded, through the European Gravitational Observatory "
    "(EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian "
    "Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions "
    "by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, "
    "Portugal, Spain. KAGRA is supported by Ministry of Education, Culture, Sports, Science "
    "and Technology (MEXT), Japan Society for the Promotion of Science (JSPS) in Japan; "
    "National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea; "
    "Academia Sinica (AS) and National Science and Technology Council (NSTC) in Taiwan."
)
"""Standard GWOSC acknowledgement (https://gwosc.org/acknowledgement/)."""

GWOSC_CITATION = (
    "Please cite the GWOSC data release papers for the observing run(s) used "
    "(https://gwosc.org/acknowledgement/), the catalog paper for the event, and the "
    "event's data DOI."
)

# Data-release papers to cite, by observing run (https://gwosc.org/acknowledgement/,
# checked 2026-09-30).
_O12 = "LVC, SoftwareX 13, 100658 (2021), doi:10.1016/j.softx.2021.100658"
_O3 = "LVK, ApJS 267, 29 (2023), doi:10.3847/1538-4365/acdc9f"
RUN_PAPERS = {
    "O1": _O12, "O2": _O12, "O3a": _O3, "O3b": _O3, "O3GK": _O3,
    "O4a": "LVK, ApJ 1004, 2329 (2026), doi:10.3847/1538-4357/ae211e",
    "O4b": "LVK, arXiv:2605.27090", "O4c": "LVK, arXiv:2605.27090",
}


def data_release_paper(run: str) -> str | None:
    """Citation for the GWOSC data release of observing run ``run`` (e.g. ``O3b``)."""
    for key in sorted(RUN_PAPERS, key=len, reverse=True):
        if run.startswith(key):
            return RUN_PAPERS[key]
    return None


_PACKAGES = ("numpy", "scipy", "matplotlib", "gwpy", "gwosc", "pesummary", "lalsuite",
             "python-lalsimulation", "h5py")


def software_versions() -> dict[str, str]:
    """Installed versions of gwsonify and its scientific dependencies."""
    out = {"gwsonify": gwsonify.__version__, "python": platform.python_version()}
    for pkg in _PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    try:
        import lalsimulation

        out["lalsimulation"] = lalsimulation.__version__
    except Exception:
        pass
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build(kind: str, *, event=None, inputs: dict, processing: dict, outputs: list,
          data_files: list | None = None, model: dict | None = None,
          notes: list | None = None) -> dict:
    """Assemble a provenance record (plain JSON-serialisable dict)."""
    from gwsonify.cache import sha256

    record = {
        "gwsonify_provenance_version": 1,
        "kind": kind,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv) if Path(sys.argv[0]).name.startswith("gwsonify") else None,
        "software": software_versions(),
        "platform": platform.platform(),
        "inputs": inputs,
        "processing": processing,
        "outputs": [Path(p).name for p in outputs],
        "data_files": [
            {"url": url, "sha256": sha256(path) if path and Path(path).exists() else None}
            for url, path in (data_files or [])
        ],
        "notes": notes or [],
        "citation": {
            "acknowledgement": GWOSC_ACKNOWLEDGEMENT,
            "how_to_cite": GWOSC_CITATION,
            "gwsonify": "See CITATION.cff in the gwsonify repository.",
        },
    }
    if event is not None:
        record["event"] = event.summary()
        record["citation"]["event_doi"] = event.doi
        record["citation"]["data_release_paper"] = data_release_paper(event.run)
    if model is not None:
        record["model"] = model
    return _jsonable(record)


def write(path: Path, record: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    return path
