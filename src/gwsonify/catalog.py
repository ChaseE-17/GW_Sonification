"""GWOSC event catalog access (GWOSC API v2, https://gwosc.org/api/v2/).

Everything the rest of the package needs to know about an event comes from here:
the canonical name, the GPS merger time, the catalog version, source parameters,
the parameter-estimation (PE) release files with their posterior labels, and the
strain files that exist for each detector. Nothing is hard-coded: labels,
detectors and GPS times are all read from the API (and, for labels, re-checked
against the PE file itself in :mod:`gwsonify.model`).

Event specs
-----------
``get_event`` accepts the forms the API understands:

* short or long names and aliases: ``GW150914``, ``GW190521``, ``GW190521_030229``
* an explicit version: ``GW150914-v3``
* a catalog: ``GW150914@GWTC-1-confident``

Without an explicit version, the newest version from a *confident* catalog is used
(marginal and auxiliary catalogs only if nothing else exists). See
:func:`choose_version`.
"""

from __future__ import annotations

import difflib
import logging
import math
import re
from dataclasses import dataclass, field

from gwsonify import GwsonifyError
from gwsonify.cache import get_json

log = logging.getLogger(__name__)

API = "https://gwosc.org/api/v2"

# Catalogs whose versions are only used when no confident version exists.
_LOW_PRIORITY = re.compile(r"marginal|auxiliary|preliminary|IAS", re.IGNORECASE)

_G = 6.67430e-11
_C = 299792458.0
_MSUN = 1.988409870698051e30
_T_SUN = _G * _MSUN / _C**3  # solar mass in seconds


@dataclass(frozen=True)
class PEAnalysis:
    """One parameter-estimation analysis listed by GWOSC for an event version."""

    name: str
    label: str
    data_url: str
    preferred: bool
    pipeline: str = ""
    links: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrainFile:
    """A GWOSC strain file (HDF5) for one detector."""

    detector: str
    sample_rate: int
    gps_start: float
    duration: float
    url: str
    rank: int = 0  # 0 = the chosen event version; larger = less preferred source

    @property
    def gps_end(self) -> float:
        return self.gps_start + self.duration


@dataclass
class Event:
    """A GWOSC event version and the metadata gwsonify uses."""

    name: str
    version: int
    catalog: str
    gps: float
    run: str
    doi: str = ""
    aliases: tuple[str, ...] = ()
    listed_detectors: tuple[str, ...] = ()
    parameters: dict[str, float] = field(default_factory=dict)
    pe: tuple[PEAnalysis, ...] = ()
    other_versions: tuple[tuple[int, str], ...] = ()

    # -- identity -------------------------------------------------------------------
    @property
    def spec(self) -> str:
        """Unambiguous identifier, for example ``GW150914-v4``."""
        return f"{self.name}-v{self.version}"

    @property
    def url(self) -> str:
        return f"{API}/event-versions/{self.spec}"

    @property
    def landing_page(self) -> str:
        return f"https://gwosc.org/eventapi/html/{self.catalog}/{self.name}/v{self.version}/"

    @property
    def utc(self) -> str:
        from gwpy.time import from_gps

        return from_gps(self.gps).strftime("%Y-%m-%d %H:%M:%S UTC")

    # -- source properties ----------------------------------------------------------
    def _detector_frame(self, key: str) -> float | None:
        """Detector-frame value (source value x (1+z)); falls back to the source-frame
        value when no redshift is listed (adequate for choosing defaults)."""
        if key in self.parameters:
            return self.parameters[key]
        value = self.parameters.get(f"{key}_source")
        if value is None:
            return None
        return value * (1 + self.parameters.get("redshift", 0.0))

    @property
    def chirp_mass_det(self) -> float | None:
        """Detector-frame chirp mass in solar masses (from the API summary values)."""
        mc = self._detector_frame("chirp_mass")
        if mc is None:
            m1, m2 = self._detector_frame("mass_1"), self._detector_frame("mass_2")
            if m1 and m2:
                mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        return mc

    @property
    def total_mass_det(self) -> float | None:
        mt = self._detector_frame("total_mass")
        if mt is None:
            m1, m2 = self._detector_frame("mass_1"), self._detector_frame("mass_2")
            mt = m1 + m2 if m1 and m2 else None
        return mt

    @property
    def source_class(self) -> str:
        """``BNS``, ``NSBH``, ``BBH`` or ``unknown``, from median source-frame masses.

        Uses the conventional 3 solar-mass boundary between neutron stars and black
        holes. This is only used to pick defaults, never as a scientific claim.
        """
        m1, m2 = self.parameters.get("mass_1_source"), self.parameters.get("mass_2_source")
        if m1 is None or m2 is None:
            return "unknown"
        n_ns = sum(m < 3.0 for m in (m1, m2))
        return {0: "BBH", 1: "NSBH", 2: "BNS"}[n_ns]

    def time_to_merger(self, f_low: float) -> float | None:
        """Leading-order (Newtonian) time from GW frequency ``f_low`` to merger, in s.

        tau = (5/256) (pi f)^(-8/3) (G Mc / c^3)^(-5/3)   (e.g. Maggiore 2008, eq. 4.21)
        """
        mc = self.chirp_mass_det
        if not mc:
            return None
        return 5.0 / 256.0 * (math.pi * f_low) ** (-8 / 3) * (mc * _T_SUN) ** (-5 / 3)

    def f_isco(self) -> float | None:
        """GW frequency at the Schwarzschild ISCO of the total mass: c^3 / (6^1.5 pi G M)."""
        mt = self.total_mass_det
        return 1.0 / (6**1.5 * math.pi * mt * _T_SUN) if mt else None

    # -- data products --------------------------------------------------------------
    def preferred_pe(self) -> PEAnalysis | None:
        for pe in self.pe:
            if pe.preferred:
                return pe
        return self.pe[0] if self.pe else None

    def strain_files(self, sample_rate: int = 4096) -> list[StrainFile]:
        """All HDF5 strain files GWOSC lists for this event at ``sample_rate``.

        Combines the event-version files (32 s and 4096 s; published for older catalogs,
        possibly under another version of the same event, since the strain is shared)
        with the bulk 4096 s files that exist for every event.
        """
        khz = {4096: 4, 16384: 16}[int(sample_rate)]
        files: list[StrainFile] = []
        # chosen version first, then newer releases before older ones, bulk files last
        # (the order becomes StrainFile.rank, used by strain.pick_file)
        versions = [self.version, *sorted((v for v, _ in self.other_versions), reverse=True)]
        for rank, version in enumerate(versions):
            url = f"{API}/event-versions/{self.name}-v{version}/strain-files"
            for item in _paged(url):
                if item.get("file_format") == "HDF" and item.get("sample_rate_kHz") == khz:
                    files.append(StrainFile(item["detector"], int(sample_rate),
                                            float(item["gps_start"]), float(item["duration"]),
                                            item["download_url"], rank))
        for item in _paged(f"{API}/events/{self.name}/strain-files"):
            if item.get("sample_rate_kHz") == khz and item.get("hdf5_url"):
                files.append(StrainFile(item["detector"], int(sample_rate),
                                        float(item["gps_start"]), 4096.0, item["hdf5_url"],
                                        len(versions)))
        return files

    def detectors(self, sample_rate: int = 4096) -> list[str]:
        """Detectors with public strain for this event (not just those listed)."""
        return sorted({f.detector for f in self.strain_files(sample_rate)}, key=_ifo_order)

    def summary(self) -> dict:
        """JSON-serialisable description (used by ``gwsonify info --json``)."""
        pe = self.preferred_pe()
        return {
            "name": self.name,
            "version": self.version,
            "spec": self.spec,
            "catalog": self.catalog,
            "gps": self.gps,
            "utc": self.utc,
            "run": self.run,
            "doi": self.doi,
            "aliases": list(self.aliases),
            "source_class": self.source_class,
            "parameters": self.parameters,
            "listed_detectors": list(self.listed_detectors),
            "pe_analyses": [
                {"label": p.label, "preferred": p.preferred, "url": p.data_url} for p in self.pe
            ],
            "preferred_pe_label": pe.label if pe else None,
            "other_versions": [f"{self.name}-v{v} ({c})" for v, c in self.other_versions],
            "gwosc_url": self.landing_page,
        }


def _ifo_order(ifo: str) -> tuple[int, str]:
    order = ["H1", "L1", "V1", "K1", "G1"]
    return (order.index(ifo) if ifo in order else len(order), ifo)


def _paged(url: str, params: dict | None = None) -> list[dict]:
    params = dict(params or {}, pagesize=500)
    out: list[dict] = []
    while url:
        try:
            data = get_json(url, params=params)
        except FileNotFoundError:
            return out
        out.extend(data.get("results", []))
        url, params = data.get("next"), None
    return out


# ---------------------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------------------

_SPEC = re.compile(r"^(?P<name>[^@]+?)(?:-v(?P<version>\d+))?(?:@(?P<catalog>.+))?$")


def list_events(catalog: str | None = None) -> list[dict]:
    """Return GWOSC events as dicts with ``name``, ``aliases`` and ``catalogs``.

    With ``catalog``, only that catalog's events (``gps``, ``detectors`` included).
    """
    if catalog:
        try:
            get_json(f"{API}/catalogs/{catalog}")
        except FileNotFoundError:
            names = [c["name"] for c in _paged(f"{API}/catalogs")]
            raise GwsonifyError(
                f"Unknown catalog {catalog!r}. Available: {', '.join(names)}"
            ) from None
        return [
            {"name": e["name"], "version": e["version"], "catalog": e["catalog"],
             "gps": e["gps"], "detectors": e.get("detectors", [])}
            for e in _paged(f"{API}/catalogs/{catalog}/events")
        ]
    return [
        {"name": e["name"], "aliases": e.get("aliases", []),
         "catalogs": sorted({v["catalog"] for v in e.get("versions", [])})}
        for e in _paged(f"{API}/events")
    ]


def suggest(name: str, limit: int = 5) -> list[str]:
    """Event names close to ``name`` (prefix matches first, then fuzzy matches)."""
    names: set[str] = set()
    for e in list_events():
        names.add(e["name"])
        names.update(e["aliases"])
    upper = name.upper()
    prefix = sorted(n for n in names if n.upper().startswith(upper))
    fuzzy = difflib.get_close_matches(upper, sorted(names), n=limit, cutoff=0.75)
    out = list(dict.fromkeys(prefix + fuzzy))
    return out[:limit]


def choose_version(versions: list[dict]) -> dict:
    """Pick the default version: newest from a confident catalog, else newest overall."""
    good = [v for v in versions if not _LOW_PRIORITY.search(v["catalog"])]
    return max(good or versions, key=lambda v: v["version"])


def get_event(spec: str) -> Event:
    """Resolve an event spec (see module docstring) into an :class:`Event`.

    Raises :class:`~gwsonify.GwsonifyError` with suggestions if the name is unknown.
    """
    m = _SPEC.match(spec.strip())
    if not m:
        raise GwsonifyError(f"Could not parse event name {spec!r}")
    name, version, catalog = m["name"].strip(), m["version"], m["catalog"]
    try:
        ev = get_json(f"{API}/events/{name}")
    except FileNotFoundError:
        hints = suggest(name)
        hint = f" Did you mean: {', '.join(hints)}?" if hints else ""
        raise GwsonifyError(
            f"No GWOSC event named {name!r}.{hint} Run `gwsonify list` to browse events."
        ) from None

    versions = ev.get("versions", [])
    if not versions:
        raise GwsonifyError(f"GWOSC lists no releases for {ev['name']}.")
    if version is not None:
        chosen = [v for v in versions if v["version"] == int(version)]
    elif catalog is not None:
        chosen = [v for v in versions if v["catalog"].lower() == catalog.lower()]
        chosen = [max(chosen, key=lambda v: v["version"])] if chosen else []
    else:
        chosen = [choose_version(versions)]
    if not chosen:
        avail = ", ".join(f"v{v['version']} ({v['catalog']})" for v in versions)
        raise GwsonifyError(f"{ev['name']} has no version matching {spec!r}. Available: {avail}")
    v = chosen[0]

    detail = get_json(v["detail_url"])
    params = _paged(f"{API}/event-versions/{ev['name']}-v{v['version']}/parameters")
    pe_entries = [
        p for p in params
        if p.get("pipeline_type") == "pe" and p.get("data_url")
        and p.get("waveform_family") not in (None, "", "na")
    ]
    pe = tuple(
        PEAnalysis(
            name=p["name"], label=p["waveform_family"], data_url=p["data_url"],
            preferred=bool(p.get("is_preferred")), pipeline=p.get("pipeline", ""),
            links=tuple(link["url"] for link in p.get("links", [])),
        )
        for p in pe_entries
    )
    summary = next((p for p in pe_entries if p.get("is_preferred")), None) or next(
        (p for p in params if p.get("is_preferred")), None
    )
    values = {
        q["name"]: q["best"] for q in (summary or {}).get("parameters", [])
        if q.get("best") is not None
    }
    event = Event(
        name=ev["name"],
        version=v["version"],
        catalog=v["catalog"],
        gps=float(detail["gps"]),
        run=detail.get("run", ""),
        doi=detail.get("doi") or "",
        aliases=tuple(ev.get("aliases", [])),
        listed_detectors=tuple(detail.get("detectors", [])),
        parameters=values,
        pe=pe,
        other_versions=tuple(
            (o["version"], o["catalog"]) for o in versions if o["version"] != v["version"]
        ),
    )
    if name.upper() != event.name.upper() and "_" not in name:
        log.info("Resolved %s to %s (%s).", name, event.name, event.catalog)
    return event
