#!/usr/bin/env python
"""GWOSC compatibility sweep (needs the network; writes docs/sweep-results.md).

Stage A  ``info`` for every GWOSC event: resolve, choose a version, list PE analyses and
         strain detectors.
Stage B  ``data`` for a stratified sample: up to ``--per-catalog`` events from every
         catalog (the first, middle and last by GPS time).
Stage C  ``model`` for a fixed list covering BBH/BNS/NSBH across GWTC-1 ... GWTC-5.

Every outcome is classified as OK, FAIL (a clear GwsonifyError message, the expected way
to fail) or BUG (any other exception). Usage::

    python scripts/sweep.py                    # all stages
    python scripts/sweep.py --stages A         # metadata only (fast)
    python scripts/sweep.py --per-catalog 1 --stages AB
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import gwsonify
from gwsonify import GwsonifyError, catalog

MODEL_SAMPLE = [
    "GW150914",            # GWTC-2.1 mixed file, O1
    "GW151226",            # GWTC-2.1, low-mass BBH
    "GW170817",            # GWTC-1 format, BNS, tidal model
    "GW190425",            # BNS, GWTC-2.1
    "GW190521",            # heavy BBH
    "GW200105_162426",     # NSBH; default version has no PE -> falls back to one that does
    "GW200115_042309",     # NSBH, GWTC-3
    "GW230529_181500",     # NSBH, GWTC-4.x
    "GW231123_135430",     # O4 discovery, very heavy BBH
    "GW240109_050431",     # GWTC-4.1, single-detector
    "GW250114_082203-v1",  # O4 discovery, one file per approximant, NRSur preferred
    "GW250114_082203",     # GWTC-5.0: 1.7 GB file -> refused with alternatives
]


def classify(fn):
    t0 = time.monotonic()
    try:
        detail = fn()
        return {"status": "OK", "detail": detail, "seconds": round(time.monotonic() - t0, 1)}
    except GwsonifyError as exc:
        return {"status": "FAIL", "detail": str(exc), "seconds": round(time.monotonic() - t0, 1)}
    except Exception as exc:
        return {"status": "BUG", "detail": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=6), "seconds": round(time.monotonic() - t0, 1)}


def stage_a(names):
    def one(name):
        def run():
            ev = catalog.get_event(name)
            pe = ev.preferred_pe()
            return {"spec": ev.spec, "catalog": ev.catalog, "class": ev.source_class,
                    "pe": pe.label if pe else None, "strain": ev.detectors(4096)}
        return name, classify(run)

    with cf.ThreadPoolExecutor(3) as pool:  # be gentle with the GWOSC API
        return dict(pool.map(one, names))


def stage_b(events_by_catalog, per_catalog, outdir):
    picks = []
    for cat, unsorted in sorted(events_by_catalog.items()):
        evs = sorted(unsorted, key=lambda e: e["gps"])
        idx = sorted({0, len(evs) // 2, len(evs) - 1})[:per_catalog]
        picks += [(cat, evs[i]["name"], evs[i]["version"]) for i in idx]
    out = {}
    for cat, name, version in picks:
        spec = f"{name}-v{version}"
        print(f"[B] {spec} ({cat})", file=sys.stderr, flush=True)

        def run(spec=spec):
            r = gwsonify.sonify_data(spec, plots=False, outdir=outdir, quiet=True)
            return {"detectors": list(r.series), "seconds_audio": round(
                next(iter(r.audio.values())).shape[0] / r.rate, 1), "notes": r.notes}

        out[spec] = {"catalog": cat, **classify(run)}
    return out


def stage_c(outdir):
    out = {}
    for name in MODEL_SAMPLE:
        print(f"[C] {name}", file=sys.stderr, flush=True)

        def run(name=name):
            r = gwsonify.sonify_model(name, plots=False, outdir=outdir, quiet=True)
            m = r.provenance["model"]
            return {"label": m["label"], "approximant": m["approximant"],
                    "sample": m["sample"], "choice": m["label_choice"],
                    "assumed": sorted(m["assumed_parameters"])}

        out[name] = classify(run)
    return out


def table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="ABC")
    ap.add_argument("--per-catalog", type=int, default=2)
    ap.add_argument("--out", default="docs/sweep-results.md")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.getLogger("gwsonify").setLevel(logging.ERROR)
    outdir = Path(tempfile.mkdtemp(prefix="gwsonify-sweep-"))

    results = {"date": dt.date.today().isoformat(), "gwsonify": gwsonify.__version__}
    all_events = catalog.list_events()
    names = sorted(e["name"] for e in all_events)
    md = [f"# GWOSC compatibility sweep\n\nRun {results['date']} with gwsonify "
          f"{results['gwsonify']} against the live GWOSC API v2 ({len(names)} events). "
          "Generated by `scripts/sweep.py`.\n\nStatus: **OK** = worked; **FAIL** = stopped "
          "with a clear message (expected for unavailable data); **BUG** = unexpected "
          "exception.\n"]

    if "A" in args.stages:
        a = stage_a(names)
        results["A"] = a
        counts = {s: sum(r["status"] == s for r in a.values()) for s in ("OK", "FAIL", "BUG")}
        no_pe = sorted(n for n, r in a.items() if r["status"] == "OK" and not r["detail"]["pe"])
        no_strain = sorted(n for n, r in a.items() if r["status"] == "OK" and not r["detail"]["strain"])
        cats = {}
        for r in a.values():
            if r["status"] == "OK":
                cats.setdefault(r["detail"]["catalog"], []).append(r)
        md.append(f"## Stage A: metadata for every event\n\n{counts['OK']} OK, {counts['FAIL']} "
                  f"FAIL, {counts['BUG']} BUG.\n")
        md.append(table(
            [(c, len(rs), sum(bool(r["detail"]["pe"]) for r in rs),
              sum(bool(r["detail"]["strain"]) for r in rs),
              ", ".join(sorted({r["detail"]["pe"] or "-" for r in rs}))[:120])
             for c, rs in sorted(cats.items())],
            ["default catalog version", "events", "with PE release", "with strain", "preferred labels seen"]))
        md.append(f"\nEvents without a PE release in their default version ({len(no_pe)}): "
                  f"{', '.join(no_pe) or 'none'}.\n")
        md.append(f"Events without published strain ({len(no_strain)}): {', '.join(no_strain) or 'none'}.\n")
        bad = [(n, r["status"], r["detail"]) for n, r in a.items() if r["status"] != "OK"]
        if bad:
            md.append(table(bad, ["event", "status", "message"]))

    if "B" in args.stages:
        by_cat = {}
        for c in [c["name"] for c in catalog._paged(f"{catalog.API}/catalogs")]:
            try:
                by_cat[c] = catalog.list_events(c)
            except GwsonifyError:
                continue
        b = stage_b({k: v for k, v in by_cat.items() if v}, args.per_catalog, outdir)
        results["B"] = b
        md.append("\n## Stage B: `data` on a stratified sample\n")
        md.append(table([(k, v["catalog"], v["status"], v["seconds"],
                          json.dumps(v["detail"]) if v["status"] == "OK" else v["detail"])
                         for k, v in b.items()], ["event", "catalog", "status", "s", "result"]))

    if "C" in args.stages:
        c = stage_c(outdir)
        results["C"] = c
        md.append("\n## Stage C: `model` on BBH/BNS/NSBH across catalogs\n")
        md.append(table([(k, v["status"], v["seconds"],
                          (f"{v['detail']['label']} / {v['detail']['approximant']} / "
                           f"{v['detail']['sample']}; assumed: {v['detail']['assumed'] or '-'}")
                          if v["status"] == "OK" else v["detail"])
                         for k, v in c.items()], ["event", "status", "s", "result"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=1, default=str))
    bugs = sum(r["status"] == "BUG" for s in "ABC" for r in results.get(s, {}).values())
    print(f"wrote {args.out}; {bugs} BUG outcomes", file=sys.stderr)
    return 1 if bugs else 0


if __name__ == "__main__":
    sys.exit(main())
