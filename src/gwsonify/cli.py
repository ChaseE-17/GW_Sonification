"""Command-line interface: ``gwsonify``.

A thin layer over :mod:`gwsonify.pipeline` and :mod:`gwsonify.catalog`. Messages go
to stderr and results go to stdout (the written file paths, or JSON with ``--json``).
Exit codes: 0 success, 1 error, 2 usage error, 3 missing optional dependency,
130 interrupted.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
import warnings

import gwsonify
from gwsonify import GwsonifyError, MissingDependencyError

log = logging.getLogger("gwsonify")

COMMANDS = ("data", "model", "info", "list", "cache")

MAIN_EPILOG = """\
examples:
  gwsonify GW150914                    hear the whitened H1 and L1 data (same as `data`)
  gwsonify GW150914 --fshift 400       ... shifted up 400 Hz, like the GWOSC audio files
  gwsonify GW150914 --template --video add the whitened model and an MP4 for teaching
  gwsonify model GW170817 --speed 0.5  the waveform model at half speed
  gwsonify info GW190521               what GWOSC has for an event
  gwsonify list GW2301*                search event names

Outputs go to ./gwsonify-output/<EVENT>/ (set -o or $GWSONIFY_OUTDIR). Downloads are
cached (see `gwsonify cache`). Run `gwsonify <command> --help` for all options.
"""

DATA_EPILOG = """\
examples:
  gwsonify data GW150914                         whitened H1+L1, default window and band
  gwsonify data GW150914 -d L1 --stereo          ...one detector, stereo file
  gwsonify data GW170817 --window -30 2          last 30 s before the BNS merger
  gwsonify data GW150914 --mode raw              the raw strain (mostly seismic rumble)
  gwsonify data GW150914 --template              + whitened waveform model overlay
  gwsonify data --gps 1126259462.4 -d H1,L1      any GPS time (window -8..+2 s)

Defaults: whitened + bandpassed, window from the chirp's time in band, glitch gating
on in whiten mode (--no-gate to disable). See the README for how defaults are chosen.
"""

MODEL_EPILOG = """\
examples:
  gwsonify model GW150914                        max-likelihood waveform in H1, L1
  gwsonify model GW150914 --label C01:SEOBNRv4PHM --sample median
  gwsonify model GW170817 --speed 0.25 --video   a slowed-down BNS chirp with video

The posterior label defaults to the GWOSC-preferred analysis; mixed-model posteriors
fall back to a single-model posterior in the same file (see `gwsonify info EVENT`).
"""


class _Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def _dets(text: str) -> list[str]:
    return [d.strip().upper() for d in text.replace(" ", ",").split(",") if d.strip()]


def _audio_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("audio (independent operations; see README 'Pitch and time')")
    g.add_argument("--speed", type=float, default=1.0, metavar="X",
                   help="playback speed: duration / X and frequencies * X (like tape)")
    g.add_argument("--stretch", type=float, default=1.0, metavar="X",
                   help="duration * X, pitch unchanged (phase vocoder)")
    g.add_argument("--pitch", type=float, default=0.0, metavar="SEMITONES",
                   help="pitch shift, duration unchanged (12 = one octave)")
    g.add_argument("--fshift", type=float, default=0.0, metavar="HZ",
                   help="add HZ to every frequency (GWOSC audio uses 400)")
    g.add_argument("--fade", type=float, default=0.05, metavar="SEC", help="fade in/out length")
    g.add_argument("--norm", choices=("peak", "fixed"), default="peak",
                   help="peak: loudest sample at --level; fixed: a fixed physical amplitude "
                        "maps to --level so loudness is comparable across events")
    g.add_argument("--level", type=float, default=None, metavar="DBFS",
                   help="target level (default -3 for peak, -30 for fixed)")
    o = p.add_argument_group("output")
    o.add_argument("-d", "--detectors", type=_dets, action="extend", metavar="IFOS",
                   help="comma-separated detectors, e.g. H1,L1 (default: all available)")
    o.add_argument("-o", "--outdir", metavar="DIR",
                   help="output directory; files go to DIR/<EVENT>/ (default: "
                        "$GWSONIFY_OUTDIR or ./gwsonify-output)")
    o.add_argument("--stereo", action="store_true",
                   help="one stereo file with detectors panned left to right")
    o.add_argument("--video", action="store_true", help="also write an MP4 (needs ffmpeg)")
    o.add_argument("--no-plots", dest="plots", action="store_false", help="skip figures")
    o.add_argument("--json", action="store_true", help="print a JSON summary to stdout")
    o.add_argument("-y", "--yes", action="store_true",
                   help="allow downloads above the size limit ($GWSONIFY_MAX_DOWNLOAD_MB)")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="debug output, library warnings and tracebacks")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gwsonify", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hear gravitational waves: sonify GWOSC events and detector data.",
        epilog=MAIN_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"gwsonify {gwsonify.__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    d = sub.add_parser("data", help="sonify detector strain (default command)",
                       description="Sonify GWOSC detector strain around an event or GPS time.",
                       epilog=DATA_EPILOG, formatter_class=_Formatter)
    d.add_argument("event", nargs="?", help="event name, e.g. GW150914, GW190521_030229, "
                                            "GW150914-v3")
    d.add_argument("--gps", type=float, help="GPS time to centre on instead of an event")
    s = d.add_argument_group("signal processing")
    s.add_argument("--mode", choices=("whiten", "bandpass", "raw"), default="whiten",
                   help="whiten = whiten + bandpass; bandpass = bandpass + mains notches")
    s.add_argument("--window", type=float, nargs=2, metavar=("START", "END"),
                   help="seconds relative to merger (default: from the chirp duration)")
    s.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   help="bandpass edges in Hz (default: 20 Hz to a mass-dependent 300-1000 Hz)")
    s.add_argument("--sample-rate", type=int, choices=(4096, 16384), default=4096,
                   help="GWOSC strain sample rate")
    s.add_argument("--no-gate", dest="gate", action="store_false", default=None,
                   help="do not gate loud glitches (gating is on in whiten mode)")
    s.add_argument("--template", action="store_true",
                   help="add the whitened waveform model (needs gwsonify[model])")
    _audio_args(d)
    _common(d)

    m = sub.add_parser("model", help="sonify the waveform model of a posterior sample",
                       description="Sonify the detector-projected waveform model from the "
                                   "event's GWOSC parameter-estimation release.",
                       epilog=MODEL_EPILOG, formatter_class=_Formatter)
    m.add_argument("event", help="event name")
    w = m.add_argument_group("waveform")
    w.add_argument("--label", help="posterior label in the PE file (see `gwsonify info`)")
    w.add_argument("--approximant", help="waveform model (default: the label's)")
    w.add_argument("--sample", default="maxl",
                   help="maxl, maxp, median (closest sample), or a sample index")
    w.add_argument("--f-low", type=float, default=20.0, metavar="HZ",
                   help="waveform start frequency")
    w.add_argument("--sample-rate", type=float, default=None, metavar="HZ",
                   help="waveform sample rate (default 4096; 8192 if a neutron star may be present)")
    _audio_args(m)
    _common(m)

    i = sub.add_parser("info", help="show what GWOSC has for an event",
                       formatter_class=_Formatter)
    i.add_argument("event")
    i.add_argument("--json", action="store_true", help="machine-readable output")
    _common(i)

    ls = sub.add_parser("list", help="list or search GWOSC events", formatter_class=_Formatter,
                        epilog="examples:\n  gwsonify list\n  gwsonify list 'GW19*'\n"
                               "  gwsonify list --catalog GWTC-4.0 --json")
    ls.add_argument("pattern", nargs="?", help="substring or glob, e.g. GW1908 or 'GW19*'")
    ls.add_argument("--catalog", help="only this catalog, e.g. GWTC-3-confident")
    ls.add_argument("--json", action="store_true", help="machine-readable output")
    _common(ls)

    c = sub.add_parser("cache", help="show or clear the download cache",
                       formatter_class=_Formatter)
    c.add_argument("--clear", action="store_true", help="delete all cached downloads")
    _common(c)
    return parser


def _configure_logging(quiet: bool, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(level)
    log.propagate = False
    if not verbose:
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("default", category=gwsonify.GwsonifyWarning)


def _print_files(result, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.summary(), indent=2))
        return
    for f in result.files:
        print(f)
    for note in result.notes:
        log.info("note: %s", note)


def _cmd_info(args) -> None:
    from gwsonify.catalog import get_event

    ev = get_event(args.event)
    info = ev.summary()
    try:
        info["strain_detectors"] = ev.detectors(4096)
    except GwsonifyError:
        info["strain_detectors"] = []
    if args.json:
        print(json.dumps(info, indent=2))
        return
    p = ev.parameters
    rows = [
        ("event", f"{ev.name}  (version {ev.version}, {ev.catalog})"),
        ("time", f"GPS {ev.gps}  ·  {ev.utc}"),
        ("source", ev.source_class),
    ]
    for key, lab in (("mass_1_source", "m1 [Msun]"), ("mass_2_source", "m2 [Msun]"),
                     ("chirp_mass_source", "chirp mass [Msun]"), ("chi_eff", "chi_eff"),
                     ("luminosity_distance", "distance [Mpc]"), ("redshift", "redshift"),
                     ("network_matched_filter_snr", "network SNR")):
        if key in p:
            rows.append((lab, f"{p[key]:g}"))
    rows.append(("strain data", ", ".join(info["strain_detectors"]) or "none published"))
    if ev.pe:
        for pe in ev.pe:
            rows.append(("PE analysis", f"{pe.label}{'  (preferred)' if pe.preferred else ''}"))
    else:
        rows.append(("PE analysis", "none (the `model` command is unavailable)"))
    if ev.other_versions:
        rows.append(("other versions", ", ".join(f"-v{v} {c}" for v, c in ev.other_versions)))
    rows += [("DOI", ev.doi or "-"), ("GWOSC page", ev.landing_page)]
    width = max(len(r[0]) for r in rows)
    for key, value in rows:
        print(f"{key:>{width}}  {value}")


def _cmd_list(args) -> None:
    from gwsonify.catalog import list_events

    events = list_events(args.catalog)
    if args.pattern:
        pat = args.pattern.upper()
        glob = any(ch in pat for ch in "*?[")

        def ok(e):
            names = [e["name"], *e.get("aliases", [])]
            return any(fnmatch.fnmatch(n.upper(), pat) if glob else pat in n.upper()
                       for n in names)

        events = [e for e in events if ok(e)]
    events.sort(key=lambda e: e["name"])
    if args.json:
        print(json.dumps(events, indent=2))
        return
    for e in events:
        extra = e.get("catalogs") or [e.get("catalog", "")]
        print(f"{e['name']:<18} {', '.join(x for x in extra if x)}")
    log.info("%d events", len(events))


def _cmd_cache(args) -> None:
    from gwsonify import cache

    if args.clear:
        freed = cache.clear()
        log.info("Cleared %.1f MB from %s", freed / 1e6, cache.cache_dir())
        return
    root, n, size = cache.usage()
    print(root)
    log.info("%d files, %.1f MB (set $GWSONIFY_CACHE_DIR to move it; --clear to empty)",
             n, size / 1e6)


def _normalize_argv(argv: list[str]) -> list[str]:
    """``gwsonify EVENT ...`` is shorthand for ``gwsonify data EVENT ...``."""
    if argv and argv[0] not in (*COMMANDS, "-h", "--help", "--version"):
        return ["data", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    argv = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    if not argv:
        parser.print_help(sys.stderr)
        return 2
    args = parser.parse_args(argv)
    _configure_logging(args.quiet, args.verbose)
    try:
        if args.command == "data":
            if not args.event and args.gps is None:
                parser.parse_args(["data", "--help"])
            result = gwsonify.sonify_data(
                args.event, gps=args.gps, detectors=args.detectors, mode=args.mode,
                window=args.window, band=args.band, sample_rate=args.sample_rate,
                gate=args.gate, template=args.template, speed=args.speed,
                stretch=args.stretch, pitch=args.pitch, fshift=args.fshift, fade=args.fade,
                norm=args.norm, level=args.level, stereo=args.stereo, video=args.video,
                plots=args.plots, outdir=args.outdir, quiet=args.quiet, yes=args.yes)
            _print_files(result, args.json)
        elif args.command == "model":
            sample = int(args.sample) if args.sample.isdigit() else args.sample
            result = gwsonify.sonify_model(
                args.event, detectors=args.detectors, label=args.label,
                approximant=args.approximant, sample=sample, f_low=args.f_low,
                sample_rate=args.sample_rate, speed=args.speed, stretch=args.stretch,
                pitch=args.pitch, fshift=args.fshift, fade=args.fade, norm=args.norm,
                level=args.level, stereo=args.stereo, video=args.video, plots=args.plots,
                outdir=args.outdir, quiet=args.quiet, yes=args.yes)
            _print_files(result, args.json)
        elif args.command == "info":
            _cmd_info(args)
        elif args.command == "list":
            _cmd_list(args)
        elif args.command == "cache":
            _cmd_cache(args)
    except MissingDependencyError as exc:
        log.error("gwsonify: %s", exc)
        return 3
    except GwsonifyError as exc:
        log.error("gwsonify: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("gwsonify: interrupted")
        return 130
    except Exception as exc:
        if args.verbose:
            raise
        log.error("gwsonify: unexpected error: %s: %s\nRerun with -v for details, and please "
                  "report it at https://github.com/ChaseE-17/GW_Sonification/issues",
                  type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
