"""Download cache.

Large GWOSC and Zenodo files (strain frames of 40-130 MB, PE releases of up to a few
hundred MB) are downloaded once into a per-user cache directory and reused. Nothing is
downloaded silently: every download is announced on stderr with its size, and cache
hits are reported as well.

The cache location is, in order of precedence:

1. ``$GWSONIFY_CACHE_DIR``
2. ``$XDG_CACHE_HOME/gwsonify`` (Linux and others), ``~/Library/Caches/gwsonify``
   (macOS), ``%LOCALAPPDATA%\\gwsonify\\Cache`` (Windows)
3. ``~/.cache/gwsonify``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from gwsonify import GwsonifyError

log = logging.getLogger(__name__)

USER_AGENT = "gwsonify (https://github.com/ChaseE-17/GW_Sonification)"
CHUNK = 1 << 20


def max_download_bytes() -> int:
    """Downloads larger than this need confirmation (``$GWSONIFY_MAX_DOWNLOAD_MB``,
    default 1000 MB). Some recent PE releases are well over 1 GB."""
    return int(float(os.environ.get("GWSONIFY_MAX_DOWNLOAD_MB", "1000")) * 1e6)


class LargeDownloadError(GwsonifyError):
    """A download exceeds :func:`max_download_bytes` and was not confirmed."""

    def __init__(self, url: str, size: int):
        self.url, self.size = url, size
        super().__init__(
            f"{Path(urlparse(url).path.replace('/content', '')).name} is {_human(size)}, "
            f"above the {_human(max_download_bytes())} download limit. Rerun with --yes to "
            "download it (it is cached afterwards), or raise GWSONIFY_MAX_DOWNLOAD_MB."
        )


def cache_dir() -> Path:
    """Return (and create) the cache directory."""
    env = os.environ.get("GWSONIFY_CACHE_DIR")
    if env:
        path = Path(env).expanduser()
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Caches" / "gwsonify"
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        path = Path(os.environ["LOCALAPPDATA"]) / "gwsonify" / "Cache"
    else:
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        path = Path(base) / "gwsonify"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(url: str, subdir: str) -> Path:
    """Map a URL to a stable, human-readable cache path."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    # Zenodo API URLs end in ".../files/<name>/content".
    name = parts[-2] if parts and parts[-1] == "content" and len(parts) > 1 else parts[-1]
    digest = hashlib.sha1(url.encode()).hexdigest()[:10]
    return cache_dir() / subdir / f"{digest}-{name}"


def _human(nbytes: float | None) -> str:
    if not nbytes:
        return "unknown size"
    for unit in ("B", "kB", "MB", "GB"):
        if nbytes < 1000 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1000
    return f"{nbytes:.1f} GB"  # pragma: no cover


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file (recorded in provenance sidecars)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, subdir: str = "files", *, quiet: bool = False, yes: bool = False) -> Path:
    """Return a local path for ``url``, downloading it into the cache if needed.

    Downloads go to a ``.part`` file that is renamed only when complete, so an
    interrupted download never leaves a corrupt cache entry. Files larger than
    :func:`max_download_bytes` need ``yes=True`` or an interactive confirmation;
    otherwise :class:`LargeDownloadError` is raised.
    """
    import requests

    path = _cache_path(url, subdir)
    if path.exists():
        log.info("Using cached %s (%s)", path.name.split("-", 1)[1], _human(path.stat().st_size))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(
            url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0) or None
            if total and total > max_download_bytes() and not yes:
                if not (sys.stdin.isatty() and sys.stderr.isatty() and not quiet):
                    raise LargeDownloadError(url, total)
                sys.stderr.write(f"{path.name.split('-', 1)[1]} is {_human(total)}. "
                                 "Download it now? [y/N] ")
                sys.stderr.flush()
                if input().strip().lower() not in ("y", "yes"):
                    raise LargeDownloadError(url, total)
            log.warning(
                "Downloading %s (%s) to the gwsonify cache ...",
                path.name.split("-", 1)[1], _human(total),
            )
            done, t0, last = 0, time.monotonic(), 0.0
            show = not quiet and sys.stderr.isatty()
            with tmp.open("wb") as f:
                for block in resp.iter_content(CHUNK):
                    f.write(block)
                    done += len(block)
                    now = time.monotonic()
                    if show and (now - last > 0.2):
                        last = now
                        pct = f"{100 * done / total:5.1f}%" if total else ""
                        rate = done / max(now - t0, 1e-6)
                        sys.stderr.write(f"\r  {pct} {_human(done)} at {_human(rate)}/s   ")
                        sys.stderr.flush()
            if show:
                sys.stderr.write("\r" + " " * 60 + "\r")
        tmp.replace(path)
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        raise GwsonifyError(f"Download failed for {url}: {exc}") from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def get_json(url: str, *, params: dict | None = None, ttl: float = 86400.0) -> dict:
    """GET a JSON document, caching the response for ``ttl`` seconds.

    API metadata changes rarely, so a one-day cache keeps repeated commands fast and
    lets ``info`` work briefly offline. A stale cache entry is used (with a warning)
    if the network is unreachable.
    """
    import requests

    key = url + ("?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "")
    path = cache_dir() / "api" / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        return json.loads(path.read_text())
    for attempt in range(6):
        try:
            resp = requests.get(url, params=params, timeout=30,
                                headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            if path.exists():
                log.warning("GWOSC unreachable (%s); using cached metadata.",
                            exc.__class__.__name__)
                return json.loads(path.read_text())
            raise GwsonifyError(
                f"Could not reach {urlparse(url).netloc}: {exc.__class__.__name__}. "
                "Check your internet connection."
            ) from exc
        # rate limited or temporarily unavailable: back off politely and retry
        if resp.status_code not in (429, 502, 503, 504) or attempt == 5:
            break
        retry_after = resp.headers.get("Retry-After", "")
        delay = float(retry_after) if retry_after.isdigit() else min(2.0 ** attempt, 30.0)
        log.info("GWOSC asked us to slow down (HTTP %d); retrying in %.0f s.",
                 resp.status_code, delay)
        time.sleep(delay)
    if resp.status_code == 404:
        raise FileNotFoundError(url)
    if not resp.ok:
        raise GwsonifyError(f"GWOSC API error {resp.status_code} for {resp.url}")
    data = resp.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def usage() -> tuple[Path, int, int]:
    """Return ``(cache_dir, number_of_files, total_bytes)``."""
    root = cache_dir()
    files = [p for p in root.rglob("*") if p.is_file()]
    return root, len(files), sum(p.stat().st_size for p in files)


def clear() -> int:
    """Delete everything in the cache; return the number of bytes freed."""
    root, _, total = usage()
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return total
