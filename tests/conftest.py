"""Shared fixtures.

Offline by default: ``fake_api`` serves recorded GWOSC API v2 responses from
``tests/data/api`` (recorded 2026-09-30), and every test gets an empty temporary cache.
Opt in to network tests with ``--online`` and to notebook execution with ``--notebooks``.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

DATA = Path(__file__).parent / "data"


def pytest_addoption(parser):
    parser.addoption("--online", action="store_true", help="run tests that use GWOSC")
    parser.addoption("--notebooks", action="store_true", help="execute the notebooks")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "online" in item.keywords and not config.getoption("--online"):
            item.add_marker(pytest.mark.skip(reason="needs --online"))
        if "notebook" in item.keywords and not config.getoption("--notebooks"):
            item.add_marker(pytest.mark.skip(reason="needs --notebooks"))


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch, request):
    """Private cache and output dirs for every test."""
    if "online" not in request.keywords:
        monkeypatch.setenv("GWSONIFY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GWSONIFY_OUTDIR", str(tmp_path / "out"))


@pytest.fixture
def fake_api(monkeypatch):
    """Replace network JSON access with recorded fixtures; unknown URLs 404."""
    from gwsonify import cache, catalog

    calls = []

    def get_json(url, *, params=None, ttl=0):
        calls.append(url)
        path = urlparse(url).path.replace("/api/v2/", "").strip("/")
        f = DATA / "api" / (path.replace("/", "__") + ".json")
        if not f.exists():
            raise FileNotFoundError(url)
        return json.loads(f.read_text())

    monkeypatch.setattr(cache, "get_json", get_json)
    monkeypatch.setattr(catalog, "get_json", get_json)
    return calls


@pytest.fixture
def synthetic_strain():
    """Factory for coloured Gaussian noise with an optional injected chirp-like burst.

    Returns a gwpy TimeSeries (so it exercises the real conditioning code).
    """
    import numpy as np
    from gwpy.timeseries import TimeSeries
    from scipy import signal

    def make(t0=1126259462.4, span=(-14.0, 14.0), fs=4096, inject=0.0, glitch=None,
             line=None, seed=0, det="H1"):
        rng = np.random.default_rng(seed)
        n = int((span[1] - span[0]) * fs)
        # red noise: strongly coloured like real detector noise
        x = signal.lfilter([1.0], [1.0, -0.98], rng.normal(size=n))
        t = span[0] + np.arange(n) / fs
        if inject:
            # sine-Gaussian burst at 150 Hz centred on the merger time
            x += inject * np.std(x) * np.exp(-(t / 0.05) ** 2) * np.sin(2 * np.pi * 150 * t)
        if glitch is not None:
            k = int((glitch - span[0]) * fs)
            x[k : k + 20] += 1e4 * np.std(x)
        if line is not None:
            x += 5 * np.std(x) * np.sin(2 * np.pi * line * t)
        return TimeSeries(x * 1e-21, t0=t0 + span[0], sample_rate=fs, name=det)

    return make
