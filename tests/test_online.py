"""Real GWOSC round trips (``pytest --online``). Downloads ~10 MB into a temp cache."""

import numpy as np
import pytest

import gwsonify

pytestmark = pytest.mark.online


@pytest.fixture(autouse=True)
def _cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GWSONIFY_CACHE_DIR", str(tmp_path_factory.getbasetemp() / "cache"))


def test_gw150914_chirp_is_loudest_near_merger(tmp_path):
    r = gwsonify.sonify_data("GW150914", outdir=tmp_path, plots=False)
    assert set(r.series) == {"H1", "L1"}
    for det, ts in r.series.items():
        t = ts.times.value[np.argmax(np.abs(ts.value))] - r.event.gps
        assert -0.05 < t < 0.05, det
    assert len(r.provenance["data_files"]) == 2
    assert all(f["sha256"] for f in r.provenance["data_files"])


def test_newest_catalog_event_resolves():
    events = gwsonify.list_events()
    newest = max(events, key=lambda e: e["name"])
    ev = gwsonify.get_event(newest["name"])
    assert ev.gps > 1.4e9 and ev.catalog


@pytest.mark.model
def test_bns_model_gwtc1_file(tmp_path):
    pytest.importorskip("lalsimulation")
    r = gwsonify.sonify_model("GW170817", detectors=["L1"], f_low=40, outdir=tmp_path, plots=False)
    m = r.provenance["model"]
    assert m["approximant"] == "IMRPhenomPv2_NRTidal"
    assert "psi" in m["assumed_parameters"]  # GWTC-1 files lack psi
    assert len(r.audio["L1"]) / r.rate > 20  # BNS from 40 Hz lasts ~25 s
