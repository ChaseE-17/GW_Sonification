"""Event resolution and source heuristics (offline, recorded API responses)."""

import math

import pytest

from gwsonify import GwsonifyError
from gwsonify import catalog as C
from gwsonify import strain as S


def test_default_version_is_newest_confident(fake_api):
    ev = C.get_event("GW150914")
    assert ev.spec == "GW150914-v4"
    assert ev.catalog == "GWTC-2.1-confident"
    assert ev.gps == pytest.approx(1126259462.4)
    assert ev.doi.startswith("https://doi.org/")


def test_explicit_version_and_catalog(fake_api):
    assert C.get_event("GW150914-v3").catalog == "GWTC-1-confident"
    assert C.get_event("GW150914@GWTC-1-confident").version == 3
    with pytest.raises(GwsonifyError, match="no version matching"):
        C.get_event("GW150914-v9")


def test_choose_version_skips_marginal_and_preliminary():
    versions = [{"version": 1, "catalog": "GWTC-3-confident"},
                {"version": 2, "catalog": "GWTC-3-marginal"},
                {"version": 3, "catalog": "O1_O2-Preliminary"}]
    assert C.choose_version(versions)["version"] == 1
    assert C.choose_version(versions[1:])["version"] == 3


def test_unknown_event_suggests_names(fake_api):
    with pytest.raises(GwsonifyError) as err:
        C.get_event("GW15091")
    assert "Did you mean: GW150914" in str(err.value)
    with pytest.raises(GwsonifyError, match="GW190521, GW190521_030229, GW190521_074359"):
        C.get_event("GW19052")


def test_pe_analyses_and_labels_come_from_api(fake_api):
    ev = C.get_event("GW150914")
    pe = ev.preferred_pe()
    assert pe.label == "C01:Mixed"
    assert pe.data_url.endswith(".h5")
    assert ev.parameters["mass_1_source"] == pytest.approx(34.6)


def test_strain_files_and_available_detectors(fake_api):
    ev = C.get_event("GW150914")
    assert ev.detectors(4096) == ["H1", "L1"]
    files = ev.strain_files(4096)
    assert {f.duration for f in files} >= {32.0, 4096.0}
    # listed detectors are not the same as strain availability, so both are kept
    assert ev.listed_detectors == ("H1", "L1")


def test_pick_file_prefers_smallest_covering_file(fake_api):
    ev = C.get_event("GW150914")
    start, end = S._span(ev.gps, (-3, 1))
    f = S.pick_file(ev.strain_files(4096), "H1", start, end)
    assert f.duration == 32.0
    assert "GWTC-1-confident" in f.url  # chosen version's catalog family beats preliminary
    big = S.pick_file(ev.strain_files(4096), "H1", ev.gps - 100, ev.gps + 100)
    assert big.duration == 4096.0
    assert S.pick_file(ev.strain_files(4096), "V1", start, end) is None


def test_source_heuristics(fake_api):
    ev = C.get_event("GW150914")
    assert ev.source_class == "BBH"
    # leading-order chirp time from 20 Hz for Mc_det ~ 30.7 Msun is ~0.73 s
    assert ev.time_to_merger(20.0) == pytest.approx(0.73, rel=0.05)
    # f_ISCO = 4400 Hz / (M/Msun)
    assert ev.f_isco() == pytest.approx(4397 / ev.total_mass_det, rel=0.01)
    assert S.default_window(ev) == (-3.0, 1.0)
    assert S.default_band(ev) == (20.0, 300)


def test_time_to_merger_scaling():
    ev = C.Event("X", 1, "c", 0.0, "O1", parameters={"chirp_mass": 1.2})
    tau = ev.time_to_merger(20.0)
    assert tau == pytest.approx(157, rel=0.05)  # classic BNS number (~2.5 minutes from 20 Hz)
    assert ev.time_to_merger(40.0) == pytest.approx(tau * 2 ** (-8 / 3), rel=1e-9)
    ev2 = C.Event("X", 1, "c", 0.0, "O1", parameters={"mass_1_source": 1.4, "mass_2_source": 1.3})
    assert ev2.source_class == "BNS"
    assert S.default_band(ev2) == (20.0, 1000)
    assert S.default_window(ev2)[0] == -120.0
    assert S.default_window(None) == (-8.0, 2.0)
    assert math.isclose(S.default_band(None)[1], 500)


def test_list_events(fake_api):
    events = C.list_events()
    names = {e["name"] for e in events}
    assert {"GW150914", "GW170817"} <= names
