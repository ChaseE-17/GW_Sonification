"""Posterior label discovery and sample selection (no PE files, no LALSuite needed)."""

import numpy as np
import pytest

from gwsonify import GwsonifyError
from gwsonify import model as M
from gwsonify.catalog import Event, PEAnalysis

KNOWN = ["IMRPhenomXPHM", "IMRPhenomXO4a", "IMRPhenomPv2", "IMRPhenomPv2_NRTidal",
         "SEOBNRv4PHM", "SEOBNRv3", "NRSur7dq4", "IMRPhenomNSBH", "IMRPhenomXAS"]


@pytest.mark.parametrize(("label", "approx", "flags"), [
    ("C01:IMRPhenomXPHM", "IMRPhenomXPHM", {}),
    ("C00:IMRPhenomXPHM-SpinTaylor", "IMRPhenomXPHM", {"PhenomXPrecVersion": 320}),
    ("bilby-IMRPhenomXPHM-SpinTaylor_prod-reweighted", "IMRPhenomXPHM", {"PhenomXPrecVersion": 320}),
    ("bilby-NRSur7dq4_prod-reweighted", "NRSur7dq4", {}),
    ("PhenomXO4a", "IMRPhenomXO4a", {}),
    ("IMRPhenomPv2NRT_lowSpin", "IMRPhenomPv2_NRTidal", {}),
    ("SEOBNRv3", "SEOBNRv3", {}),
    ("C01:SEOBNRv4PHM", "SEOBNRv4PHM", {}),
    ("C01:Mixed", None, {}),
    ("TDB:Mixed", None, {}),
    ("C00:Mixed:HighSpin", None, {}),
    ("Overall", None, {}),
    ("PublicationSamples", None, {}),
    ("C01:SomethingNew", None, {}),
])
def test_approximant_from_label(label, approx, flags):
    assert M.approximant_from_label(label, KNOWN) == (approx, flags)


def test_rank_prefers_fast_phenom_models():
    ordered = sorted(["SEOBNRv4PHM", "NRSur7dq4", "IMRPhenomXPHM", "IMRPhenomPv2_NRTidal"],
                     key=M.rank_approximant)
    assert ordered[0] == "IMRPhenomXPHM"
    assert ordered.index("IMRPhenomPv2_NRTidal") < ordered.index("SEOBNRv4PHM")
    assert M.rank_approximant(None) > M.rank_approximant("Anything")


@pytest.mark.parametrize(("api", "labels", "expected"), [
    ("C01:Mixed", ["C01:IMRPhenomXPHM", "C01:Mixed", "C01:SEOBNRv4PHM"], "C01:Mixed"),
    ("PhenomXPHM", ["bilby-IMRPhenomXPHM-SpinTaylor_prod-reweighted"],
     "bilby-IMRPhenomXPHM-SpinTaylor_prod-reweighted"),
    ("IMRPhenomPv2NRT_lowSpin_prior", ["IMRPhenomPv2NRT_highSpin", "IMRPhenomPv2NRT_lowSpin"],
     "IMRPhenomPv2NRT_lowSpin"),
    ("C01:Nope", ["A", "B"], None),
])
def test_match_label(api, labels, expected):
    assert M.match_label(api, labels) == expected


def _posterior(labels, approximants, api_label):
    pe = PEAnalysis("x", api_label, "https://example/pe.h5", True)
    return M.Posterior(path=None, url=pe.data_url, analysis=pe, labels=labels,
                       approximants=dict(zip(labels, approximants, strict=True)),
                       meta={lab: {} for lab in labels})


def test_choose_label_mixed_falls_back_to_best_single_model():
    post = _posterior(["C01:Mixed", "C01:SEOBNRv4PHM", "C01:IMRPhenomXPHM"],
                      [None, "SEOBNRv4PHM", "IMRPhenomXPHM"], "C01:Mixed")
    label, why = M.choose_label(post)
    assert label == "C01:IMRPhenomXPHM"
    assert "combines several waveform models" in why


def test_choose_label_uses_preferred_when_generatable():
    post = _posterior(["C00:IMRPhenomXPHM-SpinTaylor", "C00:SEOBNRv5PHM"],
                      ["IMRPhenomXPHM", "SEOBNRv5PHM"], "C00:IMRPhenomXPHM-SpinTaylor")
    assert M.choose_label(post)[0] == "C00:IMRPhenomXPHM-SpinTaylor"


def test_choose_label_explicit_and_errors():
    post = _posterior(["A", "B"], ["IMRPhenomXPHM", None], "B")
    assert M.choose_label(post, "B") == ("B", "requested")
    with pytest.raises(GwsonifyError, match="Available: A, B"):
        M.choose_label(post, "C")
    empty = _posterior(["Mixed"], [None], "Mixed")
    with pytest.raises(GwsonifyError, match="--approximant"):
        M.choose_label(empty)


class FakeSamples(dict):
    @property
    def parameters(self):
        return list(self)

    @property
    def number_of_samples(self):
        return len(next(iter(self.values())))


def _samples(n=200, with_likelihood=True, seed=1):
    rng = np.random.default_rng(seed)
    s = FakeSamples(mass_1=rng.normal(36, 3, n), mass_2=rng.normal(30, 3, n),
                    luminosity_distance=rng.normal(440, 100, n), a_1=rng.uniform(0, 1, n))
    s["log_likelihood"] = rng.normal(100, 5, n) if with_likelihood else np.zeros(n)
    s["log_prior"] = rng.normal(0, 1, n)
    return s


def test_pick_sample_maxl_maxp_index_median():
    s = _samples()
    sample, desc, idx = M.pick_sample(s, "maxl")
    assert idx == int(np.argmax(s["log_likelihood"])) and "likelihood" in desc
    assert sample["mass_1"] == s["mass_1"][idx]
    _, _, idx_p = M.pick_sample(s, "maxp")
    assert idx_p == int(np.argmax(s["log_likelihood"] + s["log_prior"]))
    assert M.pick_sample(s, 7)[2] == 7
    sample, desc, _ = M.pick_sample(s, "median")
    assert abs(sample["mass_1"] - np.median(s["mass_1"])) < 3  # a real, central sample
    with pytest.raises(GwsonifyError, match="out of range"):
        M.pick_sample(s, 10_000)
    with pytest.raises(GwsonifyError, match="--sample"):
        M.pick_sample(s, "best")


def test_pick_sample_without_likelihood_falls_back_to_median():
    s = _samples(with_likelihood=False)
    _, desc, _ = M.pick_sample(s, "maxl")
    assert "median" in desc
    with pytest.raises(GwsonifyError):
        M.pick_sample(s, "maxp")


def test_complete_sample_gwtc1_style_fills_and_reports_assumptions():
    ev = Event("GW170817", 3, "GWTC-1-confident", 1187008882.4, "O2")
    raw = {"mass_1": 1.5, "mass_2": 1.3, "luminosity_distance": 40.0, "cos_theta_jn": -0.8,
           "a_1": 0.02, "a_2": 0.01, "cos_tilt_1": 0.5, "cos_tilt_2": -0.3, "ra": 3.4, "dec": -0.4,
           "lambda_1": 300.0, "lambda_2": 400.0}
    full, assumed = M.complete_sample(raw, ev)
    assert full["theta_jn"] == pytest.approx(np.arccos(-0.8))
    assert full["tilt_1"] == pytest.approx(np.arccos(0.5))
    assert full["geocent_time"] == ev.gps and assumed["geocent_time"] == ev.gps
    assert set(assumed) >= {"psi", "phase", "phi_jl", "phi_12", "geocent_time"}
    assert "a_1" not in assumed  # present in the file, not assumed
    assert raw == {k: raw[k] for k in raw}  # input not modified
    assert "theta_jn" not in raw


def test_complete_sample_keeps_file_values():
    ev = Event("GW150914", 4, "c", 1126259462.4, "O1")
    raw = {"mass_1": 36.0, "mass_2": 30.0, "luminosity_distance": 400.0, "theta_jn": 2.0,
           "phi_jl": 1.0, "tilt_1": 0.3, "tilt_2": 0.2, "phi_12": 0.5, "a_1": 0.3, "a_2": 0.2,
           "phase": 1.1, "psi": 0.7, "geocent_time": 1126259462.41, "ra": 1.0, "dec": -1.0}
    full, assumed = M.complete_sample(raw, ev)
    assert assumed == {}
    assert full["geocent_time"] == 1126259462.41  # never overwritten
