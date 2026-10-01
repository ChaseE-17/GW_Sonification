"""End-to-end pipeline and CLI behaviour, offline (recorded API + synthetic strain)."""

import json

import numpy as np
import pytest
from scipy.io import wavfile

import gwsonify
from gwsonify import GwsonifyError, MissingDependencyError, cli
from gwsonify import strain as S


@pytest.fixture
def fake_strain(monkeypatch, synthetic_strain, fake_api):
    """Serve synthetic strain instead of downloading GWOSC files."""

    def load(det, start, end, *, sample_rate=4096, event=None, quiet=False):
        if event is not None and det not in event.detectors(sample_rate):
            raise GwsonifyError(f"No {det} strain is published for {event.name}. "
                                f"Available: {', '.join(event.detectors(sample_rate))}.")
        t0 = event.gps if event is not None else (start + end) / 2
        ts = synthetic_strain(t0=t0, span=(start - t0, end - t0), fs=sample_rate,
                              inject=5.0, det=det, seed=hash(det) % 100)
        return ts, [(f"https://gwosc.example/{det}.hdf5", None)]

    monkeypatch.setattr(S, "load", load)


def test_sonify_data_end_to_end(fake_strain, tmp_path):
    r = gwsonify.sonify_data("GW150914", outdir=tmp_path)
    names = sorted(p.name for p in r.files)
    assert names == ["GW150914_H1_data-whiten.wav", "GW150914_L1_data-whiten.wav",
                     "GW150914_data-whiten.png", "GW150914_data-whiten.provenance.json"]
    rate, pcm = wavfile.read(tmp_path / "GW150914" / "GW150914_H1_data-whiten.wav")
    assert rate == 44100 and pcm.dtype == np.int16
    assert len(pcm) == pytest.approx(4.0 * 44100, abs=2)  # default BBH window [-3, 1]
    assert pcm[0] == 0 and pcm[-1] == 0  # faded
    # joint normalisation: the loudest channel peaks at -3 dBFS
    peak = max(np.max(np.abs(v)) for v in r.audio.values())
    assert peak == pytest.approx(10 ** (-3 / 20), rel=1e-3)

    prov = json.loads((tmp_path / "GW150914" / "GW150914_data-whiten.provenance.json").read_text())
    assert prov["event"]["spec"] == "GW150914-v4"
    assert prov["processing"]["strain"]["mode"] == "whiten"
    assert prov["processing"]["strain"]["gate"] is True
    assert prov["processing"]["audio"]["speed"] == 1.0
    assert "Gravitational Wave Open Science Center" in prov["citation"]["acknowledgement"]
    assert prov["data_files"][0]["url"].startswith("https://")
    assert prov["software"]["gwsonify"] == gwsonify.__version__


def test_settings_tag_and_stereo(fake_strain, tmp_path):
    r = gwsonify.sonify_data("GW150914", outdir=tmp_path, stereo=True, speed=0.5, fshift=400,
                             plots=False)
    wavs = [p.name for p in r.files if p.suffix == ".wav"]
    assert wavs == ["GW150914_data-whiten_speed0.5_fshift400_stereo.wav"]
    _, pcm = wavfile.read(r.files[0])
    assert pcm.shape[1] == 2
    assert len(pcm) == pytest.approx(8.0 * 44100, abs=4)  # half speed doubles duration


def test_fixed_norm_is_comparable_across_runs(fake_strain, tmp_path):
    a = gwsonify.sonify_data("GW150914", outdir=tmp_path / "a", norm="fixed", plots=False)
    b = gwsonify.sonify_data("GW150914", outdir=tmp_path / "b", norm="fixed", plots=False,
                             detectors=["H1"])
    # same gain regardless of which detectors were included
    assert a.provenance["processing"]["gain"] == b.provenance["processing"]["gain"]
    assert a.provenance["processing"]["gain"] == pytest.approx(10 ** (-30 / 20))


def test_unavailable_detector_message(fake_strain):
    with pytest.raises(GwsonifyError, match="No V1 strain for GW150914; available: H1, L1"):
        gwsonify.sonify_data("GW150914", detectors=["V1"])


def test_argument_errors(fake_api):
    with pytest.raises(GwsonifyError, match="event name or --gps"):
        gwsonify.sonify_data()
    with pytest.raises(GwsonifyError, match="choose detectors"):
        gwsonify.sonify_data(gps=1126259462.4)
    with pytest.raises(GwsonifyError, match="--template needs --mode whiten"):
        gwsonify.sonify_data("GW150914", mode="raw", template=True)


# -- CLI ---------------------------------------------------------------------------------


def test_bare_event_is_data_command():
    assert cli._normalize_argv(["GW150914", "-d", "H1"]) == ["data", "GW150914", "-d", "H1"]
    assert cli._normalize_argv(["info", "GW150914"]) == ["info", "GW150914"]
    assert cli._normalize_argv(["--version"]) == ["--version"]
    args = cli.build_parser().parse_args(["data", "GW150914", "-d", "H1,l1", "-d", "V1",
                                          "--window", "-8", "2", "--fshift", "400"])
    assert args.detectors == ["H1", "L1", "V1"]
    assert args.window == [-8.0, 2.0]
    assert args.gate is None  # "auto": on in whiten mode


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "gwsonify GW150914" in out and "examples:" in out
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert gwsonify.__version__ in capsys.readouterr().out
    assert cli.main([]) == 2


def test_usage_error_exit_code():
    with pytest.raises(SystemExit) as e:
        cli.main(["data", "GW150914", "--mode", "loud"])
    assert e.value.code == 2


def test_cli_error_is_clean_message(fake_api, capsys):
    assert cli.main(["info", "GW15091"]) == 1
    err = capsys.readouterr().err
    assert "Did you mean: GW150914" in err
    assert "Traceback" not in err


def test_cli_missing_dependency_exit_code(fake_api, monkeypatch, capsys):
    def boom(*a, **k):
        raise MissingDependencyError("Waveform models need the 'model' extra")

    monkeypatch.setattr(gwsonify, "sonify_model", boom)
    assert cli.main(["model", "GW150914"]) == 3
    assert "model' extra" in capsys.readouterr().err


def test_cli_unexpected_error_hides_traceback(fake_api, monkeypatch, capsys):
    def boom(*a, **k):
        raise ZeroDivisionError("oops")

    monkeypatch.setattr(gwsonify, "sonify_data", boom)
    assert cli.main(["GW150914"]) == 1
    err = capsys.readouterr().err
    assert "unexpected error" in err and "Traceback" not in err


def test_cli_data_prints_files_and_json(fake_strain, tmp_path, capsys):
    assert cli.main(["GW150914", "-o", str(tmp_path), "--no-plots", "-q"]) == 0
    out = capsys.readouterr().out.split()
    assert all(p.startswith(str(tmp_path)) for p in out) and len(out) == 3
    assert cli.main(["GW150914", "-o", str(tmp_path), "--no-plots", "--json", "-q"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "data-whiten" and len(summary["files"]) == 3


def test_cli_info_json(fake_api, capsys):
    assert cli.main(["info", "GW150914", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["spec"] == "GW150914-v4"
    assert info["strain_detectors"] == ["H1", "L1"]
    assert info["preferred_pe_label"] == "C01:Mixed"


def test_cli_info_text(fake_api, capsys):
    assert cli.main(["info", "GW150914"]) == 0
    out = capsys.readouterr().out
    assert "C01:Mixed  (preferred)" in out and "H1, L1" in out


def test_cli_list_filters(fake_api, capsys):
    assert cli.main(["list", "GW1905"]) == 0
    out = capsys.readouterr().out
    assert "GW190521" in out and "GW150914" not in out
    assert cli.main(["list", "GW17*", "--json"]) == 0
    assert [e["name"] for e in json.loads(capsys.readouterr().out)] == ["GW170817"]


def test_cli_cache(capsys):
    assert cli.main(["cache"]) == 0
    assert "cache" in capsys.readouterr().out
    assert cli.main(["cache", "--clear"]) == 0


def test_data_release_citation_by_run():
    from gwsonify.provenance import data_release_paper

    assert "SoftwareX" in data_release_paper("O1")
    assert "ApJS 267" in data_release_paper("O3b")
    assert "ae211e" in data_release_paper("O4a")
    assert "2605.27090" in data_release_paper("O4c1DiscC00")
    assert data_release_paper("S5") is None
