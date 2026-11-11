"""Integrity and smoke tests for the bundled real-data examples.

The examples ship compact replay stacks derived from public CC-BY-4.0
products.  These tests verify that

* the committed arrays still match the checksums in ``replay_config.json``;
* metadata and array shapes agree;
* both the D-InSAR and SBAS routes can reproduce the published reference
  products within a tight tolerance.

They do not download anything and do not touch ``examples/*/results``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from insar_studio.importer import import_stack
from insar_studio.pipeline import run_pipeline
from insar_studio.project import ProjectStore

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
DINGRI = EXAMPLES / "01_dingri_2025_dinsar"
BAIGE = EXAMPLES / "02_baige_landslide_sbas"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_dir", [DINGRI, BAIGE], ids=["dingri", "baige"])
def test_example_payload_checksums(case_dir: Path) -> None:
    config_path = case_dir / "data" / "replay_config.json"
    config = _load_json(config_path)
    assert config["schema"] == "insar-studio/example-replay/v1"
    for name, expected in config["outputs"].items():
        path = case_dir / "data" / name
        assert path.exists(), f"{path} 缺失"
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_dingri_metadata() -> None:
    meta = _load_json(DINGRI / "data" / "meta.json")
    slc = np.load(DINGRI / "data" / "slc.npy", mmap_mode="r")
    reference = np.load(DINGRI / "data" / "reference_los_mm.npy")
    mask = np.load(DINGRI / "data" / "mask.npy")
    assert slc.shape == (2, 500, 500)
    assert slc.dtype == np.complex64
    assert meta["rows"] == 500 and meta["cols"] == 500
    assert meta["n_dates"] == 2
    assert meta["replay_phase_scale"] == 10.0
    assert reference.shape == (500, 500)
    assert mask.shape == (500, 500)
    assert not mask[0].any() and not mask[-1].any()


def test_baige_metadata() -> None:
    meta = _load_json(BAIGE / "data" / "meta.json")
    slc = np.load(BAIGE / "data" / "slc.npy", mmap_mode="r")
    truth = np.load(BAIGE / "data" / "truth_disp_mm.npy")
    mask = np.load(BAIGE / "data" / "mask.npy")
    assert slc.shape == (41, 51, 95)
    assert slc.dtype == np.complex64
    assert meta["rows"] == 51 and meta["cols"] == 95
    assert meta["n_dates"] == 41
    assert meta["dates"][0] == "2018-12-14"
    assert meta["dates"][-1] == "2022-03-16"
    assert meta["dates_source"] == "reconstructed"
    assert truth.shape == (41, 51, 95)
    assert mask.shape == (51, 95)
    assert mask.sum() == 1800


def test_dingri_dinsar_replay(tmp_path: Path) -> None:
    meta = _load_json(DINGRI / "data" / "meta.json")
    reference = np.load(DINGRI / "data" / "reference_los_mm.npy")
    store = ProjectStore(tmp_path / "workspace")
    project = store.create("dingri test", "dinsar", params={"remove_ramp": False}, source="import")
    root = store.path(project["id"])
    import_stack(root, DINGRI / "data", technique="dinsar", meta_overrides={})
    run_pipeline(
        root,
        technique="dinsar",
        params={
            "remove_ramp": False,
            "multilook": 1,
            "filter": "none",
            "coherence_window": 5,
            "min_coherence": 0.0,
            "unwrap_coherence_min": 0.0,
            "unwrap_method": "quality",
            "reference_mode": "manual",
            "reference_row": 24,
            "reference_col": 63,
        },
        steps=["network", "interferometry", "inversion"],
    )
    output = np.load(root / "products" / "arrays" / "disp_pair_mm.npy", allow_pickle=False)
    scaled = output.astype(np.float64) * float(meta["replay_phase_scale"])
    valid = np.isfinite(scaled) & np.isfinite(reference)
    residual = scaled[valid] - reference[valid]
    residual -= np.median(residual)
    assert np.sqrt(np.mean(residual**2)) < 0.01
    assert np.corrcoef(scaled[valid], reference[valid])[0, 1] > 0.999999


def test_baige_sbas_replay(tmp_path: Path) -> None:
    truth = np.load(BAIGE / "data" / "truth_disp_mm.npy")
    mask = np.load(BAIGE / "data" / "mask.npy").astype(bool)
    store = ProjectStore(tmp_path / "workspace")
    project = store.create(
        "baige test",
        "sbas",
        params={"remove_ramp": False, "max_temporal_baseline": 60, "max_pairs": 300},
        source="import",
    )
    root = store.path(project["id"])
    import_stack(root, BAIGE / "data", technique="sbas", meta_overrides={})
    run_pipeline(
        root,
        technique="sbas",
        params={
            "remove_ramp": False,
            "multilook": 1,
            "filter": "none",
            "coherence_window": 5,
            "min_coherence": 0.0,
            "unwrap_coherence_min": 0.0,
            "unwrap_method": "quality",
            "max_temporal_baseline": 60,
            "max_pairs": 300,
            "estimate_dem_error": False,
            "aps_method": "off",
            "aps_iterations": 0,
            "robust_iterations": 1,
            "regularization": 0.0,
            "reference_mode": "manual",
            "reference_row": 9,
            "reference_col": 30,
        },
        steps=["network", "interferometry", "inversion"],
    )
    inversion = _load_json(root / "products" / "inversion.json")
    row, col = (int(v) for v in inversion["reference_pixel"])
    output = np.load(root / "products" / "arrays" / "disp_ts_mm.npy", allow_pickle=False)
    expected = truth - truth[:, row, col][:, None, None]
    actual = output - output[:, row, col][:, None, None]
    valid = np.isfinite(actual) & mask[None, :, :]
    residual = actual[valid] - expected[valid]
    assert np.sqrt(np.mean(residual**2)) < 1.0
    assert np.corrcoef(actual[valid], expected[valid])[0, 1] > 0.999
