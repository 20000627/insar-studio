"""DS parameter surface and stage-fingerprint regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from insar_studio.config import DEFAULT_PROCESSING
from insar_studio.params import (
    ENUMS,
    INT_RANGES,
    STAGE_PARAM_KEYS,
    changed_keys,
    fingerprint,
    validate_params,
)
from insar_studio.project import ROUTE_PARAM_KEYS, ProjectStore, route_defaults

REMOVED_DS_KEYS = ("ds_overlap", "ds_phase_linking")


@pytest.mark.parametrize("key", REMOVED_DS_KEYS)
def test_removed_ds_keys_are_absent_from_defaults_and_validation(key):
    assert key not in DEFAULT_PROCESSING
    assert key not in INT_RANGES
    assert key not in ENUMS
    assert key not in STAGE_PARAM_KEYS["interferometry"]
    assert key not in STAGE_PARAM_KEYS["inversion"]


@pytest.mark.parametrize("key", REMOVED_DS_KEYS)
def test_removed_ds_keys_are_absent_from_route_switch(key):
    assert key not in ROUTE_PARAM_KEYS
    assert key not in route_defaults("ds")


def test_ds_project_has_no_removed_keys(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("DS 默认参数", "ds")

    assert project["params"]["ds_patch"] == 9
    for key in REMOVED_DS_KEYS:
        assert key not in project["params"]


def test_ds_patch_is_owned_by_the_interferometry_stage():
    params = validate_params({"ds_patch": 9}, technique="ds")
    changed = validate_params({"ds_patch": 11}, technique="ds")

    assert "ds_patch" in STAGE_PARAM_KEYS["interferometry"]
    assert "ds_patch" not in STAGE_PARAM_KEYS["inversion"]
    assert fingerprint(params, stage="interferometry", technique="ds") != fingerprint(
        changed, stage="interferometry", technique="ds"
    )
    assert fingerprint(params, stage="inversion", technique="ds") == fingerprint(
        changed, stage="inversion", technique="ds"
    )
    assert changed_keys(params, changed, stage="interferometry") == ["ds_patch"]


def test_dinsar_pair_is_owned_by_the_network_stage():
    params = validate_params({"dinsar_pair": [0, 5]}, technique="dinsar")
    changed = validate_params({"dinsar_pair": [1, 6]}, technique="dinsar")

    assert "dinsar_pair" in STAGE_PARAM_KEYS["network"]
    assert "dinsar_pair" in STAGE_PARAM_KEYS["inversion"]
    assert fingerprint(params, stage="network", technique="dinsar") != fingerprint(
        changed, stage="network", technique="dinsar"
    )
    assert changed_keys(params, changed, stage="network") == ["dinsar_pair"]


@pytest.mark.parametrize("stage", ["coregistration", "network", "interferometry", "atmosphere", "inversion"])
def test_removed_ds_keys_no_longer_affect_any_stage_fingerprint(stage):
    base = validate_params({"ds_patch": 9}, technique="ds")
    legacy = validate_params(
        {"ds_patch": 9, "ds_overlap": 0.8, "ds_phase_linking": "eigenvalue"},
        technique="ds",
    )

    assert fingerprint(base, stage=stage, technique="ds") == fingerprint(
        legacy, stage=stage, technique="ds"
    )
    assert changed_keys(base, legacy, stage=stage) == []


def test_frontend_no_longer_offers_removed_ds_controls():
    app_js = (
        Path(__file__).resolve().parent.parent
        / "insar_studio"
        / "static"
        / "js"
        / "app.js"
    )
    text = app_js.read_text(encoding="utf-8")

    for key in REMOVED_DS_KEYS:
        assert f'data-param="{key}"' not in text
        assert key not in text
    assert 'data-param="ds_patch"' in text
