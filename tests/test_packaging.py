"""Packaging, version and launcher consistency checks."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

import insar_studio

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"insar_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bootstrap():
    return _load_script("bootstrap")


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None, "pyproject.toml 缺少 version"
    return match.group(1)


def test_project_version_matches_package_version():
    assert _pyproject_version() == insar_studio.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", insar_studio.__version__)


def test_package_data_patterns_match_non_empty_files():
    package = ROOT / "insar_studio"
    patterns = (
        "templates/*.html",
        "static/css/*.css",
        "static/js/*.js",
    )
    for pattern in patterns:
        matches = [path for path in package.glob(pattern) if path.is_file()]
        assert matches, f"package-data 模式没有匹配文件: {pattern}"
        for path in matches:
            assert path.stat().st_size > 0, f"空资源文件: {path}"


def test_bootstrap_resolves_venv_and_detects_current_interpreter():
    bootstrap = _load_bootstrap()

    assert bootstrap.ROOT == ROOT
    assert bootstrap.VENV == ROOT / ".venv"
    expected = bootstrap.VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert bootstrap._venv_python() == expected
    assert bootstrap._python_is_supported(sys.executable)
    assert bootstrap._venv_dependencies_ready(Path(sys.executable))
    assert bootstrap.main(["--check"]) == 0


def test_requirements_names_are_declared_in_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert dependencies
    for requirement in dependencies:
        name = re.split(r"[<>=!~ ]", requirement, maxsplit=1)[0]
        assert name.lower() in pyproject.lower(), f"pyproject.toml 未声明依赖 {name}"
    assert "insar-studio" in pyproject
    assert "insar_studio.cli:main" in pyproject


def test_start_scripts_prefer_validated_python_versions():
    batch = (ROOT / "start.bat").read_text(encoding="utf-8")
    shell = (ROOT / "start.sh").read_text(encoding="utf-8")

    # Windows: try the validated interpreters before falling back to PATH or py -3.
    match = re.search(r"for %%V in \(([^)]+)\)", batch)
    assert match is not None
    assert match.group(1).split() == ["3.11", "3.10", "3.9", "3.12", "3.13"]
    assert "scripts\\bootstrap.py" in batch

    # POSIX: same preference order in the candidate list.
    candidates = re.findall(r"python3\.\d+|python3\b|python\b", shell)
    assert candidates[:5] == ["python3.11", "python3.10", "python3.9", "python3.12", "python3.13"]
    assert "scripts/bootstrap.py" in shell


def test_bootstrap_check_flag_is_documented():
    for name in ("README.md", "docs/使用指南.md", "docs/用户手册.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "start.bat --check" in text, f"{name} 未说明 --check"


def test_release_forbidden_paths_are_rejected():
    packaging = _load_script("package_release")

    for name in (
        "workspace/projects/prj_1/stack.npy",
        "stress/run.log",
        "stress_calibration/result.json",
        ".venv/Lib/site-packages/x.py",
        "insar_studio/__pycache__/api.cpython-311.pyc",
        "insar_studio.egg-info/SOURCES.txt",
        ".env",
        "secrets/token.json",
        "release/insar-studio-1.2.0.zip",
        "examples/01_dingri_2025_dinsar/source/ASC_20250105_20250117.adf_quad.unw.geo.tif",
        "examples/01_dingri_2025_dinsar/workspace/project.json",
        "examples/02_baige_landslide_sbas/source/disp_data.rar",
        "examples/02_baige_landslide_sbas/raw/Dt.npy",
    ):
        assert packaging._is_forbidden(PurePosixPath(name)), name

    for name in (
        "insar_studio/api.py",
        "insar_studio/static/js/app.js",
        "docs/使用指南.md",
        "docs/images/ui-results.png",
        "tests/test_packaging.py",
        "RELEASE.md",
        ".env.example",
        "examples/README.md",
        "examples/01_dingri_2025_dinsar/data/slc.npy",
        "examples/01_dingri_2025_dinsar/results/dingri_los_mm.tif",
        "examples/02_baige_landslide_sbas/data/slc.npy",
        "examples/02_baige_landslide_sbas/results/baige_velocity_mm_yr.tif",
    ):
        assert not packaging._is_forbidden(PurePosixPath(name)), name


def test_release_archive_contains_only_whitelisted_files(tmp_path):
    packaging = _load_script("package_release")
    output = tmp_path / "insar-studio.zip"
    checksum_file = tmp_path / "insar-studio.zip.sha256"
    checksum_file.write_text("stale checksum\n", encoding="utf-8")

    built = packaging.build_release(output)

    assert built == output.resolve()
    with zipfile.ZipFile(built) as archive:
        names = set(archive.namelist())
        required = {
            "README.md",
            "RELEASE.md",
            "LICENSE",
            "pyproject.toml",
            "requirements.txt",
            "start.bat",
            "start.sh",
            "scripts/bootstrap.py",
            "scripts/package_release.py",
            "insar_studio/api.py",
            "insar_studio/templates/index.html",
            "insar_studio/static/js/app.js",
            "insar_studio/static/css/app.css",
            "tests/test_packaging.py",
            "tests/test_examples.py",
            "docs/新手指南.md",
            "docs/技术说明.md",
            "docs/使用指南.md",
            "docs/用户手册.md",
            "examples/README.md",
            "examples/01_dingri_2025_dinsar/README.md",
            "examples/01_dingri_2025_dinsar/PROVENANCE.md",
            "examples/01_dingri_2025_dinsar/data/slc.npy",
            "examples/01_dingri_2025_dinsar/results/summary.json",
            "examples/02_baige_landslide_sbas/README.md",
            "examples/02_baige_landslide_sbas/PROVENANCE.md",
            "examples/02_baige_landslide_sbas/data/slc.npy",
            "examples/02_baige_landslide_sbas/results/summary.json",
        }
        assert required <= names
        assert archive.testzip() is None
        assert all(not packaging._is_forbidden(PurePosixPath(name)) for name in names)
        assert not any(name.startswith(("workspace/", "stress", ".venv/")) for name in names)
        assert not any(name.startswith(("examples/01_dingri_2025_dinsar/source/", "examples/02_baige_landslide_sbas/source/")) for name in names)
        assert not any("/workspace/" in f"/{name}" for name in names)
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
        assert archive.getinfo("insar_studio/templates/index.html").file_size > 0

    digest = hashlib.sha256(built.read_bytes()).hexdigest()
    assert packaging.checksum_path(built) == checksum_file
    assert checksum_file.read_text(encoding="utf-8") == f"{digest}  {built.name}\n"
    assert not list(tmp_path.glob("*.part"))

