"""Build the self-contained source release archive.

The release is a whitelist, not a copy of the working tree: runtime data,
stress workspaces, virtual environments, caches and local credentials must
never leak into a package that is handed to another user.

Usage::

    python scripts/package_release.py
    python scripts/package_release.py --output D:/tmp/insar-studio.zip
"""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "insar_studio"
DEFAULT_OUTPUT_DIR = ROOT / "release"

INCLUDE_ROOT_FILES = (
    ".env.example",
    ".gitignore",
    "LICENSE",
    "README.md",
    "RELEASE.md",
    "pyproject.toml",
    "requirements.txt",
    "start.bat",
    "start.sh",
)
INCLUDE_EXTRA_FILES = (
    "scripts/bootstrap.py",
    "scripts/package_release.py",
)
INCLUDE_DIRS = {
    "insar_studio": ("*.py", "*.html", "*.css", "*.js"),
    "docs": ("*.md", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg"),
    "tests": ("*.py",),
    "examples": ("*.md", "*.py", "*.json", "*.npy", "*.tif", "*.png"),
}

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".egg-info",
    "build",
    "dist",
    "workspace",
    "release",
    "node_modules",
}
FORBIDDEN_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".npy",
    ".npz",
    ".pkl",
    ".joblib",
    ".h5",
    ".hdf5",
    ".tif",
    ".tiff",
    ".log",
    ".tmp",
    ".temp",
    ".bak",
    ".swp",
    ".zip",
    ".7z",
    ".tar",
    ".gz",
)


def _say(message: str = "") -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def _read_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
    if not match:
        raise SystemExit("pyproject.toml 缺少 version")
    version = match.group(1)
    init = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'^__version__\s*=\s*"([^"]+)"', init, flags=re.MULTILINE)
    if not init_match:
        raise SystemExit("insar_studio/__init__.py 缺少 __version__")
    if init_match.group(1) != version:
        raise SystemExit(
            f"版本不一致: pyproject={version}, __init__={init_match.group(1)}"
        )
    return version


def _is_example_payload(relative: PurePosixPath) -> bool:
    """Allow the compact replay data and results that ship with the examples.

    ``*.npy`` and ``*.tif`` remain forbidden everywhere else; the exception is
    limited to ``examples/<case>/data`` and ``examples/<case>/results``.
    """
    parts = relative.parts
    if len(parts) < 4 or parts[0] != "examples":
        return False
    if parts[2] not in ("data", "results"):
        return False
    return relative.suffix.lower() in (".npy", ".tif", ".tiff")


def _is_forbidden(relative: PurePosixPath) -> bool:
    lower_parts = {part.lower() for part in relative.parts}
    if lower_parts & FORBIDDEN_PARTS:
        return True
    if len(relative.parts) >= 3 and relative.parts[0] == "examples" and relative.parts[2] == "source":
        return True
    if any(part.endswith(".egg-info") for part in lower_parts):
        return True
    if any(part.startswith("stress") for part in lower_parts):
        return True
    name = relative.name.lower()
    if name == ".env":
        return True
    if name.startswith(".env.") and name != ".env.example":
        return True
    if name.startswith(("credentials", "secrets")):
        return True
    if name.endswith(".json") and ("token" in name or "secret" in name):
        return True
    if _is_example_payload(relative):
        return False
    return name.endswith(FORBIDDEN_SUFFIXES)


def _collect_files() -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE_ROOT_FILES + INCLUDE_EXTRA_FILES:
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f"发布清单缺少文件: {name}")
        files.append(path)

    for directory, patterns in INCLUDE_DIRS.items():
        root = ROOT / directory
        if not root.is_dir():
            raise SystemExit(f"发布清单缺少目录: {directory}")
        for pattern in patterns:
            files.extend(path for path in root.rglob(pattern) if path.is_file())

    unique: dict[PurePosixPath, Path] = {}
    for path in files:
        relative = PurePosixPath(path.relative_to(ROOT).as_posix())
        if _is_forbidden(relative):
            continue
        unique[relative] = path
    return [unique[key] for key in sorted(unique)]


def _zip_info(relative: PurePosixPath, source: Path) -> zipfile.ZipInfo:
    # Fixed timestamp keeps repeated builds byte-for-byte reproducible.
    info = zipfile.ZipInfo(str(relative), date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = source.stat().st_mode & 0o777
    if not mode:
        mode = 0o644
    if source.suffix.lower() in (".sh", ".bat"):
        mode |= 0o755
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def _required_files(version: str) -> set[str]:
    return {
        "README.md",
        "RELEASE.md",
        "LICENSE",
        "pyproject.toml",
        "requirements.txt",
        ".env.example",
        "start.bat",
        "start.sh",
        "scripts/bootstrap.py",
        "scripts/package_release.py",
        "insar_studio/__init__.py",
        "insar_studio/cli.py",
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


def validate_archive(path: Path, version: str) -> dict[str, object]:
    required = _required_files(version)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        files = [name for name in names if not name.endswith("/")]
        if len(files) != len(set(files)):
            raise SystemExit("发布包包含重复文件")
        if not files:
            raise SystemExit("发布包为空")

        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise SystemExit(f"发布包包含非法路径: {name}")
            if _is_forbidden(relative):
                raise SystemExit(f"发布包包含禁止内容: {name}")

        missing = sorted(required - set(files))
        if missing:
            raise SystemExit("发布包缺少必需文件: " + ", ".join(missing))

        for name in required:
            if archive.getinfo(name).file_size <= 0:
                raise SystemExit(f"发布包必需文件为空: {name}")

        bad = archive.testzip()
        if bad:
            raise SystemExit(f"发布包 CRC 校验失败: {bad}")

        # pyproject and the package version must agree inside the archive too.
        pyproject = archive.read("pyproject.toml").decode("utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
        if not match or match.group(1) != version:
            raise SystemExit("发布包内 pyproject.toml 版本不一致")

        return {
            "files": len(files),
            "bytes": sum(archive.getinfo(name).file_size for name in files),
        }


def checksum_path(output: Path) -> Path:
    """Sidecar file used by recipients to verify the downloaded archive."""
    return output.with_name(output.name + ".sha256")


def build_release(output: Path | None = None) -> Path:
    version = _read_version()
    output = output or (DEFAULT_OUTPUT_DIR / f"insar-studio-{version}.zip")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum = checksum_path(output)
    # Never leave a checksum from a previous build next to a new archive.
    if checksum.exists():
        checksum.unlink()

    collected = _collect_files()
    if not collected:
        raise SystemExit("没有收集到任何待发布文件")

    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for source in collected:
                relative = PurePosixPath(source.relative_to(ROOT).as_posix())
                archive.writestr(_zip_info(relative, source), source.read_bytes())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = validate_archive(output, version)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    temporary_checksum = checksum.with_name(checksum.name + ".part")
    try:
        temporary_checksum.write_bytes(
            f"{digest}  {output.name}\n".encode("utf-8")
        )
        temporary_checksum.replace(checksum)
    finally:
        if temporary_checksum.exists():
            temporary_checksum.unlink()
    _say(f"发布包: {output}")
    _say(f"校验文件: {checksum}")
    _say(
        f"版本 {version}：{summary['files']} 个文件，"
        f"解压后 {summary['bytes'] / 1024 / 1024:.2f} MiB，"
        f"ZIP {output.stat().st_size / 1024 / 1024:.2f} MiB"
    )
    _say(f"SHA-256: {digest}")
    _say("校验通过：路径安全，必需文件齐全，未包含运行数据、缓存或凭据。")
    return output


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="生成 InSAR Studio 发布 ZIP")
    parser.add_argument("--output", type=Path, default=None, help="输出 ZIP 路径")
    args = parser.parse_args(argv)
    try:
        build_release(args.output)
    except OSError as exc:
        _say(f"[ERROR] 打包失败: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
