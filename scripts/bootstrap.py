"""Zero-configuration launcher used by start.bat / start.sh.

Only the Python standard library is imported here.  If the current interpreter
already has the runtime dependencies, it is used directly.  Otherwise a local
``.venv`` is created and ``requirements.txt`` is installed once, so the user
never has to edit an interpreter path by hand.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"

# module name -> minimum version (informational, avoids re-installing forever)
DEPS = {
    "numpy": (1, 24),
    "scipy": (1, 10),
    "skimage": (0, 20),
    "matplotlib": (3, 7),
    "flask": (2, 2),
    "tifffile": (2021, 7, 2),
    "docx": (1, 1),
    "requests": (2, 28),
}


def _say(message: str = "") -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.replace("-", ".").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts or [0])


def _missing_dependencies() -> list[str]:
    missing: list[str] = []
    for module, minimum in DEPS.items():
        if importlib.util.find_spec(module) is None:
            missing.append(module)
            continue
        try:
            from importlib.metadata import version

            package = {"skimage": "scikit-image", "docx": "python-docx", "flask": "Flask"}.get(module, module)
            installed = _version_tuple(version(package))
            if installed < minimum:
                missing.append(f"{module} (<{'.'.join(map(str, minimum))})")
        except Exception:
            # The module imports; a metadata lookup failure is not fatal.
            pass
    return missing


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _python_is_supported(python: Path | str) -> bool:
    """Return True for a 64-bit Python 3.9+ interpreter."""
    try:
        probe = subprocess.run(
            [
                str(python),
                "-c",
                "import sys, struct; raise SystemExit(0 if sys.version_info >= (3, 9) and struct.calcsize('P') * 8 == 64 else 1)",
            ],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _venv_dependencies_ready(python: Path) -> bool:
    try:
        probe = subprocess.run(
            [
                str(python),
                "-c",
                "import numpy, scipy, skimage, matplotlib, flask, tifffile, docx, requests",
            ],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _install_into_venv(no_install: bool) -> Path:
    python = _venv_python()
    if python.exists() and not _python_is_supported(python):
        _say("[ERROR] 现有 .venv 不是 64 位 Python 3.9+。")
        _say("请删除项目目录下的 .venv 后重新运行 start.bat。")
        raise SystemExit(3)
    if not python.exists():
        _say("首次运行：正在创建本地 Python 环境 .venv ...")
        if no_install:
            _say("[ERROR] 依赖缺失且指定了 --no-install。请先运行: python -m pip install -r requirements.txt")
            raise SystemExit(3)
        try:
            subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            _say(f"[ERROR] 无法创建 .venv：{exc}")
            _say("请确认 Python 安装包含 venv/ensurepip，或手动安装 requirements.txt。")
            raise SystemExit(3)
    _say("正在安装运行依赖（只需一次，之后会直接启动）...")
    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)],
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        _say(f"[ERROR] 依赖安装失败：{exc}")
        _say("请检查网络，或在项目目录手动运行：")
        _say(f'  "{python}" -m pip install -r requirements.txt')
        raise SystemExit(3)
    return python


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.version_info < (3, 9):
        _say(f"[ERROR] 需要 Python 3.9 或更高版本，当前为 {sys.version.split()[0]}。")
        _say("请从 https://www.python.org/downloads/ 安装，并勾选 Add python.exe to PATH。")
        return 2
    if sys.maxsize <= 2**32:
        _say("[ERROR] 需要 64 位 Python；当前解释器是 32 位，无法安装 NumPy/SciPy 轮子。")
        _say("请安装 64 位 Python 3.11，并重新运行 start.bat。")
        return 2
    if sys.version_info[:2] > (3, 11):
        _say(f"[提示] 当前 Python {sys.version.split()[0]} 高于已验证的 3.11；如果依赖安装失败，请改用 Python 3.11。")
    if not REQUIREMENTS.exists():
        _say(f"[ERROR] 找不到 {REQUIREMENTS}")
        return 2

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--check", action="store_true", help="只检查环境与依赖，不启动服务")
    parser.add_argument("--no-install", action="store_true", help="依赖缺失时不要自动创建虚拟环境")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--port", type=int, default=None, help="指定端口（默认自动选择 8765 起）")
    parser.add_argument("--strict-port", action="store_true")
    known, passthrough = parser.parse_known_args(argv)

    missing = _missing_dependencies()
    if known.check:
        venv_python = _venv_python()
        if (
            missing
            and venv_python.exists()
            and _python_is_supported(venv_python)
            and _venv_dependencies_ready(venv_python)
        ):
            _say("环境检查通过：使用现有 .venv，依赖完整。")
            return 0
        if missing:
            _say("环境检查：当前解释器缺少依赖: " + ", ".join(missing))
            _say("运行 start.bat（或 python scripts/bootstrap.py）可自动创建 .venv 并安装。")
            return 3
        _say(f"环境检查通过：Python {sys.version.split()[0]}，依赖完整。")
        return 0
    if missing:
        _say("检测到缺失依赖: " + ", ".join(missing))
        python = _install_into_venv(known.no_install)
        # Verify inside the new environment.
        check = subprocess.run([str(python), "-c", "import numpy, scipy, skimage, matplotlib, flask, tifffile, docx, requests"])
        if check.returncode != 0:
            _say("[ERROR] .venv 中的依赖仍不完整，请查看上面的 pip 输出。")
            return 3
    else:
        python = Path(sys.executable)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONPATH", str(ROOT))
    command = [str(python), "-m", "insar_studio", "serve"]
    if not known.no_browser:
        command.append("--open")
    if known.port is not None:
        command.extend(["--port", str(known.port)])
    if known.strict_port:
        command.append("--strict-port")
    command.extend(passthrough)
    _say("启动 InSAR Studio...")
    try:
        return int(subprocess.call(command, cwd=str(ROOT), env=env))
    except KeyboardInterrupt:
        _say("\n已停止。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
