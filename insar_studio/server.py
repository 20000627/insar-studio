"""WSGI entry point for the local web application.

The server is deliberately a normal single-process WSGI server.  It chooses an
available local port, optionally opens the browser, and keeps all data inside
the project's ``workspace`` directory.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser

from .api import app


def _port_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def choose_port(host: str, requested: int, *, strict: bool = False) -> int:
    """Return a usable port, moving forward when the requested one is busy."""
    if requested == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])
    if _port_free(host, requested):
        return requested
    if strict:
        raise OSError(f"端口 {requested} 已被占用")
    for candidate in range(requested + 1, min(65535, requested + 30)):
        if _port_free(host, candidate):
            return candidate
    raise OSError(f"端口 {requested} 附近没有可用端口")


def _open_browser(url: str, delay: float = 1.0) -> None:
    timer = threading.Timer(delay, lambda: webbrowser.open(url, new=2))
    timer.daemon = True
    timer.start()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    host = os.environ.get("INSAR_STUDIO_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("INSAR_STUDIO_PORT", "8765"))
    except ValueError:
        port = 8765
    debug = os.environ.get("INSAR_STUDIO_DEBUG", "0") == "1"
    open_browser = os.environ.get("INSAR_STUDIO_OPEN", "0") == "1"
    strict = os.environ.get("INSAR_STUDIO_STRICT_PORT", "0") == "1"

    for i, arg in enumerate(argv):
        if arg in ("--host",) and i + 1 < len(argv):
            host = argv[i + 1]
        elif arg in ("--port",) and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                print("[ERROR] --port 必须是整数")
                return 2
        elif arg == "--debug":
            debug = True
        elif arg in ("--open", "--browser"):
            open_browser = True
        elif arg in ("--no-browser", "--no-open"):
            open_browser = False
        elif arg == "--strict-port":
            strict = True
        elif arg in ("--help", "-h"):
            print(
                "用法: python -m insar_studio serve "
                "[--host 127.0.0.1] [--port 8765] [--open|--no-browser] [--strict-port] [--debug]"
            )
            return 0

    try:
        actual_port = choose_port(host, port, strict=strict)
    except OSError as exc:
        print(f"[ERROR] {exc}")
        print("请关闭占用端口的程序，或使用 --port 0 让系统自动选择端口。")
        return 2
    if actual_port != port:
        print(f"提示: 端口 {port} 已被占用，自动改用 {actual_port}。")

    url_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{url_host}:{actual_port}/"
    print(f"InSAR Studio · {url}")
    print("按 Ctrl+C 停止。数据只写入当前工作区的 workspace/ 目录。")
    if open_browser:
        _open_browser(url)

    try:
        from werkzeug.serving import make_server

        server = make_server(host, actual_port, app, threaded=True)
        if debug:
            app.debug = True
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    except OSError as exc:
        print(f"[ERROR] 无法启动服务: {exc}")
        return 2
    finally:
        try:
            server.shutdown()  # type: ignore[possibly-undefined]
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
