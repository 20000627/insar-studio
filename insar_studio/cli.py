"""Command-line interface: serve, demo, run, doctor, projects."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import TECHNIQUES, ensure_runtime_dirs
from .pipeline import run_pipeline
from .project import ProjectStore
from .validation import validate_demo


def _progress(value: float, message: str) -> None:
    bar_len = 24
    filled = int(max(0.0, min(1.0, value)) * bar_len)
    bar = "█" * filled + "·" * (bar_len - filled)
    print(f"\r[{bar}] {value * 100:5.1f}%  {message[:70]:<70}", end="", flush=True)


def cmd_serve(args) -> int:
    from .server import main as server_main

    extra = ["--host", args.host, "--port", str(args.port)]
    if args.debug:
        extra.append("--debug")
    if args.open:
        extra.append("--open")
    if args.no_browser:
        extra.append("--no-browser")
    if args.strict_port:
        extra.append("--strict-port")
    return server_main(extra)


def _find(store: ProjectStore, token: str) -> dict:
    projects = store.list()
    for p in projects:
        if p["id"] == token:
            return store.get(p["id"])
    for p in projects:
        if p["name"] == token:
            return store.get(p["id"])
    raise SystemExit(f"找不到项目: {token}")


def cmd_demo(args) -> int:
    ensure_runtime_dirs()
    store = ProjectStore()
    technique = args.technique.lower()
    if technique not in TECHNIQUES:
        raise SystemExit(f"未知技术路线: {technique}")
    print(f"创建演示项目 · {technique.upper()}")
    project = store.create_demo(args.name, technique)
    print(f"项目 {project['id']} · {project['name']}")
    if args.no_run:
        print(f"数据目录: {store.path(project['id'])}")
        return 0
    t0 = time.time()
    result = run_pipeline(
        store.path(project["id"]),
        technique=technique,
        params=project.get("params", {}),
        progress=_progress,
    )
    print()
    inv = result.get("inversion", {})
    print(f"完成 · {time.time() - t0:.1f} s")
    if technique == "dinsar":
        print(f"位移中位数: {inv.get('displacement_median_mm', 0):.2f} mm")
    else:
        print(f"速率中位数: {inv.get('velocity_median_mm_yr', 0):.2f} mm/yr")
        print(f"有效像元: {inv.get('valid_pixels', 0)}")
    val = validate_demo(store.path(project["id"]), force=True)
    if val:
        if technique == "dinsar":
            print(f"真值校验: 位移 RMSE {val.get('displacement_rmse_mm', 0):.2f} mm · {val.get('grade')}")
        else:
            print(f"真值校验: 位移 RMSE {val.get('displacement_rmse_mm', 0):.2f} mm · 速率 RMSE {val.get('velocity_rmse_mm_yr', 0):.2f} mm/yr · {val.get('grade')}")
    print(f"项目目录: {store.path(project['id'])}")
    return 0


def cmd_run(args) -> int:
    store = ProjectStore()
    project = _find(store, args.project)
    technique = (args.technique or project.get("technique", "sbas")).lower()
    t0 = time.time()
    result = run_pipeline(store.path(project["id"]), technique=technique, params=project.get("params", {}), progress=_progress)
    print()
    print(f"完成 · {time.time() - t0:.1f} s · {json.dumps(result.get('inversion', {}), ensure_ascii=False)[:300]}")
    return 0


def cmd_doctor(args) -> int:
    from .api import _doctor

    data = _doctor()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"InSAR Studio {data['version']} · {data['platform']}")
        print(f"工作区: {data['workspace']}")
        for check in data["checks"]:
            mark = "[OK]" if check["ok"] else "[!!]"
            print(f"  {mark:<5} {check['name']:<18} {check['value']}")
    return 0 if data["ok"] else 1


def cmd_projects(args) -> int:
    store = ProjectStore()
    rows = store.list()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("暂无项目")
        return 0
    print(f"{'ID':<26} {'名称':<24} {'路线':<7} {'进度':<8} {'更新'}")
    for row in rows:
        print(f"{row['id']:<26} {row['name'][:22]:<24} {row['technique'].upper():<7} {row['steps_done']}/{row['steps_total']:<6} {row['updated_at']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="insar-studio", description="InSAR Studio · 本地时序干涉形变处理平台")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="启动本地 Web 界面")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765, help="8765，或 0 表示自动选择空闲端口")
    p_serve.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    p_serve.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p_serve.add_argument("--strict-port", action="store_true", help="端口被占用时直接退出")
    p_serve.add_argument("--debug", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_demo = sub.add_parser("demo", help="创建并运行合成演示项目")
    p_demo.add_argument("--name", default="阿尔卑斯山谷 · 演示")
    p_demo.add_argument("--technique", default="sbas", choices=["sbas", "ps", "ds", "dinsar"])
    p_demo.add_argument("--no-run", action="store_true", help="只生成数据，不运行处理链")
    p_demo.set_defaults(func=cmd_demo)

    p_run = sub.add_parser("run", help="运行已有项目的全流程")
    p_run.add_argument("project", help="项目 ID 或名称")
    p_run.add_argument("--technique", default=None, choices=["sbas", "ps", "ds", "dinsar"])
    p_run.set_defaults(func=cmd_run)

    p_doc = sub.add_parser("doctor", help="检查运行环境")
    p_doc.add_argument("--json", action="store_true")
    p_doc.set_defaults(func=cmd_doctor)

    p_proj = sub.add_parser("projects", help="列出项目")
    p_proj.add_argument("--json", action="store_true")
    p_proj.set_defaults(func=cmd_projects)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
