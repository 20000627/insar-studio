#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
      PY="$candidate"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "[ERROR] 需要 Python 3.9 或更高版本。"
  echo "请从 https://www.python.org/downloads/ 安装后重试。"
  exit 1
fi

exec "$PY" scripts/bootstrap.py "$@"
