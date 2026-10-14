#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then printf 'usage: %s /path/to/application.app\n' "$0" >&2; exit 2; fi
loom_bundle="$1"
if [[ ! -d "$loom_bundle/Contents" || -L "$loom_bundle" ]]; then
    printf 'A complete native application bundle is required: %s\n' "$loom_bundle" >&2; exit 1
fi
loom_unexpected="$(/usr/bin/find "$loom_bundle/Contents" \( -iname '*.py' -o -iname '*.pyc' -o -iname '*.pyo' \
    -o -iname 'libpython*' -o -iname 'python*.framework' -o -iname python -o -iname 'python[0-9]*' \
    -o -name pyvenv.cfg -o -name .venv -o -name venv -o -name __pycache__ -o -name '*.egg-info' \
    -o -name site-packages -o -name setup_runtime.sh -o -name Runtime -o -name loom_factory -o -name loom_workshop \) -print -quit)"
if [[ -n "$loom_unexpected" ]]; then
    printf 'Native application contains a retired interpreter or worker resource: %s\n' "$loom_unexpected" >&2; exit 1
fi
