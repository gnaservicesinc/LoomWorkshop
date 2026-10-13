#!/bin/bash
# Build a private runtime in the selected source project. No global packages.
set -euo pipefail
project_source="$1"
runtime_code="$(cd "$(dirname "$0")" && pwd)"
project_storage="$project_source/.loom-workshop"
runtime_directory="$project_storage/runtime"
bootstrap_directory="$project_storage/runtime-bootstrap"
runtime_child=""
cancel_setup() {
  if [[ -n "$runtime_child" ]]; then kill "$runtime_child" 2>/dev/null || true; fi
  printf '%s\n' '{"event":"cancelled","message":"Runtime preparation stopped."}'
  exit 130
}
trap cancel_setup INT TERM
run_setup_command() {
  "$@" &
  runtime_child=$!
  wait "$runtime_child"
  runtime_child=""
}
mkdir -p "$bootstrap_directory"
if [[ -x "$runtime_directory/bin/python" ]] && "$runtime_directory/bin/python" -c 'import torch,coremltools,safetensors,numpy,PIL,png,tifffile,pillow_heif,OpenEXR' >/dev/null 2>&1; then
  runtime_python="$runtime_directory/bin/python"
else
  printf '%s\n' '{"event":"status","message":"Preparing the project training runtime…"}'
  if command -v uv >/dev/null 2>&1; then
    runtime_uv="$(command -v uv)"
  elif [[ -x "$bootstrap_directory/uv" ]]; then
    runtime_uv="$bootstrap_directory/uv"
  else
    machine_arch="$(uname -m)"
    if [[ "$machine_arch" == "arm64" ]]; then release_arch="aarch64"; else release_arch="x86_64"; fi
    run_setup_command /usr/bin/curl --fail --location --retry 2 --connect-timeout 15 --max-time 300 \
      "https://github.com/astral-sh/uv/releases/latest/download/uv-$release_arch-apple-darwin.tar.gz" \
      --output "$bootstrap_directory/uv.tar.gz"
    /usr/bin/tar -xzf "$bootstrap_directory/uv.tar.gz" -C "$bootstrap_directory" --strip-components=1
    runtime_uv="$bootstrap_directory/uv"
  fi
  export UV_PYTHON_INSTALL_DIR="$project_storage/python"
  export UV_CACHE_DIR="$project_storage/runtime-cache"
  run_setup_command "$runtime_uv" python install 3.12
  project_python="$("$runtime_uv" python find --managed-python 3.12)"
  run_setup_command "$runtime_uv" venv --allow-existing --python "$project_python" "$runtime_directory"
  run_setup_command "$runtime_uv" pip install --python "$runtime_directory/bin/python" --requirement "$runtime_code/requirements.lock"
  runtime_python="$runtime_directory/bin/python"
fi
"$runtime_python" -c 'import json,sys;print(json.dumps({"event":"runtime_ready","python":sys.executable}))'
