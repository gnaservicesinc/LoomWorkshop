#!/bin/bash
set -euo pipefail
workshop_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -d "$workshop_root/.build/application/Loom Workshop.app" ]]; then
    "$workshop_root/Scripts/build-app.sh"
fi
open "$workshop_root/.build/application/Loom Workshop.app" --args "$@"
