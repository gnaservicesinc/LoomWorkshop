#!/bin/bash
set -euo pipefail
workshop_root="$(cd "$(dirname "$0")/.." && pwd)"
workshop_skip_build=false
workshop_destdir=/
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) workshop_skip_build=true; shift ;;
        --destdir)
            [[ $# -ge 2 ]] || { printf '%s\n' '--destdir requires a destination root.' >&2; exit 2; }
            workshop_destdir="$2"; shift 2 ;;
        *) printf 'Unknown install argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ "$workshop_skip_build" != true ]]; then "$workshop_root/Scripts/build-app.sh"; fi
swift "$workshop_root/Scripts/InstallApp.swift" "$workshop_root/.build/application/Loom Workshop.app" "$workshop_destdir"
