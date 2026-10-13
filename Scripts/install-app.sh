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
python3 - "$workshop_root/.build/application/Loom Workshop.app" "$workshop_destdir" <<'PY'
import datetime
import pathlib
import plistlib
import subprocess
import sys
import uuid

source = pathlib.Path(sys.argv[1])
installation_root = pathlib.Path(sys.argv[2]).expanduser().resolve()
destination = installation_root / 'Applications/Loom Workshop.app'
destination.parent.mkdir(parents=True, exist_ok=True)
with (source / 'Contents/Info.plist').open('rb') as handle:
    metadata = plistlib.load(handle)
    if metadata.get('CFBundleIdentifier') != 'com.gnaservicesinc.LoomWorkshop':
        raise SystemExit('The built bundle is not Loom Workshop.')
    if metadata.get('LoomBuildConfiguration') != 'Release':
        raise SystemExit('Install requires an optimized Release build of Loom Workshop.')
running = subprocess.run(['/bin/ps', '-axo', 'command='], check=True, text=True, capture_output=True).stdout
if str(destination / 'Contents/MacOS/LoomWorkshop') in running:
    raise SystemExit('Quit Loom Workshop before installing this build.')
stage = destination.with_name(f'.Loom Workshop.staging-{uuid.uuid4().hex}.app')
subprocess.run(['/usr/bin/ditto', str(source), str(stage)], check=True)
subprocess.run(['/usr/bin/codesign', '--verify', '--strict', '--verbose=2', str(stage)], check=True)
backup = None
if destination.exists():
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    backup_root = (pathlib.Path.home() / 'Library/Application Support/Loom Workshop/Backups'
                   if installation_root == pathlib.Path('/') else installation_root / '.loom-suite-backups/Loom Workshop')
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f'Loom Workshop.backup-{timestamp}-{uuid.uuid4().hex[:8]}.app'
    destination.rename(backup)
try:
    stage.rename(destination)
    subprocess.run(['/usr/bin/codesign', '--verify', '--strict', '--verbose=2', str(destination)], check=True)
except BaseException:
    if destination.exists():
        destination.rename(stage)
    if backup is not None:
        backup.rename(destination)
    raise
print(destination)
if backup is not None:
    print(f'Previous app retained at {backup}')
PY
