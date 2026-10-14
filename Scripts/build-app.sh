#!/bin/bash
set -euo pipefail
workshop_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$workshop_root"
swift build --configuration release
workshop_binary_dir="$(swift build --configuration release --show-bin-path)"
workshop_bundle="$workshop_root/.build/application/Loom Workshop.app"
workshop_stage="$workshop_root/.build/application/Loom Workshop.staging.app"
if [[ -e "$workshop_stage" ]]; then /bin/rm -rf "$workshop_stage"; fi
mkdir -p "$workshop_stage/Contents/MacOS" "$workshop_stage/Contents/Resources"
cp "$workshop_binary_dir/LoomWorkshop" "$workshop_stage/Contents/MacOS/LoomWorkshop"
cp Resources/Info.plist "$workshop_stage/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :LoomBuildConfiguration string Release' "$workshop_stage/Contents/Info.plist"
swift Scripts/MakeIcon.swift "$workshop_stage/Contents/Resources"
"$workshop_root/Scripts/verify-native-bundle.sh" "$workshop_stage"
codesign --force --sign - --options runtime "$workshop_stage"
codesign --verify --strict --verbose=2 "$workshop_stage"
if [[ -e "$workshop_bundle" ]]; then
    workshop_previous="$workshop_root/.build/application/Loom Workshop.previous.app"
    if [[ -e "$workshop_previous" ]]; then /bin/rm -rf "$workshop_previous"; fi
    /bin/mv "$workshop_bundle" "$workshop_previous"
fi
/bin/mv "$workshop_stage" "$workshop_bundle"
printf '%s\n' "$workshop_bundle"
