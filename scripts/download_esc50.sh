#!/usr/bin/env bash
# Download the ESC-50 dataset and arrange it as ./ESC-50/{audio,meta}.
# No git submodule is used. Run from the repository root:
#     bash scripts/download_esc50.sh
#
# The repo already ships ESC-50/meta/esc50.csv and a precomputed feature cache,
# so this is only needed to re-extract features from the raw audio
# (python3 experiment.py --no-cache).
set -euo pipefail

URL="https://github.com/karolpiczak/ESC-50/archive/master.zip"
ZIP="esc50.zip"

if [ -d "ESC-50/audio" ] && [ "$(ls -A ESC-50/audio 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "ESC-50/audio already present — nothing to do."
    exit 0
fi

echo "Downloading ESC-50 (~600 MB)..."
curl -L -o "$ZIP" "$URL"

echo "Unzipping..."
unzip -q "$ZIP"

mkdir -p ESC-50
mv ESC-50-master/audio ESC-50/audio
mkdir -p ESC-50/meta
mv ESC-50-master/meta/esc50.csv ESC-50/meta/esc50.csv

rm -rf ESC-50-master "$ZIP"
echo "Done. Audio is in ESC-50/audio ($(ls ESC-50/audio | wc -l) files)."
