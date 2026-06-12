#!/usr/bin/env bash
# Export build/full_book_portrait.html → build/full_book_portrait.pdf
# using Edge headless. Works on Windows Git Bash; override EDGE for other paths.

set -e
cd "$(dirname "$0")/../.."   # repo root

HTML="$(pwd)/build/full_book_portrait.html"
PDF="$(pwd)/build/full_book_portrait.pdf"

if [ ! -f "$HTML" ]; then
  echo "missing $HTML — run scripts/book/build_full_portrait.py first" >&2
  exit 1
fi

EDGE="${EDGE:-/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe}"
if [ ! -x "$EDGE" ]; then
  EDGE_FALLBACK="/c/Program Files/Microsoft/Edge/Application/msedge.exe"
  [ -x "$EDGE_FALLBACK" ] && EDGE="$EDGE_FALLBACK"
fi
if [ ! -x "$EDGE" ]; then
  echo "Microsoft Edge not found. Set EDGE env to msedge.exe full path." >&2
  exit 1
fi

TMPPROF="$(mktemp -d)/edge_profile"
mkdir -p "$TMPPROF"

URL="file:///$HTML"
"$EDGE" --headless=new --disable-gpu --no-pdf-header-footer \
  --user-data-dir="$TMPPROF" --print-to-pdf="$PDF" "$URL"

ls -la "$PDF"
