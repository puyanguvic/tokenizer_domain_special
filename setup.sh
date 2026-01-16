#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
EXTRAS="${EXTRAS:-dev,hf}"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--extras <list>] [--no-extras]

Creates a .venv with uv and installs the project in editable mode.
Examples:
  ./setup.sh
  ./setup.sh --extras dev,hf,transformers
  EXTRAS=dev,hf ./setup.sh
  VENV_DIR=/tmp/ctok-venv ./setup.sh
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --extras)
      if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
        echo "Missing value for --extras" >&2
        exit 1
      fi
      EXTRAS="$2"
      shift 2
      ;;
    --no-extras)
      EXTRAS=""
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it from https://github.com/astral-sh/uv" >&2
  exit 1
fi

uv venv "$VENV_DIR"

INSTALL_TARGET="."
if [ -n "$EXTRAS" ]; then
  INSTALL_TARGET=".[${EXTRAS}]"
fi

uv pip install --python "$VENV_DIR/bin/python" -e "$INSTALL_TARGET"

cat <<EOF
Setup complete.
Activate the environment with:
  source "$VENV_DIR/bin/activate"
EOF
