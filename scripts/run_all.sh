#!/usr/bin/env bash
set -euo pipefail

# End-to-end paper experiment driver.
#
# Usage:
#   PYTHONPATH=src ./scripts/run_all.sh

PYTHONPATH="${PYTHONPATH:-src}" python scripts/run_paper_experiments.py "$@"
