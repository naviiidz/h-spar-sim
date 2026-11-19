#!/usr/bin/env bash
# Start the flow generator UI with the Firedrake environment.
# Override any of these by exporting them before running this script.
set -euo pipefail

: "${FIREDRAKE_VENV:=/home/navid/h-spar-test/Hspar-backup/particle_backend/venv-firedrake}"
: "${PETSC_DIR:=/home/navid/h-spar-test/Hspar-backup/particle_backend/petsc}"
: "${PETSC_ARCH:=arch-firedrake-default}"

export PETSC_DIR PETSC_ARCH
export FIREDRAKE_PYTHON="${FIREDRAKE_PYTHON:-$FIREDRAKE_VENV/bin/python}"

exec "$FIREDRAKE_PYTHON" "$(dirname "$(readlink -f "$0")")/app.py" "$@"
