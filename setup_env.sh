#!/usr/bin/env bash
# Put the shared modules on PYTHONPATH. Source this before running anything:
#     source setup_env.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$HERE/lib${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH=$PYTHONPATH"
