#!/bin/sh
# friend.cpp upstream sync. All the logic is in sync.py; see its --help or README.md.
exec python3 "$(dirname "$0")/sync.py" "$@"
