#!/usr/bin/env bash
# Actual package or iOS Simulator tests. Physical phones are never selected.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
case "${1:-logic}" in
  logic) python3 scripts/run_bounded.py logic-tests 8192 swift test --disable-sandbox -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete ;;
  native) python3 scripts/run_bounded.py native-tests 8192 python3 scripts/run_native_tests.py ;;
  *) python3 scripts/run_bounded.py filtered-tests 8192 swift test --disable-sandbox --filter "$1" ;;
esac
