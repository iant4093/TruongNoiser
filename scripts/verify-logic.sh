#!/usr/bin/env bash
# Mac package regressions, independent trained references and canonical asset hashes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TRUONGNOISER_QUALITY_EVIDENCE_DIR="$ROOT/build/test-results/quality/mac"
mkdir -p "$ROOT/build/logs" "$TRUONGNOISER_QUALITY_EVIDENCE_DIR"
python3 scripts/run_bounded.py package-tests 8192 \
  swift test --disable-sandbox -j 2 -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
checks_python="${TRUONGNOISER_CHECKS_PYTHON:-}"
if [ -z "$checks_python" ]; then
  python3 -m venv "$ROOT/build/check-python"
  checks_python="$ROOT/build/check-python/bin/python"
  "$checks_python" -m pip --disable-pip-version-check install --quiet numpy==2.5.4
fi
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
"$checks_python" -m unittest discover -s ModelConversionTools -p 'test_cgnet_promotion.py'
"$checks_python" -m unittest discover -s ModelConversionTools -p 'test_cgnet_validation.py'
python3 ModelConversionTools/verify_production_assets.py --checkpoints-dir ModelConversionTools/checkpoints
