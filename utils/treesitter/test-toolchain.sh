#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/utils/treesitter"

./build.sh

FILES=$(find "$ROOT/toolchain/parser/testdata" -not -name "fail_*" -name "*.carbon" -type f -print)
FILES_FAIL=$(find "$ROOT/toolchain/parser/testdata" -name "fail_*.carbon" -type f -print)

OUTPUT="$(for f in $FILES; do tree-sitter parse "$f" 1>/dev/null || echo "$f" ; done)"
FAILED=$(echo -n "$OUTPUT" | grep -c '^')
TOTAL=$(echo -n "$FILES" | grep -c '^')

echo "$FAILED/$TOTAL failed"
echo "$OUTPUT"
echo

OUTPUT_FAIL="$(for f in $FILES_FAIL; do tree-sitter parse "$f" 1>/dev/null && echo "$f" || true ; done)"
PASSED=$(echo -n "$OUTPUT_FAIL" | grep -c '^')
TOTAL=$(echo -n "$FILES_FAIL" | grep -c '^')
echo "$PASSED/$TOTAL false positive"
echo "$OUTPUT_FAIL"
