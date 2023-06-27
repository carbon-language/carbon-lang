#!/usr/bin/env bash

# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/utils/treesitter"

FILES=$(find "$PWD/testdata/" -not -name "fail_*" -name "*.carbon" -type f -print)
FILES_FAIL=$(find "$PWD/testdata/" -name "fail_*.carbon" -type f -print)

OUTPUT="$(for f in $FILES; do bazel run "//utils/treesitter:treesitter_carbon_tester" "$f" 1>/dev/null 2>&1 || echo "$f" ; done)"
FAILED=$(echo -n "$OUTPUT" | grep -c '^')
TOTAL=$(echo -n "$FILES" | grep -c '^')

echo "$FAILED/$TOTAL failed"
echo "$OUTPUT"
echo

OUTPUT_FAIL="$(for f in $FILES_FAIL; do bazel run "//utils/treesitter:treesitter_carbon_tester" "$f" 1>/dev/null 2>&1 && echo "$f" || true ; done)"
PASSED=$(echo -n "$OUTPUT_FAIL" | grep -c '^')
TOTAL=$(echo -n "$FILES_FAIL" | grep -c '^')
echo "$PASSED/$TOTAL false positive"
echo "$OUTPUT_FAIL"
