#!/bin/bash
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Script to run `clang-format` (from your PATH) over all of the Carbon source
# code. This should be invoked from the root of the Carbon repository.
#
# By default this uses the `clang-format` on your PATH, and the first of
# `bazelisk` or `bazel` on your PATH. These can be overridden by setting the
# environment variables `CLANG_FORMAT` and `BAZEL` respectively when invoking
# this script.
#
# Usage: ./scripts/clang_format_all.sh

set -euo pipefail

# Try to use `bazelisk` if on the PATH but fall back to `bazel`.
BAZEL_PATH_SPELLING=bazelisk
if ! type $BAZEL_PATH_SPELLING &>/dev/null; then
  BAZEL_PATH_SPELLING=bazel
fi

# Set our commands if not overridden by the user.
: ${CLANG_FORMAT:=clang-format}
: ${BAZEL:=$BAZEL_PATH_SPELLING}

# Ensure they work.
if ! type "$CLANG_FORMAT" &>/dev/null; then
  echo >&2 "Unable to run clang-format!"
  exit 1
fi
if ! type "$BAZEL" &>/dev/null; then
  echo >&2 "Unable to run bazel!"
  exit 1
fi

# Loop over all the C++ source files in the project.
$BAZEL query --keep_going --output location 2>/dev/null \
    'filter(".*\.(h|cpp)$", kind("source file", deps(//..., 1)))' \
    | cut -d: -f1 \
    | while read source; do
  echo >&2 "Formatting: $source ..."
  $CLANG_FORMAT -i $source
done
