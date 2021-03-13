#!/bin/bash
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Script to generate a compilation database for all the C++ code in Carbon, as
# well as all the LLVM dependencies and any generated code.
#
# By default this uses the first of `bazelisk` or `bazel` on your PATH. These
# can be overridden by setting the environment variable `BAZEL` when invoking
# this script.
#
# Running this script will end up building all of the targets as well in order
# to make sure that generated code and any include paths are present.
#
# You will want to periodically re-run this to generate a fresh compile
# database as files are added. However, nothing will immediately break. Tools
# like `clang-tidy` and `clangd` should largely continue working even with a
# stale database, but the cross reference index may not be complete.
#
# Usage: ./scripts/create_compdb.sh

set -euo pipefail

# Try to use `bazelisk` if on the PATH but fall back to `bazel`.
BAZEL_PATH_SPELLING=bazelisk
if ! type $BAZEL_PATH_SPELLING &>/dev/null; then
  BAZEL_PATH_SPELLING=bazel
fi

# Set our command if not overridden by the user.
: ${BAZEL:=$BAZEL_PATH_SPELLING}

# Ensure it works.
if ! type "$BAZEL" &>/dev/null; then
  echo >&2 "Unable to run bazel!"
  exit 1
fi

# Collect all of the source files in the transitive closure.
all_sources=$(
  $BAZEL query --keep_going --output location 2>/dev/null \
      'filter(".*\.(h|cpp|cc|c|cxx)$", kind("source file", deps(//...)))' \
  | cut -d: -f1
)

# Now filter into different groups that we will expose based on their
# organization in the source tree. We don't use Bazel's file organization to
# make the locations of these files work better with IDEs and editors by
# matching the way you would open the file for edit.
carbon_sources=$(
  echo "$all_sources" \
  | grep "^$PWD" \
  | sed "s!$PWD/!!"
)
llvm_sources=$(
  echo "$all_sources" \
  | grep "/external/llvm-project/" \
  | sed -E "s!.*/(external/llvm-project/)!bazel-carbon-lang/\\1!"
)

# Collect all the generated files in the transitive closure. These will be
# labels and not locations though.
generated_labels=$(
  $BAZEL query --keep_going --output label 2>/dev/null \
      'filter(".*\.(h|cpp|cc|c|cxx|def|inc)$", kind("generated file", deps(//...)))'
)

# Build these generated files as we can't do much without them. But just print
# a warning if there are any issues.
if ! $BAZEL build --keep_going 2>/dev/null $generated_labels; then
  echo >&2 "WARNING: Some generated files failed to build!"
fi

# Translate the generated file labels into the most obvious paths for opening
# in an editor or IDE. We want to index with those paths to smooth the opening
# experience.
#
# The strategy is to replace '@repository//' with
# '...-bin/external/repository/' and '//' with '...-bin/'. This leverages the
# '...-bin' symlink to find and use whatever the most recent build's generated
# files are available. Lastly, we translate the `:` into a `/`.
#
# Anything we can't translate is stripped. We also strip off `.inc` and `.def`
# files that shouldn't be indexed, but *should* be built above.
generated_sources=$(
  echo "$generated_labels" \
  | sed -E 's!^@([^/]+)//!bazel-bin/external/\1/!' \
  | sed -E 's!^//!bazel-bin/!' \
  | tr : / \
  | grep '^bazel-bin' | grep -Ev '\.(inc|def)$'
)

# Begin writing out the JSON compilation database.
(
  echo "["
  EMIT_COMMA=false
  echo -e "$carbon_sources\n$llvm_sources\n$generated_sources" \
      | while read file; do

    # JSON is very strict with commas being delimiters, so emit it specially here.
    if [[ $EMIT_COMMA == true ]]; then
      echo "  ,"
    fi

    echo "  {"
    echo "    \"file\": \"$file\","
    echo "    \"directory\": \"$PWD\","
    echo "    \"arguments\": ["
    echo "      \"clang\","
    while read flag; do
      echo "      \"${flag//\"/\\\"}\","
    done <compile_flags.txt
    echo "      \"-fsyntax-only\","
    echo "      \"$file\""
    echo "    ]"
    echo "  }"

    # Ensure a comma if we have more than one file.
    EMIT_COMMA=true
  done
  echo "]"
) >compile_commands.json
