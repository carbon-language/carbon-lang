#!/bin/bash -eux
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs an example migration of the Brotli C++ code.

# cd to the carbon-lang root.
cd "$(dirname "$0")/../../.."

EXAMPLE=third_party/examples/woff2

# Remove any previous conversion.
rm -rf "${EXAMPLE}/carbon/"

# Initialize the converted directory, omitting unnecessary subdirectories.
rsync -a \
  "${EXAMPLE}/original/" \
  "${EXAMPLE}/carbon/" \
  --exclude .git \
  --exclude .gitmodules \
  --exclude cmake

# Create a compile_flags.txt for the `carbon` directory.
cp "${EXAMPLE}/BUILD.original" \
  "${EXAMPLE}/carbon/BUILD"
cp "${EXAMPLE}/WORKSPACE.original" \
  "${EXAMPLE}/carbon/WORKSPACE"
cp "${EXAMPLE}/compile_flags.carbon.txt" \
  "${EXAMPLE}/carbon/compile_flags.txt"

# Run migration on the copy.
bazel build //migrate_cpp
./bazel-bin/migrate_cpp/migrate_cpp \
  "${EXAMPLE}/carbon"
