#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Clean out any files in the Bazel disk cache which haven't been used for over
# thirty days.

set -eu

# Default to the same directory in the project `.blazerc`, but you can set this
# environment variable to override that.
: ${BAZEL_DISK_CACHE_PATH:=~/.cache/carbon-lang-build-cache}

# As a courtesy, compute and print some approximate stats.
total_file_count=$(find "$BAZEL_DISK_CACHE_PATH" -type f | wc -l)
stale_file_count=$(find "$BAZEL_DISK_CACHE_PATH" -type f -atime +30 | wc -l)
echo "Removing $stale_file_count files out of $total_file_count total."

# Just re-running the find is simpler than managing any state.
find "$BAZEL_DISK_CACHE_PATH" -type f -atime +30 -delete
