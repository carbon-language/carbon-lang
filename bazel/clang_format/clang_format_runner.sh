#!/bin/bash
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# USAGE: clang_format_runner.sh <output-path> <clang-format> <clang-format-args>
#
# Bazel needs every action to have at least one output;
# thus, when validating formatting we have to generate a file.

set -euo pipefail

OUTPUT_PATH="${1}"
shift 1

"$@" && touch "${OUTPUT_PATH}"
