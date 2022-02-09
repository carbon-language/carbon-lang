#!/bin/bash -eu
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

readonly DIR="$(dirname "$0")"
bazel run :precedence -- \
  --dot_path="$(realpath "${DIR}/precedence.dot")" \
  --svg_path="$(realpath "${DIR}/../precedence.svg")"
