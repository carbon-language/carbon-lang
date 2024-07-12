#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

BENCHMARK="$TEST_SRCDIR/$TEST_WORKSPACE/toolchain/driver/compile_benchmark"

# Run the benchmark with the fastest size and a single iteration to make sure it
# doesn't hit errors.
exec "$BENCHMARK" \
  --benchmark_min_time=1x \
  --benchmark_filter='/256$'
