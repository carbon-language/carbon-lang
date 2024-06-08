#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

BENCHMARK="$TEST_SRCDIR/$TEST_WORKSPACE/common/set_benchmark"

exec "$BENCHMARK" \
  --benchmark_counters_tabular=true \
  --benchmark_min_time=1x \
  --benchmark_filter='^[^/]*/[1-9][0-9]{0,3}(/[0-9]+)?$'
