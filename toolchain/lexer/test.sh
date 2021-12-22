#!/bin/bash

# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

bazel test :string_literal_test :string_literal_bm --test_output=errors
echo fastbuild
bazel run -c fastbuild :string_literal_bm |& grep BM_
echo opt
bazel run -c opt :string_literal_bm |& grep BM_
