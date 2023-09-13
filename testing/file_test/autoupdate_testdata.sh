#!/usr/bin/env bash

# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

bazel run -c opt //testing/file_test:file_test_base_test -- --autoupdate "$@"
