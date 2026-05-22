// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Extend the "int" min_prelude to support basic integer operations, so we
// can use it here.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/recursion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/recursion.carbon

// --- recursion.carbon

library "[[@TEST_NAME]]";

eval fn F(n: i32) -> type {
  if (n == 0) {
    return i32;
  }
  return F(n - 1);
}

var i: F(3) = 0;

// TODO: Diagnose infinite recursion.
