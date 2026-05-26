// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self/fail_self_not_first.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self/fail_self_not_first.carbon

class Class {
  // `self` must be the first explicit parameter.
  // CHECK:STDERR: fail_self_not_first.carbon:[[@LINE+4]]:22: error: `self` must be the first explicit parameter [SelfNotFirstParam]
  // CHECK:STDERR:   fn F(other: Class, self: Self);
  // CHECK:STDERR:                      ^~~~~~~~~~
  // CHECK:STDERR:
  fn F(other: Class, self: Self);
}
