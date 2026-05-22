// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_ref_return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_ref_return.carbon

fn F() -> ref i32 {
  // CHECK:STDERR: fail_ref_return.carbon:[[@LINE+4]]:3: error: cannot bind durable reference to non-reference value of type `i32` [ConversionFailureNonRefToRef]
  // CHECK:STDERR:   return 0;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return 0;
}
