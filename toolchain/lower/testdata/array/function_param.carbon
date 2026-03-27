// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/function_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/function_param.carbon

fn F(arr: array(i32, 3), i: i32) -> i32 {
  return arr[i];
}

fn G() -> i32 {
  // CHECK:STDERR: function_param.carbon:[[@LINE+7]]:12: error: semantics TODO: `BISH: init_inst: {kind: Converted, arg0: inst640001B2, arg1: inst64000274, type: type(inst6400002D)}` [SemanticsTodo]
  // CHECK:STDERR:   return F((1, 2, 3), 1);
  // CHECK:STDERR:            ^~~~~~~~~
  // CHECK:STDERR: function_param.carbon:[[@LINE-8]]:6: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F(arr: array(i32, 3), i: i32) -> i32 {
  // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F((1, 2, 3), 1);
}
