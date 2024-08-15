// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/tuple_param_with_return_slot.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/tuple_param_with_return_slot.carbon

fn F(a: (), b: (i32,), c: (i32, i32)) -> (i32, i32, i32) {
  // CHECK:STDERR: tuple_param_with_return_slot.carbon:[[@LINE+11]]:11: ERROR: Type `(i32,)` does not support indexing.
  // CHECK:STDERR:   return (b[0], c[0], c[1]);
  // CHECK:STDERR:           ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: tuple_param_with_return_slot.carbon:[[@LINE+7]]:17: ERROR: Type `(i32, i32)` does not support indexing.
  // CHECK:STDERR:   return (b[0], c[0], c[1]);
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: tuple_param_with_return_slot.carbon:[[@LINE+3]]:23: ERROR: Type `(i32, i32)` does not support indexing.
  // CHECK:STDERR:   return (b[0], c[0], c[1]);
  // CHECK:STDERR:                       ^~~~
  return (b[0], c[0], c[1]);
}

fn Main() {
  F((), (1,), (2, 3));
}
