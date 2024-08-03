// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/index/tuple_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/index/tuple_element_access.carbon

fn Run() -> i32 {
  var a: (i32, i32, i32) = (0, 1, 2);
  // CHECK:STDERR: tuple_element_access.carbon:[[@LINE+4]]:16: ERROR: Type `(i32, i32, i32)` does not support indexing.
  // CHECK:STDERR:   var b: i32 = a[0];
  // CHECK:STDERR:                ^~~~
  // CHECK:STDERR:
  var b: i32 = a[0];
  // CHECK:STDERR: tuple_element_access.carbon:[[@LINE+3]]:16: ERROR: Type `(i32, i32, i32)` does not support indexing.
  // CHECK:STDERR:   var c: i32 = a[2];
  // CHECK:STDERR:                ^~~~
  var c: i32 = a[2];
  return 0;
}
