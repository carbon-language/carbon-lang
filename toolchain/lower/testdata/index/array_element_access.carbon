// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/index/array_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/index/array_element_access.carbon
fn A() -> (i32, i32) { return (1, 2); }

fn B() -> [i32; 2] { return (1, 2); }

fn Run() {
  var a: [i32; 2] = A();
  // CHECK:STDERR: array_element_access.carbon:[[@LINE+3]]:16: ERROR: Type `(i32, i32)` does not support indexing.
  // CHECK:STDERR:   var b: i32 = A()[0];
  // CHECK:STDERR:                ^~~~~~
  var b: i32 = A()[0];
  var c: i32 = a[b];
  var d: i32 = B()[1];
}
