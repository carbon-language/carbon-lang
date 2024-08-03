// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/index/tuple_return_value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/index/tuple_return_value_access.carbon

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  // CHECK:STDERR: tuple_return_value_access.carbon:[[@LINE+3]]:16: ERROR: Type `(i32, i32)` does not support indexing.
  // CHECK:STDERR:   var t: i32 = F()[1];
  // CHECK:STDERR:                ^~~~~~
  var t: i32 = F()[1];
}
