// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/string_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/string_indexing.carbon

// --- fail_wrong_type.carbon

fn TestWrongType() {
  var x: i32 = 42;
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:22: error: cannot access member of interface `Core.IndexWith(Core.IntLiteral)` in type `i32` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   var c: Core.Char = x[0];
  // CHECK:STDERR:                      ^~~~
  // CHECK:STDERR:
  var c: Core.Char = x[0];
}

