// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/todo_dereference.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/todo_dereference.carbon

// --- test.carbon

// TODO: Dereference operator `*p` requires context of whether there is a
// preceding expression, to distinguish from binary multiplication `a * b` and
// pointer type expression `i32*`.
fn F(p: i32*) -> i32 {
  var a: i32 = *p;
  var b: i32 = a * *p;
  return b;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // TODO: Dereference operator `*p` requires context of whether there is a
// CHECK:STDOUT: // preceding expression, to distinguish from binary multiplication `a * b` and
// CHECK:STDOUT: // pointer type expression `i32*`.
// CHECK:STDOUT: fn F(p: i32 *) -> i32 {
// CHECK:STDOUT:   var a: i32 = * p;
// CHECK:STDOUT:   var b: i32 = a * * p;
// CHECK:STDOUT:   return b;
// CHECK:STDOUT: }
