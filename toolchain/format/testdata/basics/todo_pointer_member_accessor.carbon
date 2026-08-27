// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/todo_pointer_member_accessor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/todo_pointer_member_accessor.carbon

// --- test.carbon

// TODO: Pointer member access `obj->property` should be packed without spaces,
// while function return type `fn F() -> T` should have spaces around `->`.
fn F(p: Point*) -> i32 {
  var x: i32 = p -> x;
  var y: i32 = p -> y;
  return x + y;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // TODO: Pointer member access `obj->property` should be packed without spaces,
// CHECK:STDOUT: // while function return type `fn F() -> T` should have spaces around `->`.
// CHECK:STDOUT: fn F(p: Point *) -> i32 {
// CHECK:STDOUT:   var x: i32 = p -> x;
// CHECK:STDOUT:   var y: i32 = p -> y;
// CHECK:STDOUT:   return x + y;
// CHECK:STDOUT: }
