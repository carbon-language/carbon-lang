// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "more_param_ir.carbon" {

fn Foo(a: i32, b: i32, c: i32) {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Foo

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32, %b: i32, %c: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  // Generates multiple IR instructions for the first two parameters.
  Foo(1 + 2 + 3, 4 + 5, 6);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_7: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_11: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_9: i32 = add %.loc[[@LINE-3]]_7, %.loc[[@LINE-3]]_11
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_15: i32 = int_literal 3
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_13.1: i32 = add %.loc[[@LINE-5]]_9, %.loc[[@LINE-5]]_15
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_13.2: i32 = stub_reference %.loc[[@LINE-6]]_13.1
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_18: i32 = int_literal 4
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_22: i32 = int_literal 5
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_20.1: i32 = add %.loc[[@LINE-9]]_18, %.loc[[@LINE-9]]_22
  // CHECK:STDOUT:   %.loc[[@LINE-10]]_20.2: i32 = stub_reference %.loc[[@LINE-10]]_20.1
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_25.1: i32 = int_literal 6
  // CHECK:STDOUT:   %.loc[[@LINE-12]]_25.2: i32 = stub_reference %.loc[[@LINE-12]]_25.1
  // CHECK:STDOUT:   %.loc[[@LINE-13]]_6.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-14]]_6.2: init () = call @Foo(%.loc[[@LINE-14]]_13.2, %.loc[[@LINE-14]]_20.2, %.loc[[@LINE-14]]_25.2)
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
