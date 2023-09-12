// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "params_two.carbon" {

fn Foo(a: i32, b: i32) {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Foo

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32, %b: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  Foo(1, 2);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_7.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_7.2: i32 = stub_reference %.loc[[@LINE-2]]_7.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_10.1: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_10.2: i32 = stub_reference %.loc[[@LINE-4]]_10.1
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_6.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_6.2: init () = call @Foo(%.loc[[@LINE-6]]_7.2, %.loc[[@LINE-6]]_10.2)
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
