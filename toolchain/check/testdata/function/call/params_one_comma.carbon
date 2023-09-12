// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "params_one_comma.carbon" {

fn Foo(a: i32,) {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Foo

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  Foo(1);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_7.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_7.2: i32 = stub_reference %.loc[[@LINE-2]]_7.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_6.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_6.2: init () = call @Foo(%.loc[[@LINE-4]]_7.2)
  Foo(1,);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_7.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_7.2: i32 = stub_reference %.loc[[@LINE-2]]_7.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_6: init () = call @Foo(%.loc[[@LINE-3]]_7.2)
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
