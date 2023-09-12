// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "params_zero.carbon" {

fn Foo() {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Foo

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  Foo();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_6.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_6.2: init () = call @Foo()
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
