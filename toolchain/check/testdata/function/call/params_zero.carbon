// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo() {}

fn Main() {
  Foo();
}

// CHECK:STDOUT: file "params_zero.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @Foo
// CHECK:STDOUT:   %.loc9 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_6.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc10_6.2: init () = call @Foo()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
