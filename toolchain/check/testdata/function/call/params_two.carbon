// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo(a: i32, b: i32) {}

fn Main() {
  Foo(1, 2);
}

// CHECK:STDOUT: file "params_two.carbon" {
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32, %b: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <function> = name_reference "Foo", package.%Foo
// CHECK:STDOUT:   %.loc10_7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_10: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_6.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc10_6.2: init () = call %Foo.ref(%.loc10_7, %.loc10_10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
