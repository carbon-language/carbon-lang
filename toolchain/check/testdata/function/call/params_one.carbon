// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo(a: i32) {}

fn Main() {
  Foo(1);
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "params_one.carbon" {
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <function> = name_ref Foo, file.%Foo
// CHECK:STDOUT:   %.loc10_7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_6: init () = call %Foo.ref(%.loc10_7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
