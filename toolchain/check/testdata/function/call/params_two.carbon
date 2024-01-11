// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo(a: i32, b: i32) {}

fn Main() {
  Foo(1, 2);
}

// CHECK:STDOUT: --- params_two.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_7: i32 = int_literal 1, const
// CHECK:STDOUT:   %.loc10_10: i32 = int_literal 2, const
// CHECK:STDOUT:   %.loc10_6: type = tuple_type (), const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Foo = %Foo, .Main = %Main}
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo, const
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32, %b: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <function> = name_ref Foo, file.%Foo, const = file.%Foo
// CHECK:STDOUT:   %.loc10_7: i32 = int_literal 1, const = constants.%.loc10_7
// CHECK:STDOUT:   %.loc10_10: i32 = int_literal 2, const = constants.%.loc10_10
// CHECK:STDOUT:   %.loc10_6: init () = call %Foo.ref(%.loc10_7, %.loc10_10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
