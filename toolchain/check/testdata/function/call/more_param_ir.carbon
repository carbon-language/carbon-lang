// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo(a: i32, b: i32, c: i32) {}

fn Main() {
  // Generates multiple IR instructions for the first two parameters.
  Foo(1 + 2 + 3, 4 + 5, 6);
}

// CHECK:STDOUT: file "more_param_ir.carbon" {
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32, %b: i32, %c: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <function> = name_reference "Foo", package.%Foo
// CHECK:STDOUT:   %.loc11_7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc11_11: i32 = int_literal 2
// CHECK:STDOUT:   %.loc11_9: i32 = add %.loc11_7, %.loc11_11
// CHECK:STDOUT:   %.loc11_15: i32 = int_literal 3
// CHECK:STDOUT:   %.loc11_13: i32 = add %.loc11_9, %.loc11_15
// CHECK:STDOUT:   %.loc11_18: i32 = int_literal 4
// CHECK:STDOUT:   %.loc11_22: i32 = int_literal 5
// CHECK:STDOUT:   %.loc11_20: i32 = add %.loc11_18, %.loc11_22
// CHECK:STDOUT:   %.loc11_25: i32 = int_literal 6
// CHECK:STDOUT:   %.loc11_6.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc11_6.2: init () = call %Foo.ref(%.loc11_13, %.loc11_20, %.loc11_25)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
