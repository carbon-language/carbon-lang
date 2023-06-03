// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'params_two_comma.carbon'
// CHECK:STDOUT: source_filename = "params_two_comma.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Foo(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Foo = call %EmptyTupleType @Foo(i32 1, i32 2)
// CHECK:STDOUT:   %Foo1 = call %EmptyTupleType @Foo(i32 1, i32 2)
// CHECK:STDOUT: }

fn Foo(a: i32, b: i32,) {}

fn Main() {
  Foo(1, 2);
  Foo(1, 2,);
}
