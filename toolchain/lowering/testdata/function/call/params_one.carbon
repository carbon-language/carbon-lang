// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'params_one.carbon'
// CHECK:STDOUT: source_filename = "params_one.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Foo(i32 %a) {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Main() {
// CHECK:STDOUT:   %Foo = call %EmptyTupleType @Foo(i32 1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn Foo(a: i32) {}

fn Main() {
  Foo(1);
}
