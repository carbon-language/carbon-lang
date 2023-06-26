// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @DoNothing(i32 %a) {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Main() {
// CHECK:STDOUT:   %var = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %var, align 4
// CHECK:STDOUT:   %1 = load i32, ptr %var, align 4
// CHECK:STDOUT:   %DoNothing = call %EmptyTupleType @DoNothing(i32 %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn DoNothing(a: i32) {}

fn Main() {
  var a: i32 = 0;
  DoNothing(a);
}
