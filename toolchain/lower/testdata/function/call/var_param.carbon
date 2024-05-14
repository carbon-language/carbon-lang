// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn DoNothing(a: i32) {}

fn Main() {
  var a: i32 = 0;
  DoNothing(a);
}

// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @DoNothing(i32 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %a.var, align 4
// CHECK:STDOUT:   %.loc11 = load i32, ptr %a.var, align 4
// CHECK:STDOUT:   call void @DoNothing(i32 %.loc11)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
