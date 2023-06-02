// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'var.carbon'
// CHECK:STDOUT: source_filename = "var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %var = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %var, align 4
// CHECK:STDOUT:   %0 = load i32, ptr %var, align 4
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }

fn Main() -> i32 {
  var x: i32 = 0;
  return x;
}
