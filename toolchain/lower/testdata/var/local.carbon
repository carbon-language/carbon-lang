// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: i32 = 1;
  return x;
}

// CHECK:STDOUT: ; ModuleID = 'local.carbon'
// CHECK:STDOUT: source_filename = "local.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4
// CHECK:STDOUT:   store i32 1, ptr %x.var, align 4
// CHECK:STDOUT:   %.loc9 = load i32, ptr %x.var, align 4
// CHECK:STDOUT:   ret i32 %.loc9
// CHECK:STDOUT: }
