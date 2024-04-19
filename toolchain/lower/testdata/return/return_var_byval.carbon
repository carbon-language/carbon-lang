// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  returned var x: i32 = 0;
  return var;
}

// CHECK:STDOUT: ; ModuleID = 'return_var_byval.carbon'
// CHECK:STDOUT: source_filename = "return_var_byval.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %x.var, align 4
// CHECK:STDOUT:   %.loc8_16 = load i32, ptr %x.var, align 4
// CHECK:STDOUT:   ret i32 %.loc8_16
// CHECK:STDOUT: }
