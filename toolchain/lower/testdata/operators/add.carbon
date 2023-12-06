// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Add_i32(b: i32, c: i32) -> i32 {
  return b + c;
}

// CHECK:STDOUT: ; ModuleID = 'add.carbon'
// CHECK:STDOUT: source_filename = "add.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Add_i32(i32 %b, i32 %c) {
// CHECK:STDOUT:   %1 = add i32 %b, %c
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
