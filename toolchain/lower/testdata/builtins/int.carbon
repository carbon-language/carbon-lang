// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Add(a: i32, b: i32) -> i32 = "int.add";
fn TestAdd(a: i32, b: i32) -> i32 { return Add(a, b); }

// CHECK:STDOUT: ; ModuleID = 'int.carbon'
// CHECK:STDOUT: source_filename = "int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestAdd(i32 %a, i32 %b) {
// CHECK:STDOUT:   %add = add i32 %a, %b
// CHECK:STDOUT:   ret i32 %add
// CHECK:STDOUT: }
