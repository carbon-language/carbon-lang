// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 { return 1; }
fn G() -> i32 { return 2; }

fn Select(b: bool) -> i32 {
  return if b then F() else G();
}

// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT:   ret i32 1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @G() {
// CHECK:STDOUT:   ret i32 2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Select(i1 %b) {
// CHECK:STDOUT:   br i1 %b, label %1, label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %0
// CHECK:STDOUT:   %F = call i32 @F()
// CHECK:STDOUT:   %temp = alloca i32, align 4
// CHECK:STDOUT:   store i32 %F, ptr %temp, align 4
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4
// CHECK:STDOUT:   br label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   %G = call i32 @G()
// CHECK:STDOUT:   %temp1 = alloca i32, align 4
// CHECK:STDOUT:   store i32 %G, ptr %temp1, align 4
// CHECK:STDOUT:   %4 = load i32, ptr %temp1, align 4
// CHECK:STDOUT:   br label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %3, %1
// CHECK:STDOUT:   %6 = phi i32 [ %2, %1 ], [ %4, %3 ]
// CHECK:STDOUT:   ret i32 %6
// CHECK:STDOUT: }
