// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn Or() -> bool {
  return F() or G();
}

// CHECK:STDOUT: ; ModuleID = 'or.carbon'
// CHECK:STDOUT: source_filename = "or.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @F() {
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @G() {
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Or() {
// CHECK:STDOUT:   %F = call i1 @F()
// CHECK:STDOUT:   %temp = alloca i1, align 1
// CHECK:STDOUT:   store i1 %F, ptr %temp, align 1
// CHECK:STDOUT:   %1 = load i1, ptr %temp, align 1
// CHECK:STDOUT:   %2 = xor i1 %1, true
// CHECK:STDOUT:   br i1 %2, label %3, label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   %G = call i1 @G()
// CHECK:STDOUT:   %temp1 = alloca i1, align 1
// CHECK:STDOUT:   store i1 %G, ptr %temp1, align 1
// CHECK:STDOUT:   %4 = load i1, ptr %temp1, align 1
// CHECK:STDOUT:   br label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %3, %0
// CHECK:STDOUT:   %6 = phi i1 [ true, %0 ], [ %4, %3 ]
// CHECK:STDOUT:   ret i1 %6
// CHECK:STDOUT: }
