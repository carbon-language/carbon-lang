// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn And() -> bool {
  return F() and G();
}

// CHECK:STDOUT: ; ModuleID = 'and.carbon'
// CHECK:STDOUT: source_filename = "and.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @F() {
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @G() {
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @And() {
// CHECK:STDOUT:   %F = call i1 @F()
// CHECK:STDOUT:   %temp = alloca i1, align 1
// CHECK:STDOUT:   store i1 %F, ptr %temp, align 1
// CHECK:STDOUT:   %1 = load i1, ptr %temp, align 1
// CHECK:STDOUT:   br i1 %1, label %2, label %4
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %G = call i1 @G()
// CHECK:STDOUT:   %temp1 = alloca i1, align 1
// CHECK:STDOUT:   store i1 %G, ptr %temp1, align 1
// CHECK:STDOUT:   %3 = load i1, ptr %temp1, align 1
// CHECK:STDOUT:   br label %4
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %2, %0
// CHECK:STDOUT:   %5 = phi i1 [ false, %0 ], [ %3, %2 ]
// CHECK:STDOUT:   ret i1 %5
// CHECK:STDOUT: }
