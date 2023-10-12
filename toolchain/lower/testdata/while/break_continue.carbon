// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> bool;
fn B() -> bool;
fn C() -> bool;

fn While() {
  while (A()) {
    if (B()) { continue; }
    if (C()) { break; }
  }
}

// CHECK:STDOUT: ; ModuleID = 'break_continue.carbon'
// CHECK:STDOUT: source_filename = "break_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @A()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @B()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @C()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @While() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %0
// CHECK:STDOUT:
// CHECK:STDOUT: 0:                                                ; preds = %entry, %8, %4
// CHECK:STDOUT:   %A = call i1 @A()
// CHECK:STDOUT:   %temp = alloca i1, align 1
// CHECK:STDOUT:   store i1 %A, ptr %temp, align 1
// CHECK:STDOUT:   %1 = load i1, ptr %temp, align 1
// CHECK:STDOUT:   br i1 %1, label %2, label %9
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %B = call i1 @B()
// CHECK:STDOUT:   %temp1 = alloca i1, align 1
// CHECK:STDOUT:   store i1 %B, ptr %temp1, align 1
// CHECK:STDOUT:   %3 = load i1, ptr %temp1, align 1
// CHECK:STDOUT:   br i1 %3, label %4, label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %2
// CHECK:STDOUT:   br label %0
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %2
// CHECK:STDOUT:   %C = call i1 @C()
// CHECK:STDOUT:   %temp2 = alloca i1, align 1
// CHECK:STDOUT:   store i1 %C, ptr %temp2, align 1
// CHECK:STDOUT:   %6 = load i1, ptr %temp2, align 1
// CHECK:STDOUT:   br i1 %6, label %7, label %8
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %5
// CHECK:STDOUT:   br label %9
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %5
// CHECK:STDOUT:   br label %0
// CHECK:STDOUT:
// CHECK:STDOUT: 9:                                                ; preds = %7, %0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %0, { 1, 2, 0 }
// CHECK:STDOUT: }
