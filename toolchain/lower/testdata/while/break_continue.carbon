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
// CHECK:STDOUT: 0:                                                ; preds = %entry, %5, %2
// CHECK:STDOUT:   %A = call i1 @A()
// CHECK:STDOUT:   br i1 %A, label %1, label %6
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %0
// CHECK:STDOUT:   %B = call i1 @B()
// CHECK:STDOUT:   br i1 %B, label %2, label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %1
// CHECK:STDOUT:   br label %0
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %1
// CHECK:STDOUT:   %C = call i1 @C()
// CHECK:STDOUT:   br i1 %C, label %4, label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %3
// CHECK:STDOUT:   br label %6
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %3
// CHECK:STDOUT:   br label %0
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %4, %0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %0, { 1, 2, 0 }
// CHECK:STDOUT: }
