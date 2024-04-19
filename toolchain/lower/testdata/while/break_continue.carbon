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
// CHECK:STDOUT: entry1:
// CHECK:STDOUT:   br label %entry
// CHECK:STDOUT:
// CHECK:STDOUT: entry:                                            ; preds = %entry1, %1, %if.then.loc13
// CHECK:STDOUT:   %A = call i1 @A()
// CHECK:STDOUT:   br i1 %A, label %while.body, label %2
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %entry
// CHECK:STDOUT:   %B = call i1 @B()
// CHECK:STDOUT:   br i1 %B, label %if.then.loc13, label %0
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc13:                                    ; preds = %while.body
// CHECK:STDOUT:   br label %entry
// CHECK:STDOUT:
// CHECK:STDOUT: 0:                                                ; preds = %while.body
// CHECK:STDOUT:   %C = call i1 @C()
// CHECK:STDOUT:   br i1 %C, label %if.then.loc14, label %1
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc14:                                    ; preds = %0
// CHECK:STDOUT:   br label %2
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %0
// CHECK:STDOUT:   br label %entry
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %if.then.loc14, %entry
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %entry, { 1, 2, 0 }
// CHECK:STDOUT: }
