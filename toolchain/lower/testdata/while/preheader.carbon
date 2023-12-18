// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Cond() -> bool;

fn F();
fn G();

// TODO: It would be better to preserve the loop preheader blocks from SemIR
// into LLVM IR. If we don't, LLVM will recreate them for us as part of loop
// canonicalization.
fn While() {
  while (Cond()) {
    F();
  }

  if (Cond()) {
    while (Cond()) {
      G();
    }
  }
}

// CHECK:STDOUT: ; ModuleID = 'preheader.carbon'
// CHECK:STDOUT: source_filename = "preheader.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @Cond()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @While() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %0
// CHECK:STDOUT:
// CHECK:STDOUT: 0:                                                ; preds = %entry, %1
// CHECK:STDOUT:   %Cond = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond, label %1, label %2
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %0
// CHECK:STDOUT:   call void @F()
// CHECK:STDOUT:   br label %0
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %Cond1 = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond1, label %3, label %6
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %4, %2
// CHECK:STDOUT:   %Cond2 = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond2, label %4, label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %3
// CHECK:STDOUT:   call void @G()
// CHECK:STDOUT:   br label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %3
// CHECK:STDOUT:   br label %6
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %5, %2
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %0, { 1, 0 }
// CHECK:STDOUT: }
