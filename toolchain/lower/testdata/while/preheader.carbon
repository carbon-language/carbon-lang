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
// CHECK:STDOUT: 0:                                                ; preds = %entry, %2
// CHECK:STDOUT:   %Cond = call i1 @Cond()
// CHECK:STDOUT:   %temp = alloca i1, align 1
// CHECK:STDOUT:   store i1 %Cond, ptr %temp, align 1
// CHECK:STDOUT:   %1 = load i1, ptr %temp, align 1
// CHECK:STDOUT:   br i1 %1, label %2, label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   call void @F()
// CHECK:STDOUT:   br label %0
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   %Cond1 = call i1 @Cond()
// CHECK:STDOUT:   %temp2 = alloca i1, align 1
// CHECK:STDOUT:   store i1 %Cond1, ptr %temp2, align 1
// CHECK:STDOUT:   %4 = load i1, ptr %temp2, align 1
// CHECK:STDOUT:   br i1 %4, label %5, label %9
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %7, %3
// CHECK:STDOUT:   %Cond3 = call i1 @Cond()
// CHECK:STDOUT:   %temp4 = alloca i1, align 1
// CHECK:STDOUT:   store i1 %Cond3, ptr %temp4, align 1
// CHECK:STDOUT:   %6 = load i1, ptr %temp4, align 1
// CHECK:STDOUT:   br i1 %6, label %7, label %8
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %5
// CHECK:STDOUT:   call void @G()
// CHECK:STDOUT:   br label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %5
// CHECK:STDOUT:   br label %9
// CHECK:STDOUT:
// CHECK:STDOUT: 9:                                                ; preds = %8, %3
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %0, { 1, 0 }
// CHECK:STDOUT: }
