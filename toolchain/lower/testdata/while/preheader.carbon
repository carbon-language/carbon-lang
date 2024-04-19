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
// CHECK:STDOUT: entry1:
// CHECK:STDOUT:   br label %entry
// CHECK:STDOUT:
// CHECK:STDOUT: entry:                                            ; preds = %entry1, %while.body.loc16
// CHECK:STDOUT:   %Cond.loc16 = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond.loc16, label %while.body.loc16, label %0
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc16:                                 ; preds = %entry
// CHECK:STDOUT:   call void @F()
// CHECK:STDOUT:   br label %entry
// CHECK:STDOUT:
// CHECK:STDOUT: 0:                                                ; preds = %entry
// CHECK:STDOUT:   %Cond.loc20 = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond.loc20, label %if.then, label %2
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body.loc21, %0
// CHECK:STDOUT:   %Cond.loc21 = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond.loc21, label %while.body.loc21, label %1
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc21:                                 ; preds = %if.then
// CHECK:STDOUT:   call void @G()
// CHECK:STDOUT:   br label %if.then
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %if.then
// CHECK:STDOUT:   br label %2
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %1, %0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %entry, { 1, 0 }
// CHECK:STDOUT: }
