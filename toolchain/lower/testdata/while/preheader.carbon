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
// CHECK:STDOUT:   br label %while.cond.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond.loc16:                                 ; preds = %entry, %while.body.loc16
// CHECK:STDOUT:   %Cond.call.loc16 = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond.call.loc16, label %while.body.loc16, label %while.done.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc16:                                 ; preds = %while.cond.loc16
// CHECK:STDOUT:   call void @F()
// CHECK:STDOUT:   br label %while.cond.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: while.done.loc16:                                 ; preds = %while.cond.loc16
// CHECK:STDOUT:   %Cond.call.loc20 = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond.call.loc20, label %while.cond.loc21, label %if.else
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond.loc21:                                 ; preds = %while.body.loc21, %while.done.loc16
// CHECK:STDOUT:   %Cond.call.loc21 = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond.call.loc21, label %while.body.loc21, label %while.done.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc21:                                 ; preds = %while.cond.loc21
// CHECK:STDOUT:   call void @G()
// CHECK:STDOUT:   br label %while.cond.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: while.done.loc21:                                 ; preds = %while.cond.loc21
// CHECK:STDOUT:   br label %if.else
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.done.loc21, %while.done.loc16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %while.cond.loc16, { 1, 0 }
// CHECK:STDOUT: }
