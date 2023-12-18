// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Cond() -> bool;

fn F();
fn G();
fn H();

fn While() {
  F();
  while (Cond()) {
    G();
  }
  H();
}

// CHECK:STDOUT: ; ModuleID = 'while.carbon'
// CHECK:STDOUT: source_filename = "while.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @Cond()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @H()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @While() {
// CHECK:STDOUT:   call void @F()
// CHECK:STDOUT:   br label %1
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %2, %0
// CHECK:STDOUT:   %Cond = call i1 @Cond()
// CHECK:STDOUT:   br i1 %Cond, label %2, label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %1
// CHECK:STDOUT:   call void @G()
// CHECK:STDOUT:   br label %1
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %1
// CHECK:STDOUT:   call void @H()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
