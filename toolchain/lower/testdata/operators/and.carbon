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
// CHECK:STDOUT:   br i1 %F, label %1, label %2
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %0
// CHECK:STDOUT:   %G = call i1 @G()
// CHECK:STDOUT:   br label %2
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %1, %0
// CHECK:STDOUT:   %3 = phi i1 [ false, %0 ], [ %G, %1 ]
// CHECK:STDOUT:   ret i1 %3
// CHECK:STDOUT: }
