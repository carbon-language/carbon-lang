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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @G() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @And() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i1 @F()
// CHECK:STDOUT:   br i1 %F.call, label %and.rhs, label %and.result
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs:                                          ; preds = %entry
// CHECK:STDOUT:   %G.call = call i1 @G()
// CHECK:STDOUT:   br label %and.result
// CHECK:STDOUT:
// CHECK:STDOUT: and.result:                                       ; preds = %and.rhs, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %G.call, %and.rhs ]
// CHECK:STDOUT:   ret i1 %0
// CHECK:STDOUT: }
