// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn Or() -> bool {
  return F() or G();
}

// CHECK:STDOUT: ; ModuleID = 'or.carbon'
// CHECK:STDOUT: source_filename = "or.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @F() {
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @G() {
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Or() {
// CHECK:STDOUT:   %F = call i1 @F()
// CHECK:STDOUT:   %1 = xor i1 %F, true
// CHECK:STDOUT:   br i1 %1, label %2, label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %G = call i1 @G()
// CHECK:STDOUT:   br label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %2, %0
// CHECK:STDOUT:   %4 = phi i1 [ true, %0 ], [ %G, %2 ]
// CHECK:STDOUT:   ret i1 %4
// CHECK:STDOUT: }
