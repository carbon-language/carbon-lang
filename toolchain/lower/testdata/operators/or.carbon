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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @G() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Or() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i1 @F()
// CHECK:STDOUT:   %.loc11_14.3 = xor i1 %F.call, true
// CHECK:STDOUT:   br i1 %.loc11_14.3, label %or.rhs, label %or.result
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %entry
// CHECK:STDOUT:   %G.call = call i1 @G()
// CHECK:STDOUT:   br label %or.result
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %or.rhs, %entry
// CHECK:STDOUT:   %0 = phi i1 [ true, %entry ], [ %G.call, %or.rhs ]
// CHECK:STDOUT:   ret i1 %0
// CHECK:STDOUT: }
