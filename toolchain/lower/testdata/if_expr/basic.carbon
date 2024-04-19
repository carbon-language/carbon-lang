// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 { return 1; }
fn G() -> i32 { return 2; }

fn Select(b: bool) -> i32 {
  return if b then F() else G();
}

// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @G() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Select(i1 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.expr.then, label %0
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then:                                     ; preds = %entry
// CHECK:STDOUT:   %F = call i32 @F()
// CHECK:STDOUT:   br label %if.expr.result
// CHECK:STDOUT:
// CHECK:STDOUT: 0:                                                ; preds = %entry
// CHECK:STDOUT:   %G = call i32 @G()
// CHECK:STDOUT:   br label %if.expr.result
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result:                                   ; preds = %0, %if.expr.then
// CHECK:STDOUT:   %1 = phi i32 [ %F, %if.expr.then ], [ %G, %0 ]
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
