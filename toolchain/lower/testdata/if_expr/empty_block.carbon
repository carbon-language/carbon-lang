// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Select(b: bool, c: bool, d: bool) -> i32 {
  return if b then if c then 1 else 2 else if d then 3 else 4;
}

// CHECK:STDOUT: ; ModuleID = 'empty_block.carbon'
// CHECK:STDOUT: source_filename = "empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Select(i1 %b, i1 %c, i1 %d) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.expr.then.loc8_10, label %if.expr.else.loc8_10
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc8_10:                             ; preds = %entry
// CHECK:STDOUT:   br i1 %c, label %if.expr.then.loc8_20, label %if.expr.else.loc8_20
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc8_20:                             ; preds = %if.expr.then.loc8_10
// CHECK:STDOUT:   br label %if.expr.result.loc8_20
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc8_20:                             ; preds = %if.expr.then.loc8_10
// CHECK:STDOUT:   br label %if.expr.result.loc8_20
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc8_20:                           ; preds = %if.expr.else.loc8_20, %if.expr.then.loc8_20
// CHECK:STDOUT:   %0 = phi i32 [ 1, %if.expr.then.loc8_20 ], [ 2, %if.expr.else.loc8_20 ]
// CHECK:STDOUT:   br label %if.expr.result.loc8_10
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc8_10:                             ; preds = %entry
// CHECK:STDOUT:   br i1 %d, label %if.expr.then.loc8_44, label %if.expr.else.loc8_44
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc8_44:                             ; preds = %if.expr.else.loc8_10
// CHECK:STDOUT:   br label %if.expr.result.loc8_44
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc8_44:                             ; preds = %if.expr.else.loc8_10
// CHECK:STDOUT:   br label %if.expr.result.loc8_44
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc8_44:                           ; preds = %if.expr.else.loc8_44, %if.expr.then.loc8_44
// CHECK:STDOUT:   %1 = phi i32 [ 3, %if.expr.then.loc8_44 ], [ 4, %if.expr.else.loc8_44 ]
// CHECK:STDOUT:   br label %if.expr.result.loc8_10
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc8_10:                           ; preds = %if.expr.result.loc8_44, %if.expr.result.loc8_20
// CHECK:STDOUT:   %2 = phi i32 [ %0, %if.expr.result.loc8_20 ], [ %1, %if.expr.result.loc8_44 ]
// CHECK:STDOUT:   ret i32 %2
// CHECK:STDOUT: }
