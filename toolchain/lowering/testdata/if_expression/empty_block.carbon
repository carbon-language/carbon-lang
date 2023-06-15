// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'empty_block.carbon'
// CHECK:STDOUT: source_filename = "empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Select(i1 %b, i1 %c, i1 %d) {
// CHECK:STDOUT:   br i1 %b, label %1, label %2
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %0
// CHECK:STDOUT:   br i1 %c, label %3, label %4
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   br i1 %d, label %9, label %10
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %1
// CHECK:STDOUT:   br label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %1
// CHECK:STDOUT:   br label %5
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %4, %3
// CHECK:STDOUT:   %6 = phi i32 [ 1, %3 ], [ 2, %4 ]
// CHECK:STDOUT:   br label %7
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %11, %5
// CHECK:STDOUT:   %8 = phi i32 [ %6, %5 ], [ %12, %11 ]
// CHECK:STDOUT:   ret i32 %8
// CHECK:STDOUT:
// CHECK:STDOUT: 9:                                                ; preds = %2
// CHECK:STDOUT:   br label %11
// CHECK:STDOUT:
// CHECK:STDOUT: 10:                                               ; preds = %2
// CHECK:STDOUT:   br label %11
// CHECK:STDOUT:
// CHECK:STDOUT: 11:                                               ; preds = %10, %9
// CHECK:STDOUT:   %12 = phi i32 [ 3, %9 ], [ 4, %10 ]
// CHECK:STDOUT:   br label %7
// CHECK:STDOUT: }

fn Select(b: bool, c: bool, d: bool) -> i32 {
  return if b then if c then 1 else 2 else if d then 3 else 4;
}
