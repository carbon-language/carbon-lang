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
// CHECK:STDOUT:   br i1 %b, label %1, label %6
// CHECK:STDOUT:
// CHECK:STDOUT: 1:                                                ; preds = %0
// CHECK:STDOUT:   br i1 %c, label %2, label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %1
// CHECK:STDOUT:   br label %4
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %1
// CHECK:STDOUT:   br label %4
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %3, %2
// CHECK:STDOUT:   %5 = phi i32 [ 1, %2 ], [ 2, %3 ]
// CHECK:STDOUT:   br label %11
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   br i1 %d, label %7, label %8
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %6
// CHECK:STDOUT:   br label %9
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %6
// CHECK:STDOUT:   br label %9
// CHECK:STDOUT:
// CHECK:STDOUT: 9:                                                ; preds = %8, %7
// CHECK:STDOUT:   %10 = phi i32 [ 3, %7 ], [ 4, %8 ]
// CHECK:STDOUT:   br label %11
// CHECK:STDOUT:
// CHECK:STDOUT: 11:                                               ; preds = %9, %4
// CHECK:STDOUT:   %12 = phi i32 [ %5, %4 ], [ %10, %9 ]
// CHECK:STDOUT:   ret i32 %12
// CHECK:STDOUT: }
