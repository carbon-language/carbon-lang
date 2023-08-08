// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Or(b: bool, c: bool) -> bool {
  // Note that in this case, we must generate an empty block so that the `phi`
  // has two distinct predecessors.
  return b or c;
}

// CHECK:STDOUT: ; ModuleID = 'or_empty_block.carbon'
// CHECK:STDOUT: source_filename = "or_empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Or(i1 %b, i1 %c) {
// CHECK:STDOUT:   %1 = xor i1 %b, true
// CHECK:STDOUT:   br i1 %1, label %2, label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   br label %3
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %2, %0
// CHECK:STDOUT:   %4 = phi i1 [ true, %0 ], [ %c, %2 ]
// CHECK:STDOUT:   ret i1 %4
// CHECK:STDOUT: }
