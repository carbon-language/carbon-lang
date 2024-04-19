// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn And(b: bool, c: bool) -> bool {
  // Note that in this case, we must generate an empty block so that the `phi`
  // has two distinct predecessors.
  return b and c;
}

// CHECK:STDOUT: ; ModuleID = 'and_empty_block.carbon'
// CHECK:STDOUT: source_filename = "and_empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @And(i1 %b, i1 %c) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %and.rhs, label %and.result
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs:                                          ; preds = %entry
// CHECK:STDOUT:   br label %and.result
// CHECK:STDOUT:
// CHECK:STDOUT: and.result:                                       ; preds = %and.rhs, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %c, %and.rhs ]
// CHECK:STDOUT:   ret i1 %0
// CHECK:STDOUT: }
