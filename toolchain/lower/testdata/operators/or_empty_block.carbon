// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/or_empty_block.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/lower/testdata/operators/or_empty_block.carbon

fn Or(b: bool, c: bool) -> bool {
  // Note that in this case, we must generate an empty block so that the `phi`
  // has two distinct predecessors.
  return b or c;
}

// CHECK:STDOUT: ; ModuleID = 'or_empty_block.carbon'
// CHECK:STDOUT: source_filename = "or_empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Or(i1 %b, i1 %c) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_12.1 = xor i1 %b, true
// CHECK:STDOUT:   br i1 %.loc14_12.1, label %or.rhs, label %or.result
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %entry
// CHECK:STDOUT:   br label %or.result
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %or.rhs, %entry
// CHECK:STDOUT:   %0 = phi i1 [ true, %entry ], [ %c, %or.rhs ]
// CHECK:STDOUT:   ret i1 %0
// CHECK:STDOUT: }
