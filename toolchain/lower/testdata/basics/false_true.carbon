// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/basics/false_true.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/lower/testdata/basics/false_true.carbon

fn F() -> bool {
  return false;
}

fn T() -> bool {
  return true;
}

// CHECK:STDOUT: ; ModuleID = 'false_true.carbon'
// CHECK:STDOUT: source_filename = "false_true.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 false
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @T() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
