// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/not.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/not.carbon

fn Not(b: bool) -> bool {
  return not b;
}

// CHECK:STDOUT: ; ModuleID = 'not.carbon'
// CHECK:STDOUT: source_filename = "not.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Not(i1 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12 = xor i1 %b, true
// CHECK:STDOUT:   ret i1 %.loc12
// CHECK:STDOUT: }
