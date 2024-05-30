// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/nested_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/lower/testdata/tuple/nested_tuple.carbon

fn Run() -> i32 {
  ((i32,), );
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'nested_tuple.carbon'
// CHECK:STDOUT: source_filename = "nested_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
