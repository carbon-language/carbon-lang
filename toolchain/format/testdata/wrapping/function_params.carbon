// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/function_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/function_params.carbon

// --- test.carbon

fn RegisterHandler(event_name: String, priority: i32, callback: Callback) -> bool {
  return true;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn RegisterHandler(event_name: String, priority: i32,
// CHECK:STDOUT:                    callback: Callback) -> bool {
// CHECK:STDOUT:   return true;
// CHECK:STDOUT: }
