// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/definition/empty_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/lower/testdata/function/definition/empty_struct.carbon

fn Echo(a: {}) {
}

// CHECK:STDOUT: ; ModuleID = 'empty_struct.carbon'
// CHECK:STDOUT: source_filename = "empty_struct.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Echo() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
