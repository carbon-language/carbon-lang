// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/definition/params_two.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/lower/testdata/function/definition/params_two.carbon

fn Foo(a: i32, b: i32) {}

// CHECK:STDOUT: ; ModuleID = 'params_two.carbon'
// CHECK:STDOUT: source_filename = "params_two.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Foo(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
