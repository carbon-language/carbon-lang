// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/declaration/simple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/lower/testdata/function/declaration/simple.carbon

fn F(n: i32);

fn G(n: i32) { F(n); }

// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @G(i32 %n) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @F(i32 %n)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
