// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/params_one_comma.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/params_one_comma.carbon

fn Foo(a: i32,) {}

fn Main() {
  Foo(1);
  Foo(1,);
}

// CHECK:STDOUT: ; ModuleID = 'params_one_comma.carbon'
// CHECK:STDOUT: source_filename = "params_one_comma.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Foo(i32 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @Foo(i32 1)
// CHECK:STDOUT:   call void @Foo(i32 1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
