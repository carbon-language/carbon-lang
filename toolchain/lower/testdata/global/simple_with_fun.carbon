// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/global/simple_with_fun.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/global/simple_with_fun.carbon

fn test_a() -> i32 {
  return 0;
}

var a: i32 = test_a();

// CHECK:STDOUT: ; ModuleID = 'simple_with_fun.carbon'
// CHECK:STDOUT: source_filename = "simple_with_fun.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @a = internal global i32
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @__global_init, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @test_a() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @__global_init() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %test_a.call = call i32 @test_a()
// CHECK:STDOUT:   store i32 %test_a.call, ptr @a, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 0, { 1, 0 }
