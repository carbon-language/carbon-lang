// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/i32.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/lower/testdata/function/call/i32.carbon

fn Echo(a: i32) -> i32 {
  return a;
}

fn Main() {
  var b: i32 = Echo(1);
}

// CHECK:STDOUT: ; ModuleID = 'i32.carbon'
// CHECK:STDOUT: source_filename = "i32.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Echo(i32 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca i32, align 4
// CHECK:STDOUT:   %Echo.call = call i32 @Echo(i32 1)
// CHECK:STDOUT:   store i32 %Echo.call, ptr %b.var, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
