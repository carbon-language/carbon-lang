// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/pointer/pointer_to_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/pointer/pointer_to_pointer.carbon

fn F(p: i32**) -> i32 {
  var a: i32** = p;
  var b: i32* = *p;
  var c: i32** = &b;
  return **c;
}

// CHECK:STDOUT: ; ModuleID = 'pointer_to_pointer.carbon'
// CHECK:STDOUT: source_filename = "pointer_to_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %p) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %p, ptr %a.var, align 8
// CHECK:STDOUT:   %b.var = alloca ptr, align 8
// CHECK:STDOUT:   %.loc13_17.2 = load ptr, ptr %p, align 8
// CHECK:STDOUT:   store ptr %.loc13_17.2, ptr %b.var, align 8
// CHECK:STDOUT:   %c.var = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %b.var, ptr %c.var, align 8
// CHECK:STDOUT:   %.loc15_12 = load ptr, ptr %c.var, align 8
// CHECK:STDOUT:   %.loc15_11.2 = load ptr, ptr %.loc15_12, align 8
// CHECK:STDOUT:   %.loc15_10.2 = load i32, ptr %.loc15_11.2, align 4
// CHECK:STDOUT:   ret i32 %.loc15_10.2
// CHECK:STDOUT: }
