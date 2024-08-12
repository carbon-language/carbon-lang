// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/field.carbon

class C {
  var a: i32;
  var b: C*;
}

fn F(c: C) -> i32 {
  return (*c.b).a;
}

fn Run() -> i32 {
  var c: C;
  c.a = 1;
  c.b = &c;
  return F(c);
}

// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %c) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %c, i32 0, i32 1
// CHECK:STDOUT:   %.loc17_13.2 = load ptr, ptr %.loc17_13.1.b, align 8
// CHECK:STDOUT:   %.loc17_16.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %.loc17_13.2, i32 0, i32 0
// CHECK:STDOUT:   %.loc17_16.2 = load i32, ptr %.loc17_16.1.a, align 4
// CHECK:STDOUT:   ret i32 %.loc17_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, ptr }, align 8
// CHECK:STDOUT:   %.loc22_4.a = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %.loc22_4.a, align 4
// CHECK:STDOUT:   %.loc23_4.b = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 1
// CHECK:STDOUT:   store ptr %c.var, ptr %.loc23_4.b, align 8
// CHECK:STDOUT:   %F.call = call i32 @F(ptr %c.var)
// CHECK:STDOUT:   ret i32 %F.call
// CHECK:STDOUT: }
