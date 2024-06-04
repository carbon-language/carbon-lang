// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/basic.carbon

class C {
  var a: i32;
  var b: C*;
}

fn F(c: C) -> C;

fn Run() {
  var c: C;
  var d: C = F(c);
}

// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F(ptr sret({ i32, ptr }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, ptr }, align 8
// CHECK:STDOUT:   %d.var = alloca { i32, ptr }, align 8
// CHECK:STDOUT:   call void @F(ptr %d.var, ptr %c.var)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
