// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

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
// CHECK:STDOUT:   %c = alloca { i32, ptr }, align 8
// CHECK:STDOUT:   %d = alloca { i32, ptr }, align 8
// CHECK:STDOUT:   call void @F(ptr %d, ptr %c)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
