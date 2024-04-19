// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  var a: i32 = 12;
  a = 9;
  var b: (i32, i32);
  b = (1, 2);
}

// CHECK:STDOUT: ; ModuleID = 'assignment.carbon'
// CHECK:STDOUT: source_filename = "assignment.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4
// CHECK:STDOUT:   store i32 12, ptr %a.var, align 4
// CHECK:STDOUT:   store i32 9, ptr %a.var, align 4
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc11_12.2.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %.loc11_12.2.tuple.elem, align 4
// CHECK:STDOUT:   %.loc11_12.4.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %.loc11_12.4.tuple.elem, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
