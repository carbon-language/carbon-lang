// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  var t: [i32; 2] = F();
}

// CHECK:STDOUT: ; ModuleID = 'assign_return_value.carbon'
// CHECK:STDOUT: source_filename = "assign_return_value.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F(ptr sret({ i32, i32 }) %return) {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 24, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() {
// CHECK:STDOUT:   %t = alloca [2 x i32], align 4
// CHECK:STDOUT:   %temp = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @F(ptr %temp)
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32 }, ptr %temp, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %array.index = getelementptr inbounds [2 x i32], ptr %t, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %array.index, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32 }, ptr %temp, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   %array.index2 = getelementptr inbounds [2 x i32], ptr %t, i32 0, i32 1
// CHECK:STDOUT:   store i32 %2, ptr %array.index2, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
