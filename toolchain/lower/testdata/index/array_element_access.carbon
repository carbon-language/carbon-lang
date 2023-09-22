// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
fn A() -> (i32, i32) { return (1, 2); }

fn B() -> [i32; 2] { return (1, 2); }

fn Run() {
  var a: [i32; 2] = A();
  var b: i32 = A()[0];
  var c: i32 = a[b];
  var d: i32 = B()[1];
}

// CHECK:STDOUT: ; ModuleID = 'array_element_access.carbon'
// CHECK:STDOUT: source_filename = "array_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @A(ptr sret({ i32, i32 }) %return) {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @B(ptr sret([2 x i32]) %return) {
// CHECK:STDOUT:   %array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %array.index, align 4
// CHECK:STDOUT:   %array.index1 = getelementptr inbounds [2 x i32], ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %array.index1, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Run() {
// CHECK:STDOUT:   %a = alloca [2 x i32], align 4
// CHECK:STDOUT:   %temp = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @A(ptr %temp)
// CHECK:STDOUT:   %array.index = getelementptr inbounds [2 x i32], ptr %a, i32 0, i32 0
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { i32, i32 }, ptr %temp, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.index, align 4
// CHECK:STDOUT:   store i32 %1, ptr %array.index, align 4
// CHECK:STDOUT:   %array.index1 = getelementptr inbounds [2 x i32], ptr %a, i32 0, i32 1
// CHECK:STDOUT:   %tuple.index2 = getelementptr inbounds { i32, i32 }, ptr %temp, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %tuple.index2, align 4
// CHECK:STDOUT:   store i32 %2, ptr %array.index1, align 4
// CHECK:STDOUT:   %b = alloca i32, align 4
// CHECK:STDOUT:   %temp3 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @A(ptr %temp3)
// CHECK:STDOUT:   %tuple.index4 = getelementptr inbounds { i32, i32 }, ptr %temp3, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %tuple.index4, align 4
// CHECK:STDOUT:   store i32 %3, ptr %b, align 4
// CHECK:STDOUT:   %c = alloca i32, align 4
// CHECK:STDOUT:   %4 = load i32, ptr %b, align 4
// CHECK:STDOUT:   %array.index5 = getelementptr inbounds [2 x i32], ptr %a, i32 0, i32 %4
// CHECK:STDOUT:   %5 = load i32, ptr %array.index5, align 4
// CHECK:STDOUT:   store i32 %5, ptr %c, align 4
// CHECK:STDOUT:   %d = alloca i32, align 4
// CHECK:STDOUT:   %temp6 = alloca [2 x i32], align 4
// CHECK:STDOUT:   call void @B(ptr %temp6)
// CHECK:STDOUT:   %array.index7 = getelementptr inbounds [2 x i32], ptr %temp6, i32 0, i32 1
// CHECK:STDOUT:   %6 = load i32, ptr %array.index7, align 4
// CHECK:STDOUT:   store i32 %6, ptr %d, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
