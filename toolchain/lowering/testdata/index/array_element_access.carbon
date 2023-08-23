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
// CHECK:STDOUT:   %tuple = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %1, align 4
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %2, align 4
// CHECK:STDOUT:   %3 = load { i32, i32 }, ptr %tuple, align 4
// CHECK:STDOUT:   store { i32, i32 } %3, ptr %return, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @B(ptr sret([2 x i32]) %return) {
// CHECK:STDOUT:   %tuple = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %1, align 4
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %2, align 4
// CHECK:STDOUT:   %array = alloca [2 x i32], align 4
// CHECK:STDOUT:   %array.element = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %array.element, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds [2 x i32], ptr %array, i32 0, i32 0
// CHECK:STDOUT:   store i32 %3, ptr %4, align 4
// CHECK:STDOUT:   %array.element1 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   %5 = load i32, ptr %array.element1, align 4
// CHECK:STDOUT:   %6 = getelementptr inbounds [2 x i32], ptr %array, i32 0, i32 1
// CHECK:STDOUT:   store i32 %5, ptr %6, align 4
// CHECK:STDOUT:   %7 = load [2 x i32], ptr %array, align 4
// CHECK:STDOUT:   store [2 x i32] %7, ptr %return, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Run() {
// CHECK:STDOUT:   %a = alloca [2 x i32], align 4
// CHECK:STDOUT:   %temp = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @A(ptr %temp)
// CHECK:STDOUT:   %array = alloca [2 x i32], align 4
// CHECK:STDOUT:   %array.element = getelementptr inbounds { i32, i32 }, ptr %temp, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %array.element, align 4
// CHECK:STDOUT:   %2 = getelementptr inbounds [2 x i32], ptr %array, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %2, align 4
// CHECK:STDOUT:   %array.element1 = getelementptr inbounds { i32, i32 }, ptr %temp, i32 0, i32 1
// CHECK:STDOUT:   %3 = load i32, ptr %array.element1, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds [2 x i32], ptr %array, i32 0, i32 1
// CHECK:STDOUT:   store i32 %3, ptr %4, align 4
// CHECK:STDOUT:   %5 = load [2 x i32], ptr %array, align 4
// CHECK:STDOUT:   store [2 x i32] %5, ptr %a, align 4
// CHECK:STDOUT:   %b = alloca i32, align 4
// CHECK:STDOUT:   %temp2 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @A(ptr %temp2)
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { i32, i32 }, ptr %temp2, i32 0, i32 0
// CHECK:STDOUT:   %6 = load i32, ptr %tuple.index, align 4
// CHECK:STDOUT:   store i32 %6, ptr %b, align 4
// CHECK:STDOUT:   %c = alloca i32, align 4
// CHECK:STDOUT:   %7 = load i32, ptr %b, align 4
// CHECK:STDOUT:   %array.index = getelementptr inbounds [2 x i32], ptr %a, i32 %7
// CHECK:STDOUT:   %8 = load i32, ptr %array.index, align 4
// CHECK:STDOUT:   store i32 %8, ptr %c, align 4
// CHECK:STDOUT:   %d = alloca i32, align 4
// CHECK:STDOUT:   %temp3 = alloca [2 x i32], align 4
// CHECK:STDOUT:   call void @B(ptr %temp3)
// CHECK:STDOUT:   %array.index4 = getelementptr inbounds [2 x i32], ptr %temp3, i32 0, i32 1
// CHECK:STDOUT:   %9 = load i32, ptr %array.index4, align 4
// CHECK:STDOUT:   store i32 %9, ptr %d, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
