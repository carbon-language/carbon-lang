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
// CHECK:STDOUT:   %var = alloca [2 x i32], align 4
// CHECK:STDOUT:   %temp = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @A(ptr %temp)
// CHECK:STDOUT:   %array = alloca [2 x i32], align 4
// CHECK:STDOUT:   %1 = getelementptr inbounds [2 x i32], ptr %array, i32 0, i32 0
// CHECK:STDOUT:   store i32 undef, ptr %1, align 4
// CHECK:STDOUT:   %2 = getelementptr inbounds [2 x i32], ptr %array, i32 0, i32 1
// CHECK:STDOUT:   store i32 undef, ptr %2, align 4
// CHECK:STDOUT:   %3 = load [2 x i32], ptr %array, align 4
// CHECK:STDOUT:   store [2 x i32] %3, ptr %var, align 4
// CHECK:STDOUT:   %var1 = alloca i32, align 4
// CHECK:STDOUT:   %temp2 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @A(ptr %temp2)
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { i32, i32 }, ptr %temp2, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %tuple.index, align 4
// CHECK:STDOUT:   store i32 %4, ptr %var1, align 4
// CHECK:STDOUT:   %var3 = alloca i32, align 4
// CHECK:STDOUT:   %5 = load i32, ptr %var1, align 4
// CHECK:STDOUT:   %array.index = getelementptr inbounds [2 x i32], ptr %var, i32 %5
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4
// CHECK:STDOUT:   store i32 %6, ptr %var3, align 4
// CHECK:STDOUT:   %var4 = alloca i32, align 4
// CHECK:STDOUT:   %temp5 = alloca [2 x i32], align 4
// CHECK:STDOUT:   call void @B(ptr %temp5)
// CHECK:STDOUT:   %array.index6 = getelementptr inbounds [2 x i32], ptr %temp5, i32 0, i32 1
// CHECK:STDOUT:   %7 = load i32, ptr %array.index6, align 4
// CHECK:STDOUT:   store i32 %7, ptr %var4, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
