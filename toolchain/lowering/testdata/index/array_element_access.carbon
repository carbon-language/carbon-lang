// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
fn F() -> (i32, i32) { return (1, 2); }

fn Run() {
  var a: [i32; 2] = F();
  var b: i32 = a[0];
  var c: i32 = a[b];
}

// CHECK:STDOUT: ; ModuleID = 'array_element_access.carbon'
// CHECK:STDOUT: source_filename = "array_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define { i32, i32 } @F() {
// CHECK:STDOUT:   %tuple = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %1, align 4
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %2, align 4
// CHECK:STDOUT:   %3 = load { i32, i32 }, ptr %tuple, align 4
// CHECK:STDOUT:   ret { i32, i32 } %3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Run() {
// CHECK:STDOUT:   %var = alloca [2 x i32], align 4
// CHECK:STDOUT:   %F = call { i32, i32 } @F()
// CHECK:STDOUT:   %array = alloca [2 x i32], align 4
// CHECK:STDOUT:   %1 = extractvalue { i32, i32 } %F, 0
// CHECK:STDOUT:   %2 = getelementptr inbounds [2 x i32], ptr %array, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %2, align 4
// CHECK:STDOUT:   %3 = extractvalue { i32, i32 } %F, 1
// CHECK:STDOUT:   %4 = getelementptr inbounds [2 x i32], ptr %array, i32 0, i32 1
// CHECK:STDOUT:   store i32 %3, ptr %4, align 4
// CHECK:STDOUT:   %5 = load [2 x i32], ptr %array, align 4
// CHECK:STDOUT:   store [2 x i32] %5, ptr %var, align 4
// CHECK:STDOUT:   %var1 = alloca i32, align 4
// CHECK:STDOUT:   %array.index = getelementptr inbounds [2 x i32], ptr %var, i32 0, i32 0
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4
// CHECK:STDOUT:   store i32 %6, ptr %var1, align 4
// CHECK:STDOUT:   %var2 = alloca i32, align 4
// CHECK:STDOUT:   %7 = load i32, ptr %var1, align 4
// CHECK:STDOUT:   %array.index3 = zext i32 %7 to i64
// CHECK:STDOUT:   %8 = getelementptr inbounds [2 x i32], ptr %var, i64 %array.index3
// CHECK:STDOUT:   %9 = load i32, ptr %8, align 4
// CHECK:STDOUT:   store i32 %9, ptr %var2, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
