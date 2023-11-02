// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(arr: [i32; 3], i: i32) -> i32 {
  return arr[i];
}

fn G() -> i32 {
  return F((1, 2, 3), 1);
}

// CHECK:STDOUT: ; ModuleID = 'function_param.carbon'
// CHECK:STDOUT: source_filename = "function_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %arr, i32 %i) {
// CHECK:STDOUT:   %array.index = getelementptr inbounds [3 x i32], ptr %arr, i32 0, i32 %i
// CHECK:STDOUT:   %1 = load i32, ptr %array.index, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @G() {
// CHECK:STDOUT:   %temp = alloca [3 x i32], align 4
// CHECK:STDOUT:   %array.index = getelementptr inbounds [3 x i32], ptr %temp, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %array.index, align 4
// CHECK:STDOUT:   %array.index1 = getelementptr inbounds [3 x i32], ptr %temp, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %array.index1, align 4
// CHECK:STDOUT:   %array.index2 = getelementptr inbounds [3 x i32], ptr %temp, i32 0, i32 2
// CHECK:STDOUT:   store i32 3, ptr %array.index2, align 4
// CHECK:STDOUT:   %F = call i32 @F(ptr %temp, i32 1)
// CHECK:STDOUT:   ret i32 %F
// CHECK:STDOUT: }
