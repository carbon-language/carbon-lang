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
// CHECK:STDOUT: define { i32, i32 } @F() {
// CHECK:STDOUT:   %tuple = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %1, align 4
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 24, ptr %2, align 4
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
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
