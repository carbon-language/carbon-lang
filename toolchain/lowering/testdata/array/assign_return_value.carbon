// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32,) { return (0,); }

fn Run() {
  var t: [i32; 1] = F();
}

// CHECK:STDOUT: ; ModuleID = 'assign_return_value.carbon'
// CHECK:STDOUT: source_filename = "assign_return_value.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define { i32 } @F() {
// CHECK:STDOUT:   %tuple = alloca { i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 0, ptr %1, align 4
// CHECK:STDOUT:   %2 = load { i32 }, ptr %tuple, align 4
// CHECK:STDOUT:   ret { i32 } %2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Run() {
// CHECK:STDOUT:   %var = alloca { i32 }, align 8
// CHECK:STDOUT:   %F = call { i32 } @F()
// CHECK:STDOUT:   %array = alloca { i32 }, align 8
// CHECK:STDOUT:   %1 = load { i32 }, ptr %array, align 4
// CHECK:STDOUT:   store { i32 } %1, ptr %var, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
