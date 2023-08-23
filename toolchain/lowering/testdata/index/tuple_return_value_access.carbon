// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  var t: i32 = F()[1];
}

// CHECK:STDOUT: ; ModuleID = 'tuple_return_value_access.carbon'
// CHECK:STDOUT: source_filename = "tuple_return_value_access.carbon"
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
// CHECK:STDOUT:   %t = alloca i32, align 4
// CHECK:STDOUT:   %F = call { i32, i32 } @F()
// CHECK:STDOUT:   %tuple.index = extractvalue { i32, i32 } %F, 1
// CHECK:STDOUT:   store i32 %tuple.index, ptr %t, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
