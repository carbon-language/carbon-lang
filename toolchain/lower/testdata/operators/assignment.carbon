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
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %a = alloca i32, align 4
// CHECK:STDOUT:   store i32 12, ptr %a, align 4
// CHECK:STDOUT:   store i32 9, ptr %a, align 4
// CHECK:STDOUT:   %tuple = alloca { %type, %type }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { %type, %type }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %1, align 1
// CHECK:STDOUT:   %2 = getelementptr inbounds { %type, %type }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store %type zeroinitializer, ptr %2, align 1
// CHECK:STDOUT:   %b = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %tuple1 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %3 = getelementptr inbounds { i32, i32 }, ptr %tuple1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %3, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds { i32, i32 }, ptr %tuple1, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %4, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
