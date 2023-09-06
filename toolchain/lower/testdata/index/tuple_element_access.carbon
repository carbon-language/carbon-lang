// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var a: (i32, i32, i32) = (0, 1, 2);
  var b: i32 = a[0];
  var c: i32 = a[2];
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'tuple_element_access.carbon'
// CHECK:STDOUT: source_filename = "tuple_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %tuple = alloca { %type, %type, %type }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { %type, %type, %type }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %1, align 1
// CHECK:STDOUT:   %2 = getelementptr inbounds { %type, %type, %type }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store %type zeroinitializer, ptr %2, align 1
// CHECK:STDOUT:   %3 = getelementptr inbounds { %type, %type, %type }, ptr %tuple, i32 0, i32 2
// CHECK:STDOUT:   store %type zeroinitializer, ptr %3, align 1
// CHECK:STDOUT:   %a = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple1 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple1, i32 0, i32 0
// CHECK:STDOUT:   store i32 0, ptr %4, align 4
// CHECK:STDOUT:   %5 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple1, i32 0, i32 1
// CHECK:STDOUT:   store i32 1, ptr %5, align 4
// CHECK:STDOUT:   %6 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple1, i32 0, i32 2
// CHECK:STDOUT:   store i32 2, ptr %6, align 4
// CHECK:STDOUT:   %b = alloca i32, align 4
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 0
// CHECK:STDOUT:   %7 = load i32, ptr %tuple.index, align 4
// CHECK:STDOUT:   store i32 %7, ptr %b, align 4
// CHECK:STDOUT:   %c = alloca i32, align 4
// CHECK:STDOUT:   %tuple.index2 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 2
// CHECK:STDOUT:   %8 = load i32, ptr %tuple.index2, align 4
// CHECK:STDOUT:   store i32 %8, ptr %c, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
