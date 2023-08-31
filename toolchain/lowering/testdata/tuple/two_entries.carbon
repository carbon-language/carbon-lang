// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: (i32, i32) = (12, 7);
  var y: (i32, i32) = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'two_entries.carbon'
// CHECK:STDOUT: source_filename = "two_entries.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %tuple = alloca { %type, %type }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { %type, %type }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %1, align 1
// CHECK:STDOUT:   %2 = getelementptr inbounds { %type, %type }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store %type zeroinitializer, ptr %2, align 1
// CHECK:STDOUT:   %x = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %tuple1 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %3 = getelementptr inbounds { i32, i32 }, ptr %tuple1, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %3, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds { i32, i32 }, ptr %tuple1, i32 0, i32 1
// CHECK:STDOUT:   store i32 7, ptr %4, align 4
// CHECK:STDOUT:   %tuple2 = alloca { %type, %type }, align 8
// CHECK:STDOUT:   %5 = getelementptr inbounds { %type, %type }, ptr %tuple2, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %5, align 1
// CHECK:STDOUT:   %6 = getelementptr inbounds { %type, %type }, ptr %tuple2, i32 0, i32 1
// CHECK:STDOUT:   store %type zeroinitializer, ptr %6, align 1
// CHECK:STDOUT:   %y = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
