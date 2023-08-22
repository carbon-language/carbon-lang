// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: (i32, ) = (1, );
  var y: (i32, ) = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'one_entry.carbon'
// CHECK:STDOUT: source_filename = "one_entry.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %tuple = alloca { %type }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { %type }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %1, align 1
// CHECK:STDOUT:   %x = alloca { i32 }, align 8
// CHECK:STDOUT:   %tuple1 = alloca { i32 }, align 8
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32 }, ptr %tuple1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %2, align 4
// CHECK:STDOUT:   %3 = load { i32 }, ptr %tuple1, align 4
// CHECK:STDOUT:   store { i32 } %3, ptr %x, align 4
// CHECK:STDOUT:   %tuple2 = alloca { %type }, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds { %type }, ptr %tuple2, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %4, align 1
// CHECK:STDOUT:   %y = alloca { i32 }, align 8
// CHECK:STDOUT:   %5 = load { i32 }, ptr %x, align 4
// CHECK:STDOUT:   store { i32 } %5, ptr %y, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
