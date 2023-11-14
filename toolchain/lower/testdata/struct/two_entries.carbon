// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  var y: {.a: i32, .b: i32} = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'two_entries.carbon'
// CHECK:STDOUT: source_filename = "two_entries.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT:   %x = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4
// CHECK:STDOUT:   %b = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4
// CHECK:STDOUT:   %y = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %a1 = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %a1, align 4
// CHECK:STDOUT:   %a2 = getelementptr inbounds { i32, i32 }, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %a2, align 4
// CHECK:STDOUT:   %b3 = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %b3, align 4
// CHECK:STDOUT:   %b4 = getelementptr inbounds { i32, i32 }, ptr %y, i32 0, i32 1
// CHECK:STDOUT:   store i32 %2, ptr %b4, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
