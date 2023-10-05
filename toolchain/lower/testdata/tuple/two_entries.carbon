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
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT:   %x = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 1
// CHECK:STDOUT:   store i32 7, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   %y = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.elem2, align 4
// CHECK:STDOUT:   %tuple.elem3 = getelementptr inbounds { i32, i32 }, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %tuple.elem3, align 4
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %tuple.elem4, align 4
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds { i32, i32 }, ptr %y, i32 0, i32 1
// CHECK:STDOUT:   store i32 %2, ptr %tuple.elem5, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
