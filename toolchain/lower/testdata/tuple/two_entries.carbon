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
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %x = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32 }, ptr %x, i32 0, i32 1
// CHECK:STDOUT:   store i32 7, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   %y = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
