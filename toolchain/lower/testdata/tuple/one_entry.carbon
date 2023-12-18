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
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT:   %x = alloca { i32 }, align 8
// CHECK:STDOUT:   store { i32 } { i32 1 }, ptr %x, align 4
// CHECK:STDOUT:   %y = alloca { i32 }, align 8
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32 }, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.init = insertvalue { i32 } poison, i32 %1, 0
// CHECK:STDOUT:   store { i32 } %tuple.init, ptr %y, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
