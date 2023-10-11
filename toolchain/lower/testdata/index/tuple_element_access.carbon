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
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT:   %a = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 0
// CHECK:STDOUT:   store i32 0, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 1
// CHECK:STDOUT:   store i32 1, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 2
// CHECK:STDOUT:   store i32 2, ptr %tuple.elem2, align 4
// CHECK:STDOUT:   %b = alloca i32, align 4
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.index, align 4
// CHECK:STDOUT:   store i32 %1, ptr %b, align 4
// CHECK:STDOUT:   %c = alloca i32, align 4
// CHECK:STDOUT:   %tuple.index3 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 2
// CHECK:STDOUT:   %2 = load i32, ptr %tuple.index3, align 4
// CHECK:STDOUT:   store i32 %2, ptr %c, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
