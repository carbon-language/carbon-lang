// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var a: (i32, i32, i32) = (1, 2, 3);
  var b: (i32, i32) = (4, 5);
  let c: ((i32, i32, i32), (i32, i32)) = (a, b);
  return c[1][1];
}

// CHECK:STDOUT: ; ModuleID = 'tuple.carbon'
// CHECK:STDOUT: source_filename = "tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT:   %a = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 2
// CHECK:STDOUT:   store i32 3, ptr %tuple.elem2, align 4
// CHECK:STDOUT:   %b = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.elem3 = getelementptr inbounds { i32, i32 }, ptr %b, i32 0, i32 0
// CHECK:STDOUT:   store i32 4, ptr %tuple.elem3, align 4
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds { i32, i32 }, ptr %b, i32 0, i32 1
// CHECK:STDOUT:   store i32 5, ptr %tuple.elem4, align 4
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.elem5, align 4
// CHECK:STDOUT:   %tuple.elem6 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %tuple.elem6, align 4
// CHECK:STDOUT:   %tuple.elem7 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 2
// CHECK:STDOUT:   %3 = load i32, ptr %tuple.elem7, align 4
// CHECK:STDOUT:   %tuple = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %4, align 4
// CHECK:STDOUT:   %5 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 %2, ptr %5, align 4
// CHECK:STDOUT:   %6 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 %3, ptr %6, align 4
// CHECK:STDOUT:   %tuple.elem8 = getelementptr inbounds { i32, i32 }, ptr %b, i32 0, i32 0
// CHECK:STDOUT:   %7 = load i32, ptr %tuple.elem8, align 4
// CHECK:STDOUT:   %tuple.elem9 = getelementptr inbounds { i32, i32 }, ptr %b, i32 0, i32 1
// CHECK:STDOUT:   %8 = load i32, ptr %tuple.elem9, align 4
// CHECK:STDOUT:   %tuple10 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %9 = getelementptr inbounds { i32, i32 }, ptr %tuple10, i32 0, i32 0
// CHECK:STDOUT:   store i32 %7, ptr %9, align 4
// CHECK:STDOUT:   %10 = getelementptr inbounds { i32, i32 }, ptr %tuple10, i32 0, i32 1
// CHECK:STDOUT:   store i32 %8, ptr %10, align 4
// CHECK:STDOUT:   %tuple11 = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %11 = getelementptr inbounds { ptr, ptr }, ptr %tuple11, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple, ptr %11, align 8
// CHECK:STDOUT:   %12 = getelementptr inbounds { ptr, ptr }, ptr %tuple11, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple10, ptr %12, align 8
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { ptr, ptr }, ptr %tuple11, i32 0, i32 1
// CHECK:STDOUT:   %tuple.index.load = load ptr, ptr %tuple.index, align 8
// CHECK:STDOUT:   %tuple.index12 = getelementptr inbounds { i32, i32 }, ptr %tuple.index.load, i32 0, i32 1
// CHECK:STDOUT:   %tuple.index.load13 = load i32, ptr %tuple.index12, align 4
// CHECK:STDOUT:   ret i32 %tuple.index.load13
// CHECK:STDOUT: }
