// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(ab: ((i32, i32, i32), (i32, i32, i32)));

fn F() {
  var a: (i32, i32, i32);
  var b: (i32, i32, i32);
  G((a, b));
}

// CHECK:STDOUT: ; ModuleID = 'value_formation.carbon'
// CHECK:STDOUT: source_filename = "value_formation.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %a = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %b = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 2
// CHECK:STDOUT:   %3 = load i32, ptr %tuple.elem2, align 4
// CHECK:STDOUT:   %tuple = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %4, align 4
// CHECK:STDOUT:   %5 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 %2, ptr %5, align 4
// CHECK:STDOUT:   %6 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 %3, ptr %6, align 4
// CHECK:STDOUT:   %tuple.elem3 = getelementptr inbounds { i32, i32, i32 }, ptr %b, i32 0, i32 0
// CHECK:STDOUT:   %7 = load i32, ptr %tuple.elem3, align 4
// CHECK:STDOUT:   %tuple.elem4 = getelementptr inbounds { i32, i32, i32 }, ptr %b, i32 0, i32 1
// CHECK:STDOUT:   %8 = load i32, ptr %tuple.elem4, align 4
// CHECK:STDOUT:   %tuple.elem5 = getelementptr inbounds { i32, i32, i32 }, ptr %b, i32 0, i32 2
// CHECK:STDOUT:   %9 = load i32, ptr %tuple.elem5, align 4
// CHECK:STDOUT:   %tuple6 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %10 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple6, i32 0, i32 0
// CHECK:STDOUT:   store i32 %7, ptr %10, align 4
// CHECK:STDOUT:   %11 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple6, i32 0, i32 1
// CHECK:STDOUT:   store i32 %8, ptr %11, align 4
// CHECK:STDOUT:   %12 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple6, i32 0, i32 2
// CHECK:STDOUT:   store i32 %9, ptr %12, align 4
// CHECK:STDOUT:   %tuple7 = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %13 = getelementptr inbounds { ptr, ptr }, ptr %tuple7, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple, ptr %13, align 8
// CHECK:STDOUT:   %14 = getelementptr inbounds { ptr, ptr }, ptr %tuple7, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple6, ptr %14, align 8
// CHECK:STDOUT:   call void @G(ptr %tuple7)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
