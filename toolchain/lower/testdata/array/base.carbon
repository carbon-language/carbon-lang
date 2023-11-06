// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() {
  var a: [i32; 1] = (1,);
  var b: [f64; 2] = (11.1, 2.2,);
  var c: [(); 5] = ((), (), (), (), (),);
  var d: (i32, i32, i32) = (1, 2, 3);
  var e: [i32; 3] = d;
}

// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() {
// CHECK:STDOUT:   %a = alloca [1 x i32], align 4
// CHECK:STDOUT:   %array.index = getelementptr inbounds [1 x i32], ptr %a, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %array.index, align 4
// CHECK:STDOUT:   %b = alloca [2 x double], align 8
// CHECK:STDOUT:   %array.index1 = getelementptr inbounds [2 x double], ptr %b, i32 0, i32 0
// CHECK:STDOUT:   store double 0x4026333333333334, ptr %array.index1, align 8
// CHECK:STDOUT:   %array.index2 = getelementptr inbounds [2 x double], ptr %b, i32 0, i32 1
// CHECK:STDOUT:   store double 2.200000e+00, ptr %array.index2, align 8
// CHECK:STDOUT:   %c = alloca [5 x {}], align 8
// CHECK:STDOUT:   %array.index3 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 0
// CHECK:STDOUT:   %array.index4 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 1
// CHECK:STDOUT:   %array.index5 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 2
// CHECK:STDOUT:   %array.index6 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 3
// CHECK:STDOUT:   %array.index7 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 4
// CHECK:STDOUT:   %d = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem8 = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %tuple.elem8, align 4
// CHECK:STDOUT:   %tuple.elem9 = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 2
// CHECK:STDOUT:   store i32 3, ptr %tuple.elem9, align 4
// CHECK:STDOUT:   %e = alloca [3 x i32], align 4
// CHECK:STDOUT:   %tuple.elem10 = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.elem10, align 4
// CHECK:STDOUT:   %array.index11 = getelementptr inbounds [3 x i32], ptr %e, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %array.index11, align 4
// CHECK:STDOUT:   %tuple.elem12 = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %tuple.elem12, align 4
// CHECK:STDOUT:   %array.index13 = getelementptr inbounds [3 x i32], ptr %e, i32 0, i32 1
// CHECK:STDOUT:   store i32 %2, ptr %array.index13, align 4
// CHECK:STDOUT:   %tuple.elem14 = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 2
// CHECK:STDOUT:   %3 = load i32, ptr %tuple.elem14, align 4
// CHECK:STDOUT:   %array.index15 = getelementptr inbounds [3 x i32], ptr %e, i32 0, i32 2
// CHECK:STDOUT:   store i32 %3, ptr %array.index15, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
