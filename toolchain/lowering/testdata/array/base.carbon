// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() {
  var a: [i32; 1] = (1,);
  var b: [f64; 2] = (11.1, 2.2,);
  var c: [(); 5] = ((), (), (), (), (),);
}

// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Run() {
// CHECK:STDOUT:   %var = alloca { i32 }, align 8
// CHECK:STDOUT:   %tuple = alloca { i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %1, align 4
// CHECK:STDOUT:   %array = alloca { i32 }, align 8
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32 }, ptr %array, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %2, align 4
// CHECK:STDOUT:   %3 = load { i32 }, ptr %array, align 4
// CHECK:STDOUT:   store { i32 } %3, ptr %var, align 4
// CHECK:STDOUT:   %var1 = alloca { double, double }, align 8
// CHECK:STDOUT:   %tuple2 = alloca { double, double }, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds { double, double }, ptr %tuple2, i32 0, i32 0
// CHECK:STDOUT:   store double 0x4026333333333334, ptr %4, align 8
// CHECK:STDOUT:   %5 = getelementptr inbounds { double, double }, ptr %tuple2, i32 0, i32 1
// CHECK:STDOUT:   store double 2.200000e+00, ptr %5, align 8
// CHECK:STDOUT:   %array3 = alloca { double, double }, align 8
// CHECK:STDOUT:   %6 = getelementptr inbounds { double, double }, ptr %array3, i32 0, i32 0
// CHECK:STDOUT:   store double 0x4026333333333334, ptr %6, align 8
// CHECK:STDOUT:   %7 = getelementptr inbounds { double, double }, ptr %array3, i32 0, i32 1
// CHECK:STDOUT:   store double 2.200000e+00, ptr %7, align 8
// CHECK:STDOUT:   %8 = load { double, double }, ptr %array3, align 8
// CHECK:STDOUT:   store { double, double } %8, ptr %var1, align 8
// CHECK:STDOUT:   %tuple4 = alloca {}, align 8
// CHECK:STDOUT:   %var5 = alloca { {}, {}, {}, {}, {} }, align 8
// CHECK:STDOUT:   %tuple6 = alloca {}, align 8
// CHECK:STDOUT:   %tuple7 = alloca {}, align 8
// CHECK:STDOUT:   %tuple8 = alloca {}, align 8
// CHECK:STDOUT:   %tuple9 = alloca {}, align 8
// CHECK:STDOUT:   %tuple10 = alloca {}, align 8
// CHECK:STDOUT:   %tuple11 = alloca { {}, {}, {}, {}, {} }, align 8
// CHECK:STDOUT:   %9 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple11, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple6, ptr %9, align 8
// CHECK:STDOUT:   %10 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple11, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple7, ptr %10, align 8
// CHECK:STDOUT:   %11 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple11, i32 0, i32 2
// CHECK:STDOUT:   store ptr %tuple8, ptr %11, align 8
// CHECK:STDOUT:   %12 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple11, i32 0, i32 3
// CHECK:STDOUT:   store ptr %tuple9, ptr %12, align 8
// CHECK:STDOUT:   %13 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple11, i32 0, i32 4
// CHECK:STDOUT:   store ptr %tuple10, ptr %13, align 8
// CHECK:STDOUT:   %array12 = alloca { {}, {}, {}, {}, {} }, align 8
// CHECK:STDOUT:   %14 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %array12, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple6, ptr %14, align 8
// CHECK:STDOUT:   %15 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %array12, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple7, ptr %15, align 8
// CHECK:STDOUT:   %16 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %array12, i32 0, i32 2
// CHECK:STDOUT:   store ptr %tuple8, ptr %16, align 8
// CHECK:STDOUT:   %17 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %array12, i32 0, i32 3
// CHECK:STDOUT:   store ptr %tuple9, ptr %17, align 8
// CHECK:STDOUT:   %18 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %array12, i32 0, i32 4
// CHECK:STDOUT:   store ptr %tuple10, ptr %18, align 8
// CHECK:STDOUT:   %19 = load { {}, {}, {}, {}, {} }, ptr %array12, align 1
// CHECK:STDOUT:   store { {}, {}, {}, {}, {} } %19, ptr %var5, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
