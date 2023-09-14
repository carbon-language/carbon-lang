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
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Run() {
// CHECK:STDOUT:   %a = alloca [1 x i32], align 4
// CHECK:STDOUT:   %tuple = alloca { i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %1, align 4
// CHECK:STDOUT:   %array.index = getelementptr inbounds [1 x i32], ptr %a, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %array.index, align 4
// CHECK:STDOUT:   %b = alloca [2 x double], align 8
// CHECK:STDOUT:   %tuple1 = alloca { double, double }, align 8
// CHECK:STDOUT:   %2 = getelementptr inbounds { double, double }, ptr %tuple1, i32 0, i32 0
// CHECK:STDOUT:   store double 0x4026333333333334, ptr %2, align 8
// CHECK:STDOUT:   %3 = getelementptr inbounds { double, double }, ptr %tuple1, i32 0, i32 1
// CHECK:STDOUT:   store double 2.200000e+00, ptr %3, align 8
// CHECK:STDOUT:   %array.index2 = getelementptr inbounds [2 x double], ptr %b, i32 0, i32 0
// CHECK:STDOUT:   store double 0x4026333333333334, ptr %array.index2, align 8
// CHECK:STDOUT:   %array.index3 = getelementptr inbounds [2 x double], ptr %b, i32 0, i32 1
// CHECK:STDOUT:   store double 2.200000e+00, ptr %array.index3, align 8
// CHECK:STDOUT:   %tuple4 = alloca {}, align 8
// CHECK:STDOUT:   %c = alloca [5 x {}], align 8
// CHECK:STDOUT:   %tuple5 = alloca {}, align 8
// CHECK:STDOUT:   %tuple6 = alloca {}, align 8
// CHECK:STDOUT:   %tuple7 = alloca {}, align 8
// CHECK:STDOUT:   %tuple8 = alloca {}, align 8
// CHECK:STDOUT:   %tuple9 = alloca {}, align 8
// CHECK:STDOUT:   %tuple10 = alloca { {}, {}, {}, {}, {} }, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple10, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple5, ptr %4, align 8
// CHECK:STDOUT:   %5 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple10, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple6, ptr %5, align 8
// CHECK:STDOUT:   %6 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple10, i32 0, i32 2
// CHECK:STDOUT:   store ptr %tuple7, ptr %6, align 8
// CHECK:STDOUT:   %7 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple10, i32 0, i32 3
// CHECK:STDOUT:   store ptr %tuple8, ptr %7, align 8
// CHECK:STDOUT:   %8 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple10, i32 0, i32 4
// CHECK:STDOUT:   store ptr %tuple9, ptr %8, align 8
// CHECK:STDOUT:   %array.index11 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 0
// CHECK:STDOUT:   %array.index12 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 1
// CHECK:STDOUT:   %array.index13 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 2
// CHECK:STDOUT:   %array.index14 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 3
// CHECK:STDOUT:   %array.index15 = getelementptr inbounds [5 x {}], ptr %c, i32 0, i32 4
// CHECK:STDOUT:   %tuple16 = alloca { %type, %type, %type }, align 8
// CHECK:STDOUT:   %9 = getelementptr inbounds { %type, %type, %type }, ptr %tuple16, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %9, align 1
// CHECK:STDOUT:   %10 = getelementptr inbounds { %type, %type, %type }, ptr %tuple16, i32 0, i32 1
// CHECK:STDOUT:   store %type zeroinitializer, ptr %10, align 1
// CHECK:STDOUT:   %11 = getelementptr inbounds { %type, %type, %type }, ptr %tuple16, i32 0, i32 2
// CHECK:STDOUT:   store %type zeroinitializer, ptr %11, align 1
// CHECK:STDOUT:   %d = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple17 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %12 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple17, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %12, align 4
// CHECK:STDOUT:   %13 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple17, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %13, align 4
// CHECK:STDOUT:   %14 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple17, i32 0, i32 2
// CHECK:STDOUT:   store i32 3, ptr %14, align 4
// CHECK:STDOUT:   %e = alloca [3 x i32], align 4
// CHECK:STDOUT:   %array.index18 = getelementptr inbounds [3 x i32], ptr %e, i32 0, i32 0
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 0
// CHECK:STDOUT:   %15 = load i32, ptr %tuple.index, align 4
// CHECK:STDOUT:   store i32 %15, ptr %array.index18, align 4
// CHECK:STDOUT:   %array.index19 = getelementptr inbounds [3 x i32], ptr %e, i32 0, i32 1
// CHECK:STDOUT:   %tuple.index20 = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 1
// CHECK:STDOUT:   %16 = load i32, ptr %tuple.index20, align 4
// CHECK:STDOUT:   store i32 %16, ptr %array.index19, align 4
// CHECK:STDOUT:   %array.index21 = getelementptr inbounds [3 x i32], ptr %e, i32 0, i32 2
// CHECK:STDOUT:   %tuple.index22 = getelementptr inbounds { i32, i32, i32 }, ptr %d, i32 0, i32 2
// CHECK:STDOUT:   %17 = load i32, ptr %tuple.index22, align 4
// CHECK:STDOUT:   store i32 %17, ptr %array.index21, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
