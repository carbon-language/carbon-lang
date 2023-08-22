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
// CHECK:STDOUT:   %var = alloca [1 x i32], align 4
// CHECK:STDOUT:   %tuple = alloca { i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %1, align 4
// CHECK:STDOUT:   %array = alloca [1 x i32], align 4
// CHECK:STDOUT:   %array.element = getelementptr inbounds { i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   %2 = load i32, ptr %array.element, align 4
// CHECK:STDOUT:   %3 = getelementptr inbounds [1 x i32], ptr %array, i32 0, i32 0
// CHECK:STDOUT:   store i32 %2, ptr %3, align 4
// CHECK:STDOUT:   %4 = load [1 x i32], ptr %array, align 4
// CHECK:STDOUT:   store [1 x i32] %4, ptr %var, align 4
// CHECK:STDOUT:   %var1 = alloca [2 x double], align 8
// CHECK:STDOUT:   %tuple2 = alloca { double, double }, align 8
// CHECK:STDOUT:   %5 = getelementptr inbounds { double, double }, ptr %tuple2, i32 0, i32 0
// CHECK:STDOUT:   store double 0x4026333333333334, ptr %5, align 8
// CHECK:STDOUT:   %6 = getelementptr inbounds { double, double }, ptr %tuple2, i32 0, i32 1
// CHECK:STDOUT:   store double 2.200000e+00, ptr %6, align 8
// CHECK:STDOUT:   %array3 = alloca [2 x double], align 8
// CHECK:STDOUT:   %array.element4 = getelementptr inbounds { double, double }, ptr %tuple2, i32 0, i32 0
// CHECK:STDOUT:   %7 = load double, ptr %array.element4, align 8
// CHECK:STDOUT:   %8 = getelementptr inbounds [2 x double], ptr %array3, i32 0, i32 0
// CHECK:STDOUT:   store double %7, ptr %8, align 8
// CHECK:STDOUT:   %array.element5 = getelementptr inbounds { double, double }, ptr %tuple2, i32 0, i32 1
// CHECK:STDOUT:   %9 = load double, ptr %array.element5, align 8
// CHECK:STDOUT:   %10 = getelementptr inbounds [2 x double], ptr %array3, i32 0, i32 1
// CHECK:STDOUT:   store double %9, ptr %10, align 8
// CHECK:STDOUT:   %11 = load [2 x double], ptr %array3, align 8
// CHECK:STDOUT:   store [2 x double] %11, ptr %var1, align 8
// CHECK:STDOUT:   %tuple6 = alloca {}, align 8
// CHECK:STDOUT:   %var7 = alloca [5 x {}], align 8
// CHECK:STDOUT:   %tuple8 = alloca {}, align 8
// CHECK:STDOUT:   %tuple9 = alloca {}, align 8
// CHECK:STDOUT:   %tuple10 = alloca {}, align 8
// CHECK:STDOUT:   %tuple11 = alloca {}, align 8
// CHECK:STDOUT:   %tuple12 = alloca {}, align 8
// CHECK:STDOUT:   %tuple13 = alloca { {}, {}, {}, {}, {} }, align 8
// CHECK:STDOUT:   %12 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple8, ptr %12, align 8
// CHECK:STDOUT:   %13 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple9, ptr %13, align 8
// CHECK:STDOUT:   %14 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 2
// CHECK:STDOUT:   store ptr %tuple10, ptr %14, align 8
// CHECK:STDOUT:   %15 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 3
// CHECK:STDOUT:   store ptr %tuple11, ptr %15, align 8
// CHECK:STDOUT:   %16 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 4
// CHECK:STDOUT:   store ptr %tuple12, ptr %16, align 8
// CHECK:STDOUT:   %array14 = alloca [5 x {}], align 8
// CHECK:STDOUT:   %array.element15 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 0
// CHECK:STDOUT:   %17 = load {}, ptr %array.element15, align 1
// CHECK:STDOUT:   %18 = getelementptr inbounds [5 x {}], ptr %array14, i32 0, i32 0
// CHECK:STDOUT:   store {} %17, ptr %18, align 1
// CHECK:STDOUT:   %array.element16 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 1
// CHECK:STDOUT:   %19 = load {}, ptr %array.element16, align 1
// CHECK:STDOUT:   %20 = getelementptr inbounds [5 x {}], ptr %array14, i32 0, i32 1
// CHECK:STDOUT:   store {} %19, ptr %20, align 1
// CHECK:STDOUT:   %array.element17 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 2
// CHECK:STDOUT:   %21 = load {}, ptr %array.element17, align 1
// CHECK:STDOUT:   %22 = getelementptr inbounds [5 x {}], ptr %array14, i32 0, i32 2
// CHECK:STDOUT:   store {} %21, ptr %22, align 1
// CHECK:STDOUT:   %array.element18 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 3
// CHECK:STDOUT:   %23 = load {}, ptr %array.element18, align 1
// CHECK:STDOUT:   %24 = getelementptr inbounds [5 x {}], ptr %array14, i32 0, i32 3
// CHECK:STDOUT:   store {} %23, ptr %24, align 1
// CHECK:STDOUT:   %array.element19 = getelementptr inbounds { {}, {}, {}, {}, {} }, ptr %tuple13, i32 0, i32 4
// CHECK:STDOUT:   %25 = load {}, ptr %array.element19, align 1
// CHECK:STDOUT:   %26 = getelementptr inbounds [5 x {}], ptr %array14, i32 0, i32 4
// CHECK:STDOUT:   store {} %25, ptr %26, align 1
// CHECK:STDOUT:   %27 = load [5 x {}], ptr %array14, align 1
// CHECK:STDOUT:   store [5 x {}] %27, ptr %var7, align 1
// CHECK:STDOUT:   %tuple20 = alloca { %type, %type, %type }, align 8
// CHECK:STDOUT:   %28 = getelementptr inbounds { %type, %type, %type }, ptr %tuple20, i32 0, i32 0
// CHECK:STDOUT:   store %type zeroinitializer, ptr %28, align 1
// CHECK:STDOUT:   %29 = getelementptr inbounds { %type, %type, %type }, ptr %tuple20, i32 0, i32 1
// CHECK:STDOUT:   store %type zeroinitializer, ptr %29, align 1
// CHECK:STDOUT:   %30 = getelementptr inbounds { %type, %type, %type }, ptr %tuple20, i32 0, i32 2
// CHECK:STDOUT:   store %type zeroinitializer, ptr %30, align 1
// CHECK:STDOUT:   %var21 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple22 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %31 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple22, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %31, align 4
// CHECK:STDOUT:   %32 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple22, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %32, align 4
// CHECK:STDOUT:   %33 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple22, i32 0, i32 2
// CHECK:STDOUT:   store i32 3, ptr %33, align 4
// CHECK:STDOUT:   %34 = load { i32, i32, i32 }, ptr %tuple22, align 4
// CHECK:STDOUT:   store { i32, i32, i32 } %34, ptr %var21, align 4
// CHECK:STDOUT:   %var23 = alloca [3 x i32], align 4
// CHECK:STDOUT:   %array24 = alloca [3 x i32], align 4
// CHECK:STDOUT:   %array.element25 = getelementptr inbounds { i32, i32, i32 }, ptr %var21, i32 0, i32 0
// CHECK:STDOUT:   %35 = load i32, ptr %array.element25, align 4
// CHECK:STDOUT:   %36 = getelementptr inbounds [3 x i32], ptr %array24, i32 0, i32 0
// CHECK:STDOUT:   store i32 %35, ptr %36, align 4
// CHECK:STDOUT:   %array.element26 = getelementptr inbounds { i32, i32, i32 }, ptr %var21, i32 0, i32 1
// CHECK:STDOUT:   %37 = load i32, ptr %array.element26, align 4
// CHECK:STDOUT:   %38 = getelementptr inbounds [3 x i32], ptr %array24, i32 0, i32 1
// CHECK:STDOUT:   store i32 %37, ptr %38, align 4
// CHECK:STDOUT:   %array.element27 = getelementptr inbounds { i32, i32, i32 }, ptr %var21, i32 0, i32 2
// CHECK:STDOUT:   %39 = load i32, ptr %array.element27, align 4
// CHECK:STDOUT:   %40 = getelementptr inbounds [3 x i32], ptr %array24, i32 0, i32 2
// CHECK:STDOUT:   store i32 %39, ptr %40, align 4
// CHECK:STDOUT:   %41 = load [3 x i32], ptr %array24, align 4
// CHECK:STDOUT:   store [3 x i32] %41, ptr %var23, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
