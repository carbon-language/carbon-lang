// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 5] = (
    8,
    9,
    0x8,
    0b1000,
    39999999999999999993,
  );
  var floats: [f64; 7] = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8,
    39999999999999999993.0e39999999999999999993,
  );
}

// CHECK:STDOUT: ; ModuleID = 'numeric_literals.carbon'
// CHECK:STDOUT: source_filename = "numeric_literals.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %ints = alloca [5 x i32], align 4
// CHECK:STDOUT:   %tuple = alloca { i32, i32, i32, i32, i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %1, align 4
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %2, align 4
// CHECK:STDOUT:   %3 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 8, ptr %3, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 3
// CHECK:STDOUT:   store i32 8, ptr %4, align 4
// CHECK:STDOUT:   %5 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 4
// CHECK:STDOUT:   store i32 664797177, ptr %5, align 4
// CHECK:STDOUT:   %array = alloca [5 x i32], align 4
// CHECK:STDOUT:   %array.element = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   %6 = load i32, ptr %array.element, align 4
// CHECK:STDOUT:   %7 = getelementptr inbounds [5 x i32], ptr %array, i32 0, i32 0
// CHECK:STDOUT:   store i32 %6, ptr %7, align 4
// CHECK:STDOUT:   %array.element1 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   %8 = load i32, ptr %array.element1, align 4
// CHECK:STDOUT:   %9 = getelementptr inbounds [5 x i32], ptr %array, i32 0, i32 1
// CHECK:STDOUT:   store i32 %8, ptr %9, align 4
// CHECK:STDOUT:   %array.element2 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 2
// CHECK:STDOUT:   %10 = load i32, ptr %array.element2, align 4
// CHECK:STDOUT:   %11 = getelementptr inbounds [5 x i32], ptr %array, i32 0, i32 2
// CHECK:STDOUT:   store i32 %10, ptr %11, align 4
// CHECK:STDOUT:   %array.element3 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 3
// CHECK:STDOUT:   %12 = load i32, ptr %array.element3, align 4
// CHECK:STDOUT:   %13 = getelementptr inbounds [5 x i32], ptr %array, i32 0, i32 3
// CHECK:STDOUT:   store i32 %12, ptr %13, align 4
// CHECK:STDOUT:   %array.element4 = getelementptr inbounds { i32, i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 4
// CHECK:STDOUT:   %14 = load i32, ptr %array.element4, align 4
// CHECK:STDOUT:   %15 = getelementptr inbounds [5 x i32], ptr %array, i32 0, i32 4
// CHECK:STDOUT:   store i32 %14, ptr %15, align 4
// CHECK:STDOUT:   %floats = alloca [7 x double], align 8
// CHECK:STDOUT:   %tuple5 = alloca { double, double, double, double, double, double, double }, align 8
// CHECK:STDOUT:   %16 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 0
// CHECK:STDOUT:   store double 9.000000e-01, ptr %16, align 8
// CHECK:STDOUT:   %17 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 1
// CHECK:STDOUT:   store double 8.000000e+00, ptr %17, align 8
// CHECK:STDOUT:   %18 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 2
// CHECK:STDOUT:   store double 8.000000e+01, ptr %18, align 8
// CHECK:STDOUT:   %19 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 3
// CHECK:STDOUT:   store double 1.000000e+07, ptr %19, align 8
// CHECK:STDOUT:   %20 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 4
// CHECK:STDOUT:   store double 1.000000e+08, ptr %20, align 8
// CHECK:STDOUT:   %21 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 5
// CHECK:STDOUT:   store double 1.000000e-08, ptr %21, align 8
// CHECK:STDOUT:   %22 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 6
// CHECK:STDOUT:   store double 0x7FF0000000000000, ptr %22, align 8
// CHECK:STDOUT:   %array6 = alloca [7 x double], align 8
// CHECK:STDOUT:   %array.element7 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 0
// CHECK:STDOUT:   %23 = load double, ptr %array.element7, align 8
// CHECK:STDOUT:   %24 = getelementptr inbounds [7 x double], ptr %array6, i32 0, i32 0
// CHECK:STDOUT:   store double %23, ptr %24, align 8
// CHECK:STDOUT:   %array.element8 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 1
// CHECK:STDOUT:   %25 = load double, ptr %array.element8, align 8
// CHECK:STDOUT:   %26 = getelementptr inbounds [7 x double], ptr %array6, i32 0, i32 1
// CHECK:STDOUT:   store double %25, ptr %26, align 8
// CHECK:STDOUT:   %array.element9 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 2
// CHECK:STDOUT:   %27 = load double, ptr %array.element9, align 8
// CHECK:STDOUT:   %28 = getelementptr inbounds [7 x double], ptr %array6, i32 0, i32 2
// CHECK:STDOUT:   store double %27, ptr %28, align 8
// CHECK:STDOUT:   %array.element10 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 3
// CHECK:STDOUT:   %29 = load double, ptr %array.element10, align 8
// CHECK:STDOUT:   %30 = getelementptr inbounds [7 x double], ptr %array6, i32 0, i32 3
// CHECK:STDOUT:   store double %29, ptr %30, align 8
// CHECK:STDOUT:   %array.element11 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 4
// CHECK:STDOUT:   %31 = load double, ptr %array.element11, align 8
// CHECK:STDOUT:   %32 = getelementptr inbounds [7 x double], ptr %array6, i32 0, i32 4
// CHECK:STDOUT:   store double %31, ptr %32, align 8
// CHECK:STDOUT:   %array.element12 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 5
// CHECK:STDOUT:   %33 = load double, ptr %array.element12, align 8
// CHECK:STDOUT:   %34 = getelementptr inbounds [7 x double], ptr %array6, i32 0, i32 5
// CHECK:STDOUT:   store double %33, ptr %34, align 8
// CHECK:STDOUT:   %array.element13 = getelementptr inbounds { double, double, double, double, double, double, double }, ptr %tuple5, i32 0, i32 6
// CHECK:STDOUT:   %35 = load double, ptr %array.element13, align 8
// CHECK:STDOUT:   %36 = getelementptr inbounds [7 x double], ptr %array6, i32 0, i32 6
// CHECK:STDOUT:   store double %35, ptr %36, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
