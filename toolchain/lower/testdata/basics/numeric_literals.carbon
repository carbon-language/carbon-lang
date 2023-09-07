// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 4] = (
    8,
    9,
    0x8,
    0b1000,
  );
  var floats: [f64; 6] = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8,
  );
}

// CHECK:STDOUT: ; ModuleID = 'numeric_literals.carbon'
// CHECK:STDOUT: source_filename = "numeric_literals.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %ints = alloca [4 x i32], align 4
// CHECK:STDOUT:   %tuple = alloca { i32, i32, i32, i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %1, align 4
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %2, align 4
// CHECK:STDOUT:   %3 = getelementptr inbounds { i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 8, ptr %3, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds { i32, i32, i32, i32 }, ptr %tuple, i32 0, i32 3
// CHECK:STDOUT:   store i32 8, ptr %4, align 4
// CHECK:STDOUT:   %array.index = getelementptr inbounds [4 x i32], ptr %ints, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %array.index, align 4
// CHECK:STDOUT:   %array.index1 = getelementptr inbounds [4 x i32], ptr %ints, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %array.index1, align 4
// CHECK:STDOUT:   %array.index2 = getelementptr inbounds [4 x i32], ptr %ints, i32 0, i32 2
// CHECK:STDOUT:   store i32 8, ptr %array.index2, align 4
// CHECK:STDOUT:   %array.index3 = getelementptr inbounds [4 x i32], ptr %ints, i32 0, i32 3
// CHECK:STDOUT:   store i32 8, ptr %array.index3, align 4
// CHECK:STDOUT:   %floats = alloca [6 x double], align 8
// CHECK:STDOUT:   %tuple4 = alloca { double, double, double, double, double, double }, align 8
// CHECK:STDOUT:   %5 = getelementptr inbounds { double, double, double, double, double, double }, ptr %tuple4, i32 0, i32 0
// CHECK:STDOUT:   store double 9.000000e-01, ptr %5, align 8
// CHECK:STDOUT:   %6 = getelementptr inbounds { double, double, double, double, double, double }, ptr %tuple4, i32 0, i32 1
// CHECK:STDOUT:   store double 8.000000e+00, ptr %6, align 8
// CHECK:STDOUT:   %7 = getelementptr inbounds { double, double, double, double, double, double }, ptr %tuple4, i32 0, i32 2
// CHECK:STDOUT:   store double 8.000000e+01, ptr %7, align 8
// CHECK:STDOUT:   %8 = getelementptr inbounds { double, double, double, double, double, double }, ptr %tuple4, i32 0, i32 3
// CHECK:STDOUT:   store double 1.000000e+07, ptr %8, align 8
// CHECK:STDOUT:   %9 = getelementptr inbounds { double, double, double, double, double, double }, ptr %tuple4, i32 0, i32 4
// CHECK:STDOUT:   store double 1.000000e+08, ptr %9, align 8
// CHECK:STDOUT:   %10 = getelementptr inbounds { double, double, double, double, double, double }, ptr %tuple4, i32 0, i32 5
// CHECK:STDOUT:   store double 1.000000e-08, ptr %10, align 8
// CHECK:STDOUT:   %array.index5 = getelementptr inbounds [6 x double], ptr %floats, i32 0, i32 0
// CHECK:STDOUT:   store double 9.000000e-01, ptr %array.index5, align 8
// CHECK:STDOUT:   %array.index6 = getelementptr inbounds [6 x double], ptr %floats, i32 0, i32 1
// CHECK:STDOUT:   store double 8.000000e+00, ptr %array.index6, align 8
// CHECK:STDOUT:   %array.index7 = getelementptr inbounds [6 x double], ptr %floats, i32 0, i32 2
// CHECK:STDOUT:   store double 8.000000e+01, ptr %array.index7, align 8
// CHECK:STDOUT:   %array.index8 = getelementptr inbounds [6 x double], ptr %floats, i32 0, i32 3
// CHECK:STDOUT:   store double 1.000000e+07, ptr %array.index8, align 8
// CHECK:STDOUT:   %array.index9 = getelementptr inbounds [6 x double], ptr %floats, i32 0, i32 4
// CHECK:STDOUT:   store double 1.000000e+08, ptr %array.index9, align 8
// CHECK:STDOUT:   %array.index10 = getelementptr inbounds [6 x double], ptr %floats, i32 0, i32 5
// CHECK:STDOUT:   store double 1.000000e-08, ptr %array.index10, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
