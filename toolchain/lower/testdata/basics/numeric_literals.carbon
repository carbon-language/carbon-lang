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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ints.var = alloca [4 x i32], align 4
// CHECK:STDOUT:   %.loc15_3.3.array.index = getelementptr inbounds [4 x i32], ptr %ints.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %.loc15_3.3.array.index, align 4
// CHECK:STDOUT:   %.loc15_3.6.array.index = getelementptr inbounds [4 x i32], ptr %ints.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %.loc15_3.6.array.index, align 4
// CHECK:STDOUT:   %.loc15_3.9.array.index = getelementptr inbounds [4 x i32], ptr %ints.var, i32 0, i32 2
// CHECK:STDOUT:   store i32 8, ptr %.loc15_3.9.array.index, align 4
// CHECK:STDOUT:   %.loc15_3.12.array.index = getelementptr inbounds [4 x i32], ptr %ints.var, i32 0, i32 3
// CHECK:STDOUT:   store i32 8, ptr %.loc15_3.12.array.index, align 4
// CHECK:STDOUT:   %floats.var = alloca [6 x double], align 8
// CHECK:STDOUT:   %.loc23_3.3.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 0
// CHECK:STDOUT:   store double 9.000000e-01, ptr %.loc23_3.3.array.index, align 8
// CHECK:STDOUT:   %.loc23_3.6.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 1
// CHECK:STDOUT:   store double 8.000000e+00, ptr %.loc23_3.6.array.index, align 8
// CHECK:STDOUT:   %.loc23_3.9.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 2
// CHECK:STDOUT:   store double 8.000000e+01, ptr %.loc23_3.9.array.index, align 8
// CHECK:STDOUT:   %.loc23_3.12.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 3
// CHECK:STDOUT:   store double 1.000000e+07, ptr %.loc23_3.12.array.index, align 8
// CHECK:STDOUT:   %.loc23_3.15.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 4
// CHECK:STDOUT:   store double 1.000000e+08, ptr %.loc23_3.15.array.index, align 8
// CHECK:STDOUT:   %.loc23_3.18.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 5
// CHECK:STDOUT:   store double 1.000000e-08, ptr %.loc23_3.18.array.index, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
