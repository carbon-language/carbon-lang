// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'member_access.carbon'
// CHECK:STDOUT: source_filename = "member_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %StructLiteralType = type { double, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %var = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %StructLiteralValue = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds %StructLiteralType, ptr %StructLiteralValue, i32 0, i32 0
// CHECK:STDOUT:   store double 0.000000e+00, ptr %a, align 8
// CHECK:STDOUT:   %b = getelementptr inbounds %StructLiteralType, ptr %StructLiteralValue, i32 0, i32 1
// CHECK:STDOUT:   store i32 1, ptr %b, align 4
// CHECK:STDOUT:   %1 = load %StructLiteralType, ptr %StructLiteralValue, align 8
// CHECK:STDOUT:   store %StructLiteralType %1, ptr %var, align 8
// CHECK:STDOUT:   %var1 = alloca i32, align 4
// CHECK:STDOUT:   %b2 = getelementptr inbounds %StructLiteralType, ptr %var, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %b2, align 4
// CHECK:STDOUT:   store i32 %2, ptr %var1, align 4
// CHECK:STDOUT:   %var3 = alloca i32, align 4
// CHECK:STDOUT:   %3 = load i32, ptr %var1, align 4
// CHECK:STDOUT:   store i32 %3, ptr %var3, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  var y: i32 = x.b;
  var z: i32 = y;
  return 0;
}
