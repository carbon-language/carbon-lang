// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'empty.carbon'
// CHECK:STDOUT: source_filename = "empty.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %StructLiteralType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %StructLiteralValue = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %var = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %StructLiteralValue1 = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %0 = load %StructLiteralType, ptr %StructLiteralValue1, align 1
// CHECK:STDOUT:   store %StructLiteralType %0, ptr %var, align 1
// CHECK:STDOUT:   %StructLiteralValue2 = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %var3 = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %1 = load %StructLiteralType, ptr %var, align 1
// CHECK:STDOUT:   store %StructLiteralType %1, ptr %var3, align 1
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: {} = {};
  var y: {} = x;
  return 0;
}
