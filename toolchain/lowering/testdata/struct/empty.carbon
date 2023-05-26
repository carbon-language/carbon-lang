// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'empty.carbon'
// CHECK:STDOUT: source_filename = "empty.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %StructLiteralType = type {}
// CHECK:STDOUT: %StructLiteralType.0 = type {}
// CHECK:STDOUT: %StructLiteralType.1 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %StructLiteralValue = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %var = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %StructLiteralValue1 = alloca %StructLiteralType.0, align 8
// CHECK:STDOUT:   store ptr %StructLiteralValue1, ptr %var, align 8
// CHECK:STDOUT:   %StructLiteralValue2 = alloca %StructLiteralType.1, align 8
// CHECK:STDOUT:   %var3 = alloca %StructLiteralType.1, align 8
// CHECK:STDOUT:   store ptr %var, ptr %var3, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  var x: {} = {};
  var y: {} = x;
  return 0;
}
