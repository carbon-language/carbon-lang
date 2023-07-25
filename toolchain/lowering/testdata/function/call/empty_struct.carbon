// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'empty_struct.carbon'
// CHECK:STDOUT: source_filename = "empty_struct.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %StructLiteralType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %StructLiteralType @Echo(%StructLiteralType %a) {
// CHECK:STDOUT:   %StructLiteralValue = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %1 = load %StructLiteralType, ptr %StructLiteralValue, align 1
// CHECK:STDOUT:   ret %StructLiteralType %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Main() {
// CHECK:STDOUT:   %StructLiteralValue = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %var = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %StructLiteralValue1 = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %1 = load %StructLiteralType, ptr %StructLiteralValue1, align 1
// CHECK:STDOUT:   %Echo = call %StructLiteralType @Echo(%StructLiteralType %1)
// CHECK:STDOUT:   store %StructLiteralType %Echo, ptr %var, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn Echo(a: {}) -> {} {
  return {};
}

fn Main() {
  var b: {} = Echo({});
}
