// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'nested_struct.carbon'
// CHECK:STDOUT: source_filename = "nested_struct.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %StructLiteralType.0 = type { {} }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %StructLiteralValue = alloca %StructLiteralType.0, align 8
// CHECK:STDOUT:   %b = getelementptr inbounds %StructLiteralType.0, ptr %StructLiteralValue, i32 0, i32 0
// CHECK:STDOUT:   store {} zeroinitializer, ptr %b, align 1
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

fn Run() -> i32 {
  {.b = {.a: i32}};
  return 0;
}
