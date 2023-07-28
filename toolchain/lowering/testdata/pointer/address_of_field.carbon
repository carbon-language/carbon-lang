// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'address_of_field.carbon'
// CHECK:STDOUT: source_filename = "address_of_field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %StructLiteralType = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %var = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %StructLiteralValue = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds %StructLiteralType, ptr %StructLiteralValue, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4
// CHECK:STDOUT:   %b = getelementptr inbounds %StructLiteralType, ptr %StructLiteralValue, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4
// CHECK:STDOUT:   %1 = load %StructLiteralType, ptr %StructLiteralValue, align 4
// CHECK:STDOUT:   store %StructLiteralType %1, ptr %var, align 4
// CHECK:STDOUT:   %b1 = getelementptr inbounds %StructLiteralType, ptr %var, i32 0, i32 1
// CHECK:STDOUT:   %2 = load ptr, ptr %b1, align 8
// CHECK:STDOUT:   call void @G(ptr %2)
// CHECK:STDOUT:   %TupleLiteralValue = alloca {}, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn G(p: i32*);

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // TODO: The lowering here is incorrect: we're incorrectly loading `s.b`.
  G(&s.b);
}
