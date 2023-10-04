// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(p: i32*);

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  G(&s.b);
}

// CHECK:STDOUT: ; ModuleID = 'address_of_field.carbon'
// CHECK:STDOUT: source_filename = "address_of_field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %s = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds { i32, i32 }, ptr %s, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4
// CHECK:STDOUT:   %b = getelementptr inbounds { i32, i32 }, ptr %s, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4
// CHECK:STDOUT:   %b1 = getelementptr inbounds { i32, i32 }, ptr %s, i32 0, i32 1
// CHECK:STDOUT:   call void @G(ptr %b1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
