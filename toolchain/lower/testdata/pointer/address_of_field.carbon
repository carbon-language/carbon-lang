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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc10_46.2.a = getelementptr inbounds { i32, i32 }, ptr %s.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %.loc10_46.2.a, align 4
// CHECK:STDOUT:   %.loc10_46.4.b = getelementptr inbounds { i32, i32 }, ptr %s.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %.loc10_46.4.b, align 4
// CHECK:STDOUT:   %.loc11_7.b = getelementptr inbounds { i32, i32 }, ptr %s.var, i32 0, i32 1
// CHECK:STDOUT:   call void @G(ptr %.loc11_7.b)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
