// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
}

// CHECK:STDOUT: ; ModuleID = 'nested_struct_in_place.carbon'
// CHECK:STDOUT: source_filename = "nested_struct_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: define void @G() {
// CHECK:STDOUT:   %v = alloca { { i32, i32, i32 }, { i32, i32, i32 } }, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v, i32 0, i32 0
// CHECK:STDOUT:   call void @F(ptr %a)
// CHECK:STDOUT:   %b = getelementptr inbounds { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v, i32 0, i32 1
// CHECK:STDOUT:   call void @F(ptr %b)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
