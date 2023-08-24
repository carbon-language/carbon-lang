// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}

fn Main() {
  F({}, {.a = 1}, {.a = 2, .b = 3});
}

// CHECK:STDOUT: ; ModuleID = 'struct_param.carbon'
// CHECK:STDOUT: source_filename = "struct_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F({ i32 } %b, ptr %c) {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %struct = alloca {}, align 8
// CHECK:STDOUT:   %struct1 = alloca { i32 }, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds { i32 }, ptr %struct1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4
// CHECK:STDOUT:   %struct2 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %a3 = getelementptr inbounds { i32, i32 }, ptr %struct2, i32 0, i32 0
// CHECK:STDOUT:   store i32 2, ptr %a3, align 4
// CHECK:STDOUT:   %b = getelementptr inbounds { i32, i32 }, ptr %struct2, i32 0, i32 1
// CHECK:STDOUT:   store i32 3, ptr %b, align 4
// CHECK:STDOUT:   %1 = load { i32 }, ptr %struct1, align 4
// CHECK:STDOUT:   call void @F({ i32 } %1, ptr %struct2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
