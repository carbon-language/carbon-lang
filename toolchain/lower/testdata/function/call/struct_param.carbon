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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_34.2.struct = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc10_34.2 = getelementptr inbounds { i32, i32 }, ptr %.loc10_34.2.struct, i32 0, i32 0
// CHECK:STDOUT:   store i32 2, ptr %.loc10_34.2, align 4
// CHECK:STDOUT:   %.loc10_34.21 = getelementptr inbounds { i32, i32 }, ptr %.loc10_34.2.struct, i32 0, i32 1
// CHECK:STDOUT:   store i32 3, ptr %.loc10_34.21, align 4
// CHECK:STDOUT:   call void @F({ i32 } { i32 1 }, ptr %.loc10_34.2.struct)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 2, 0, 1 }
