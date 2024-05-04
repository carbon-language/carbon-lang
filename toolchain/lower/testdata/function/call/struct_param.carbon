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
// CHECK:STDOUT: @struct.5.loc10_4.3 = internal constant { i32, i32 } { i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F({ i32 } %b, ptr %c) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @F({ i32 } { i32 1 }, ptr @struct.5.loc10_4.3)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
