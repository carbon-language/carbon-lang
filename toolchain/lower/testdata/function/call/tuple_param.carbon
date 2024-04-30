// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: (), b: (i32,), c: (i32, i32)) {}

fn Main() {
  F((), (1,), (2, 3));
}

// CHECK:STDOUT: ; ModuleID = 'tuple_param.carbon'
// CHECK:STDOUT: source_filename = "tuple_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.3.loc10_4.3 = internal constant { i32, i32 } { i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F({ i32 } %b, ptr %c) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @F({ i32 } { i32 1 }, ptr @tuple.3.loc10_4.3)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
