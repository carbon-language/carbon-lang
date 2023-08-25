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
// CHECK:STDOUT: define void @F({ i32 } %b, ptr %c) {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %tuple = alloca {}, align 8
// CHECK:STDOUT:   %tuple1 = alloca { i32 }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { i32 }, ptr %tuple1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %1, align 4
// CHECK:STDOUT:   %tuple2 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %2 = getelementptr inbounds { i32, i32 }, ptr %tuple2, i32 0, i32 0
// CHECK:STDOUT:   store i32 2, ptr %2, align 4
// CHECK:STDOUT:   %3 = getelementptr inbounds { i32, i32 }, ptr %tuple2, i32 0, i32 1
// CHECK:STDOUT:   store i32 3, ptr %3, align 4
// CHECK:STDOUT:   %4 = load { i32 }, ptr %tuple1, align 4
// CHECK:STDOUT:   call void @F({ i32 } %4, ptr %tuple2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
