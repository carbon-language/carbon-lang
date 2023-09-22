// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  var v: [(i32, i32, i32); 2] = (F(), F());
}

// CHECK:STDOUT: ; ModuleID = 'array_in_place.carbon'
// CHECK:STDOUT: source_filename = "array_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: define void @G() {
// CHECK:STDOUT:   %v = alloca [2 x { i32, i32, i32 }], align 8
// CHECK:STDOUT:   %array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %v, i32 0, i32 0
// CHECK:STDOUT:   call void @F(ptr %array.index)
// CHECK:STDOUT:   %array.index1 = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %v, i32 0, i32 1
// CHECK:STDOUT:   call void @F(ptr %array.index1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
