// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  var v: ((i32, i32, i32), (i32, i32, i32)) = (F(), F());
}

// CHECK:STDOUT: ; ModuleID = 'nested_tuple_in_place.carbon'
// CHECK:STDOUT: source_filename = "nested_tuple_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: define void @G() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { i32, i32, i32 }, { i32, i32, i32 } }, align 8
// CHECK:STDOUT:   %.loc10_56.1.tuple.elem = getelementptr inbounds { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v.var, i32 0, i32 0
// CHECK:STDOUT:   call void @F(ptr %.loc10_56.1.tuple.elem)
// CHECK:STDOUT:   %.loc10_56.2.tuple.elem = getelementptr inbounds { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v.var, i32 0, i32 1
// CHECK:STDOUT:   call void @F(ptr %.loc10_56.2.tuple.elem)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
