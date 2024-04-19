// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  var t: [i32; 2] = F();
}

// CHECK:STDOUT: ; ModuleID = 'assign_return_value.carbon'
// CHECK:STDOUT: source_filename = "assign_return_value.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F(ptr sret({ i32, i32 }) %return) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_38.2.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %.loc7_38.2.tuple.elem, align 4
// CHECK:STDOUT:   %.loc7_38.4.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 24, ptr %.loc7_38.4.tuple.elem, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %t.var = alloca [2 x i32], align 4
// CHECK:STDOUT:   %.loc10_22.1.temp = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @F(ptr %.loc10_22.1.temp)
// CHECK:STDOUT:   %.loc10_22.3.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %.loc10_22.1.temp, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_22.4 = load i32, ptr %.loc10_22.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_22.6.array.index = getelementptr inbounds [2 x i32], ptr %t.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc10_22.4, ptr %.loc10_22.6.array.index, align 4
// CHECK:STDOUT:   %.loc10_22.8.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %.loc10_22.1.temp, i32 0, i32 1
// CHECK:STDOUT:   %.loc10_22.9 = load i32, ptr %.loc10_22.8.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_22.11.array.index = getelementptr inbounds [2 x i32], ptr %t.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc10_22.9, ptr %.loc10_22.11.array.index, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
