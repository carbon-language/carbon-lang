// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(ab: ((i32, i32, i32), (i32, i32, i32)));

fn F(a: (i32, i32, i32), b: (i32, i32, i32)) {
  G((a, b));
}

// CHECK:STDOUT: ; ModuleID = 'value_forwarding.carbon'
// CHECK:STDOUT: source_filename = "value_forwarding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F(ptr %a, ptr %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_10.2.tuple = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %.loc10_10.2 = getelementptr inbounds { ptr, ptr }, ptr %.loc10_10.2.tuple, i32 0, i32 0
// CHECK:STDOUT:   store ptr %a, ptr %.loc10_10.2, align 8
// CHECK:STDOUT:   %.loc10_10.21 = getelementptr inbounds { ptr, ptr }, ptr %.loc10_10.2.tuple, i32 0, i32 1
// CHECK:STDOUT:   store ptr %b, ptr %.loc10_10.21, align 8
// CHECK:STDOUT:   call void @G(ptr %.loc10_10.2.tuple)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
