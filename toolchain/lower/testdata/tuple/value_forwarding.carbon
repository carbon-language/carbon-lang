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
// CHECK:STDOUT:   %tuple = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds { ptr, ptr }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store ptr %a, ptr %1, align 8
// CHECK:STDOUT:   %2 = getelementptr inbounds { ptr, ptr }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store ptr %b, ptr %2, align 8
// CHECK:STDOUT:   call void @G(ptr %tuple)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
