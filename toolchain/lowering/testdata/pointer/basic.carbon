// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(p: i32*);

fn F() {
  var n: i32 = 0;
  G(&n);
}

// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %var = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %var, align 4
// CHECK:STDOUT:   %1 = load ptr, ptr %var, align 8
// CHECK:STDOUT:   call void @G(ptr %1)
// CHECK:STDOUT:   %TupleLiteralValue = alloca {}, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
