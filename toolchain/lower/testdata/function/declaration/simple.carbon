// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(n: i32);

fn G(n: i32) { F(n); }

// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @G(i32 %n) {
// CHECK:STDOUT:   call void @F(i32 %n)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
