// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo(a: i32) {}

fn Main() {
  Foo(1);
}

// CHECK:STDOUT: ; ModuleID = 'params_one.carbon'
// CHECK:STDOUT: source_filename = "params_one.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Foo(i32 %a) {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   call void @Foo(i32 1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
