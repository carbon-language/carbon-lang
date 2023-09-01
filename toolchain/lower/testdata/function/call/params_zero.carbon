// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo() {}

fn Main() {
  Foo();
}

// CHECK:STDOUT: ; ModuleID = 'params_zero.carbon'
// CHECK:STDOUT: source_filename = "params_zero.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Foo() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   call void @Foo()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
