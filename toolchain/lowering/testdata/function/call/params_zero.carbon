// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'params_zero.carbon'
// CHECK:STDOUT: source_filename = "params_zero.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Foo() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Main() {
// CHECK:STDOUT:   %Foo = call i1 @Foo()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn Foo() {}

fn Main() {
  Foo();
}
