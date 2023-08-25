// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace Foo;

// Never called, just here to help catch bugs in name lookup.
fn Baz() {
}

fn Foo.Baz() {
}

fn Bar() {
  Foo.Baz();
}

// CHECK:STDOUT: ; ModuleID = 'function.carbon'
// CHECK:STDOUT: source_filename = "function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Baz() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Baz.1() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Bar() {
// CHECK:STDOUT:   call void @Baz.1()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
