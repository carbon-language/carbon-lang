// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'function.carbon'
// CHECK:STDOUT: source_filename = "function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Baz() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Baz.1() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @Bar() {
// CHECK:STDOUT:   %Baz.1 = call i1 @Baz.1()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

namespace Foo;

// Never called, just here to help catch bugs in name lookup.
fn Baz() {
}

fn Foo.Baz() {
}

fn Bar() {
  Foo.Baz();
}
