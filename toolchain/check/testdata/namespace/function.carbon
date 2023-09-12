// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "function.carbon" {

namespace Foo;
// CHECK:STDOUT:   %.loc[[@LINE-1]] = namespace {.Baz = %.loc[[@LINE+7]]}

// Never called, just here to help catch bugs in name lookup.
fn Baz() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Baz.1
}

fn Foo.Baz() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Baz.2
}

fn Bar() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Bar
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar() {
// CHECK:STDOUT: !entry:
  Foo.Baz();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_10.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_10.2: init () = call @Baz.2()
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
