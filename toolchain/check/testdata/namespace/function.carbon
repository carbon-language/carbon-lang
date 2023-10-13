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

// CHECK:STDOUT: file "function.carbon" {
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {.Baz = %Baz.loc13}
// CHECK:STDOUT:   %Baz.loc10: <function> = fn_decl @Baz.1
// CHECK:STDOUT:   %Baz.loc13: <function> = fn_decl @Baz.2
// CHECK:STDOUT:   %Bar: <function> = fn_decl @Bar
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
// CHECK:STDOUT:   %Foo.ref: <namespace> = name_reference "Foo", package.%.loc7
// CHECK:STDOUT:   %Baz.ref: <function> = name_reference "Baz", package.%Baz.loc13
// CHECK:STDOUT:   %.loc17_10.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc17_10.2: init () = call %Baz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
