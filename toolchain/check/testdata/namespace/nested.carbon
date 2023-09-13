// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace Foo;
namespace Foo.Bar;

fn Foo.Bar.Wiz() {
}

fn Foo.Bar.Baz() {
  Foo.Bar.Wiz();
}

// CHECK:STDOUT: file "nested.carbon" {
// CHECK:STDOUT:   %.loc7 = namespace {.Bar = %.loc8}
// CHECK:STDOUT:   %.loc8 = namespace {.Wiz = %.loc10, .Baz = %.loc13}
// CHECK:STDOUT:   %.loc10 = fn_decl @Wiz
// CHECK:STDOUT:   %.loc13 = fn_decl @Baz
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Wiz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_14.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc14_14.2: init () = call @Wiz()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
