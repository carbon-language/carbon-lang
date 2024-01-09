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

// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc14: type = tuple_type (), const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %.loc7}
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {.Bar = %.loc8}
// CHECK:STDOUT:   %.loc8: <namespace> = namespace {.Wiz = %Wiz, .Baz = %Baz}
// CHECK:STDOUT:   %Wiz: <function> = fn_decl @Wiz, const
// CHECK:STDOUT:   %Baz: <function> = fn_decl @Baz, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Wiz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <namespace> = name_ref Foo, file.%.loc7
// CHECK:STDOUT:   %Bar.ref: <namespace> = name_ref Bar, file.%.loc8
// CHECK:STDOUT:   %Wiz.ref: <function> = name_ref Wiz, file.%Wiz, const = file.%Wiz
// CHECK:STDOUT:   %.loc14: init () = call %Wiz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
