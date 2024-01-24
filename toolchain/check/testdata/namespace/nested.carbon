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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %.loc7} [template]
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {.Bar = %.loc8} [template]
// CHECK:STDOUT:   %.loc8: <namespace> = namespace {.Wiz = %Wiz, .Baz = %Baz} [template]
// CHECK:STDOUT:   %Wiz: <function> = fn_decl @Wiz [template]
// CHECK:STDOUT:   %Baz: <function> = fn_decl @Baz [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Wiz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <namespace> = name_ref Foo, file.%.loc7 [template = file.%.loc7]
// CHECK:STDOUT:   %Bar.ref: <namespace> = name_ref Bar, file.%.loc8 [template = file.%.loc8]
// CHECK:STDOUT:   %Wiz.ref: <function> = name_ref Wiz, file.%Wiz [template = file.%Wiz]
// CHECK:STDOUT:   %.loc14: init () = call %Wiz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
