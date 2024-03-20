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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Bar = %Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Wiz = %Wiz
// CHECK:STDOUT:     .Baz = %Baz
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Wiz: <function> = fn_decl @Wiz [template] {}
// CHECK:STDOUT:   %Baz: <function> = fn_decl @Baz [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Wiz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref: <namespace> = name_ref Foo, file.%Foo [template = file.%Foo]
// CHECK:STDOUT:   %Bar.ref: <namespace> = name_ref Bar, file.%Bar [template = file.%Bar]
// CHECK:STDOUT:   %Wiz.ref: <function> = name_ref Wiz, file.%Wiz [template = file.%Wiz]
// CHECK:STDOUT:   %.loc14: init () = call %Wiz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
