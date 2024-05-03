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
// CHECK:STDOUT:   %Wiz: type = fn_type @Wiz [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Wiz = struct_value () [template]
// CHECK:STDOUT:   %Baz: type = fn_type @Baz [template]
// CHECK:STDOUT:   %struct.2: Baz = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Foo: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Bar = %Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Wiz = %Wiz.decl
// CHECK:STDOUT:     .Baz = %Baz.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Wiz.decl: Wiz = fn_decl @Wiz [template = constants.%struct.1] {}
// CHECK:STDOUT:   %Baz.decl: Baz = fn_decl @Baz [template = constants.%struct.2] {}
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
// CHECK:STDOUT:   %Wiz.ref: Wiz = name_ref Wiz, file.%Wiz.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Wiz.call: init () = call %Wiz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
