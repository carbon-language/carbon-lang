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

// CHECK:STDOUT: --- function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Baz.1: type = fn_type @Baz.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Baz = struct_value () [template]
// CHECK:STDOUT:   %Baz.2: type = fn_type @Baz.2 [template]
// CHECK:STDOUT:   %struct.2: Baz = struct_value () [template]
// CHECK:STDOUT:   %Bar: type = fn_type @Bar [template]
// CHECK:STDOUT:   %struct.3: Bar = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:     .Baz = %Baz.decl.loc10
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Foo: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Baz = %Baz.decl.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Baz.decl.loc10: Baz = fn_decl @Baz.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %Baz.decl.loc13: Baz = fn_decl @Baz.2 [template = constants.%struct.2] {}
// CHECK:STDOUT:   %Bar.decl: Bar = fn_decl @Bar [template = constants.%struct.3] {}
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
// CHECK:STDOUT:   %Foo.ref: <namespace> = name_ref Foo, file.%Foo [template = file.%Foo]
// CHECK:STDOUT:   %Baz.ref: Baz = name_ref Baz, file.%Baz.decl.loc13 [template = constants.%struct.2]
// CHECK:STDOUT:   %Baz.call: init () = call %Baz.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
