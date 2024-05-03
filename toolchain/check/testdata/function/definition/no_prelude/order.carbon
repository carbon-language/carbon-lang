// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo() {}
fn Bar() {}
fn Baz() {}

// CHECK:STDOUT: --- order.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Foo: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Foo = struct_value () [template]
// CHECK:STDOUT:   %Bar: type = fn_type @Bar [template]
// CHECK:STDOUT:   %struct.2: Bar = struct_value () [template]
// CHECK:STDOUT:   %Baz: type = fn_type @Baz [template]
// CHECK:STDOUT:   %struct.3: Baz = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     .Baz = %Baz.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: Foo = fn_decl @Foo [template = constants.%struct.1] {}
// CHECK:STDOUT:   %Bar.decl: Bar = fn_decl @Bar [template = constants.%struct.2] {}
// CHECK:STDOUT:   %Baz.decl: Baz = fn_decl @Baz [template = constants.%struct.3] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
