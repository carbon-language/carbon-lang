// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace Foo;

fn Foo.Baz() {
}

// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+6]]:1: ERROR: Redefinition of `fn Baz`.
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE-6]]:1: Previously defined here.
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR: ^~~~~~~~~~~~~~
fn Foo.Baz() {
}

// CHECK:STDOUT: --- fail_duplicate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Baz: type = fn_type @Baz [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Baz = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Foo: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Baz = %Baz.decl.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Baz.decl.loc9: Baz = fn_decl @Baz [template = constants.%struct] {}
// CHECK:STDOUT:   %Baz.decl.loc18: Baz = fn_decl @Baz [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc19:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
