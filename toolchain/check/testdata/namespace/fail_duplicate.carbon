// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace Foo;

fn Foo.Baz() {
}

// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+6]]:1: ERROR: Redefinition of function Baz.
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE-6]]:1: Previously defined here.
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR: ^~~~~~~~~~~~~~
fn Foo.Baz() {
}

// CHECK:STDOUT: --- fail_duplicate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Baz = %Baz.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Baz.loc9: <function> = fn_decl @Baz [template] {}
// CHECK:STDOUT:   %Baz.loc18: <function> = fn_decl @Baz [template] {}
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
