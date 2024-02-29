// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

fn Foo();

// --- b.carbon

library "b" api;

import library "a";

// TODO: This should be an error until we can be clear whether `a` or `b` should
// own the definition, as there might be a separate definition in `a`'s impl.
// CHECK:STDERR: b.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Foo() {}
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: a.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn Foo();
// CHECK:STDERR: ^~~~~~~~~
fn Foo() {}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .Foo = %import_ref
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir1, inst+1, used [template = imports.%Foo]
// CHECK:STDOUT:   %.loc14: <function> = fn_decl @.1 {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
