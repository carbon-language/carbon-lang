// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fn.carbon

package Example library "fn" api;

fn Foo();

// --- var.carbon

package Example library "var" api;

// CHECK:STDERR: var.carbon:[[@LINE+6]]:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var Foo: i32;
// CHECK:STDERR:     ^~~
// CHECK:STDERR: fn.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn Foo();
// CHECK:STDERR: ^~~~~~~~~
var Foo: i32;

// --- conflict.carbon

package Example library "conflict" api;

import library "fn";
import library "var";

// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %Foo} [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %Foo} [template]
// CHECK:STDOUT:   %Foo.var: ref i32 = var Foo
// CHECK:STDOUT:   %Foo: ref i32 = bind_name Foo, %Foo.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %import_ref.1} [template]
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unused
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+2, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
