// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: conflict.carbon: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: conflict.carbon: Name is previously declared here.

// --- namespace.carbon

package Example library "fn" api;

fn Foo();

// --- var.carbon

package Example library "var" api;

var Foo: i32;

// --- conflict.carbon

package Example library "conflict" api;

import library "fn";
import library "var";

// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Foo = %Foo} [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Foo = %Foo} [template]
// CHECK:STDOUT:   %Foo.var: ref i32 = var Foo
// CHECK:STDOUT:   %Foo: ref i32 = bind_name Foo, %Foo.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Foo = %lazy_import_ref.1} [template]
// CHECK:STDOUT:   %lazy_import_ref.1 = lazy_import_ref ir1, inst+1
// CHECK:STDOUT:   %lazy_import_ref.2 = lazy_import_ref ir2, inst+2
// CHECK:STDOUT: }
// CHECK:STDOUT:
