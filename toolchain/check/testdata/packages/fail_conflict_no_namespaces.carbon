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

var Foo: i32;

// --- fail_conflict.carbon

package Example library "conflict" api;

import library "fn";
// CHECK:STDERR: fail_conflict.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import library "var";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: var.carbon:4:5: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: var Foo: i32;
// CHECK:STDERR:     ^~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "fn";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fn.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn Foo();
// CHECK:STDERR: ^~~~~~~~~
import library "var";

// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.var: ref i32 = var Foo
// CHECK:STDOUT:   %Foo: ref i32 = bind_name Foo, %Foo.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %import_ref.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%Foo]
// CHECK:STDOUT:   %import_ref.2: ref i32 = import_ref ir2, inst+2, used
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
