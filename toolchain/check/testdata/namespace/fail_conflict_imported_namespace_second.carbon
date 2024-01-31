// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fn.carbon

package Example library "fn" api;

fn NS();

// --- conflict.carbon

package Example api;

import library "fn";

// CHECK:STDERR: conflict.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fn.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
namespace NS;

// CHECK:STDERR: conflict.carbon:[[@LINE+6]]:7: ERROR: Name qualifiers are only allowed for entities that provide a scope.
// CHECK:STDERR: fn NS.Foo();
// CHECK:STDERR:       ^~~
// CHECK:STDERR: conflict.carbon:[[@LINE+3]]:4: Non-scope entity referenced here.
// CHECK:STDERR: fn NS.Foo();
// CHECK:STDERR:    ^~
fn NS.Foo();

// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %NS} [template]
// CHECK:STDOUT:   %NS: <function> = fn_decl @NS [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS();
// CHECK:STDOUT:
// CHECK:STDOUT: --- conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %import_ref} [template]
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %.loc12: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %.loc20: <function> = fn_decl @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
