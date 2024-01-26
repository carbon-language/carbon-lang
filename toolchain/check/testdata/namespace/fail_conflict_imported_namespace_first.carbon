// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- namespace.carbon

package Example library "namespace" api;

namespace NS;

// --- conflict.carbon

package Example api;

import library "namespace";

// CHECK:STDERR: conflict.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: namespace.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
fn NS();

fn NS.Foo();

// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc4} [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.2} [template]
// CHECK:STDOUT:   %import_ref: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %.2: <namespace> = namespace {.Foo = %Foo}, %import_ref [template]
// CHECK:STDOUT:   %.loc12: <function> = fn_decl @.1 [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo();
// CHECK:STDOUT:
