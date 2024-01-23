// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: fn_then_namespace.carbon: ERROR: Duplicate name being declared in the same scope.

// --- namespace.carbon

package Example library "namespace" api;

namespace NS;
fn NS.Foo() {}

// --- fn.carbon

package Example library "fn" api;

// CHECK:STDERR: fn.carbon:[[@LINE+3]]:1: Name is previously declared here.
// CHECK:STDERR: fn NS() {}
// CHECK:STDERR: ^~~~~~~~~
fn NS() {}

// --- fn_then_namespace.carbon

package Example library "fn_then_namespace" api;

import library "fn";
import library "namespace";

fn NS.Bar() {}

// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.NS = %.loc4} [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace NS, {.Foo = %Foo} [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.NS = %NS} [template]
// CHECK:STDOUT:   %NS: <function> = fn_decl @NS [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fn_then_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.NS = %.2} [template]
// CHECK:STDOUT:   %lazy_import_ref.1 = lazy_import_ref ir1, inst+1
// CHECK:STDOUT:   %.2: <namespace> = namespace NS, {.Foo = %lazy_import_ref.2, .Bar = %Bar} [template]
// CHECK:STDOUT:   %lazy_import_ref.2 = lazy_import_ref ir2, inst+2
// CHECK:STDOUT:   %Bar: <function> = fn_decl @Bar [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
