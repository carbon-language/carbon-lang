// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fn.carbon

package Example library "fn" api;

fn NS() {}

// --- namespace.carbon

package Example library "namespace" api;

// CHECK:STDERR: namespace.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fn.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn NS() {}
// CHECK:STDERR: ^~~~~~~~~
namespace NS;
fn NS.Foo() {}

// --- conflict.carbon

package Example library "fn_then_namespace" api;

import library "fn";
import library "namespace";

fn NS.Bar() {}

// CHECK:STDOUT: --- fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %NS} [template]
// CHECK:STDOUT:   %NS: <function> = fn_decl @NS [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NS() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc10} [template]
// CHECK:STDOUT:   %.loc10: <namespace> = namespace {.Foo = %Foo} [template]
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.2} [template]
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unused
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir2, inst+1, used
// CHECK:STDOUT:   %.2: <namespace> = namespace {.Foo = %import_ref.3, .Bar = %Bar}, %import_ref.2 [template]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+2, unused
// CHECK:STDOUT:   %Bar: <function> = fn_decl @Bar [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
