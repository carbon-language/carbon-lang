// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;

alias ns = NS;

fn NS.A() -> i32 { return 0; }

fn B() -> i32 { return ns.A(); }

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc7, .ns = %ns, .B = %B} [template]
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {.A = %A} [template]
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %.loc7 [template = %.loc7]
// CHECK:STDOUT:   %ns: <namespace> = bind_template_name ns, %NS.ref [template = %.loc7]
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template]
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   return %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ns.ref: <namespace> = name_ref ns, file.%ns [template = file.%.loc7]
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A [template = file.%A]
// CHECK:STDOUT:   %.loc13_28.1: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc13_30: i32 = value_of_initializer %.loc13_28.1
// CHECK:STDOUT:   %.loc13_28.2: i32 = converted %.loc13_28.1, %.loc13_30
// CHECK:STDOUT:   return %.loc13_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
