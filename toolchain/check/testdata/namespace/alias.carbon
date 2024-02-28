// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;

alias ns = NS;

fn NS.A() -> i32 { return 0; }

fn B() -> i32 { return ns.A(); }

alias C = NS.A;

fn D() -> i32 { return C(); }

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc7, .ns = %ns, .B = %B, .C = %C, .D = %D} [template]
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {.A = %A} [template]
// CHECK:STDOUT:   %NS.ref.loc9: <namespace> = name_ref NS, %.loc7 [template = %.loc7]
// CHECK:STDOUT:   %ns: <namespace> = bind_alias ns, %.loc7 [template = %.loc7]
// CHECK:STDOUT:   %A: <function> = fn_decl @A {
// CHECK:STDOUT:     %return.var.loc11: ref i32 = var <return slot>
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %B: <function> = fn_decl @B {
// CHECK:STDOUT:     %return.var.loc13: ref i32 = var <return slot>
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %NS.ref.loc15: <namespace> = name_ref NS, %.loc7 [template = %.loc7]
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, %A [template = %A]
// CHECK:STDOUT:   %C: <function> = bind_alias C, %A [template = %A]
// CHECK:STDOUT:   %D: <function> = fn_decl @D {
// CHECK:STDOUT:     %return.var.loc17: ref i32 = var <return slot>
// CHECK:STDOUT:   } [template]
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
// CHECK:STDOUT: fn @D() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%C [template = file.%A]
// CHECK:STDOUT:   %.loc17_25.1: init i32 = call %C.ref()
// CHECK:STDOUT:   %.loc17_27: i32 = value_of_initializer %.loc17_25.1
// CHECK:STDOUT:   %.loc17_25.2: i32 = converted %.loc17_25.1, %.loc17_27
// CHECK:STDOUT:   return %.loc17_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
