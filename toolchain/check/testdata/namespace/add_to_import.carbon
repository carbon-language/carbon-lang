// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

namespace NS;

// --- implicit.impl.carbon

package Implicit impl;

fn NS.A() -> i32 { return 0; }

var a: i32 = NS.A();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc4} [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.2, .a = %a} [template]
// CHECK:STDOUT:   %import_ref: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %.2: <namespace> = namespace {.A = %A}, %import_ref [template]
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %.2 [template = %.2]
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, %A [template = %A]
// CHECK:STDOUT:   %.loc6: init i32 = call %A.ref()
// CHECK:STDOUT:   assign %a.var, %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   return %.loc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
